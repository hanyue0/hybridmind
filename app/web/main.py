"""Web 层：论文精读演示界面 + 只读统计。

两个入口并存：
- ``/api/chat``：改造前的通用问答（算术题演示、旧评测仍在用），行为一行未改；
- ``/api/paper/*``：新增的论文精读入口，多出检索、锚点、证据校验三块数据。

刻意保持极简——所有逻辑在 ``app.rag.qa`` 与 ``app.pipeline`` 里，
这一层只负责把「路由决策 + 证据锚点」这两件事可视化。
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import __version__
from ..config import ROOT, settings
from ..pipeline import HybridPipeline, Ledger
from ..providers import OpenAICompatProvider, ProviderError, build_local_provider
from ..rag.anchor import extract_anchors, verify_anchors
from ..rag.indexer import build_zh_index, load_chunks, load_zh
from ..rag.parser import parse_pdf, render_page_png
from ..rag.prompt import build_messages
from ..rag.qa import PaperQA, PaperStore
from ..schemas import ChatRequest, ChatResponse

STATIC_DIR = Path(__file__).parent / "static"
PAPER_DIR = ROOT / "data" / "papers"
UPLOAD_DIR = PAPER_DIR / "uploads"
IMAGE_CACHE = PAPER_DIR / ".cache"

# 建索引是一项分钟级任务，不能占着一次 HTTP 请求（托管平台的网关通常 60 秒就断）。
# 这里用一个进程内字典记录进度，前端轮询 —— 单实例部署，够用且无需引入任务队列。
INDEX_JOBS: dict = {}

app = FastAPI(title="HybridMind", version=__version__)
ledger = Ledger()

pipeline = HybridPipeline(
    local=build_local_provider(settings),
    cloud=OpenAICompatProvider(
        settings.cloud_base_url,
        settings.cloud_api_key,
        settings.cloud_model,
        settings.price_of(settings.cloud_model),
        timeout_s=settings.timeout_s,
    ),
    settings=settings,
    ledger=ledger,
)

store = PaperStore(PAPER_DIR)
# 论文场景用单独的模型文件（18 维，含检索特征）；算术题那份 14 维模型不受影响
paper_settings = replace(settings, router_model_path=settings.router_model_path_paper)
paper_pipeline = HybridPipeline(
    local=build_local_provider(paper_settings),
    cloud=OpenAICompatProvider(
        settings.cloud_base_url,
        settings.cloud_api_key,
        settings.cloud_model,
        settings.price_of(settings.cloud_model),
        timeout_s=max(settings.timeout_s, 120),
    ),
    settings=paper_settings,
    ledger=ledger,
)
qa = PaperQA(paper_pipeline, store, top_k=5)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class PaperAsk(BaseModel):
    paper: str
    question: str
    mode: str = "auto"
    k: int = 5


def _safe_paper(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.\-]+", name or ""):
        raise HTTPException(status_code=400, detail="非法的论文名")
    for base in (PAPER_DIR, UPLOAD_DIR):
        if (base / f"{name}.pdf").exists():
            return name
    raise HTTPException(status_code=404, detail=f"论文 {name} 不存在")


def _pdf_path(name: str) -> Path:
    for base in (PAPER_DIR, UPLOAD_DIR):
        candidate = base / f"{name}.pdf"
        if candidate.exists():
            return candidate
    raise HTTPException(status_code=404, detail=f"论文 {name} 不存在")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/paper")
def paper_index() -> FileResponse:
    return FileResponse(STATIC_DIR / "paper.html")


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": __version__,
        "local_model": settings.local_model,
        "cloud_model": settings.cloud_model,
        "cloud_ready": settings.cloud_ready,
        "theta_local": settings.theta_local,
        "tau_confidence": settings.tau_confidence,
        "escalation": settings.enable_escalation,
        "router_mode": settings.router_mode,
    }


@app.get("/api/stats")
def stats() -> dict:
    return ledger.snapshot()


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    messages = [message.model_dump() for message in request.messages]
    try:
        return pipeline.answer(messages, mode=request.mode)
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


# ---------------- 论文精读 ----------------


@app.get("/api/papers")
def papers() -> dict:
    items = []
    for base in (PAPER_DIR, UPLOAD_DIR):
        if not base.exists():
            continue
        for pdf in sorted(base.glob("*.pdf")):
            name = pdf.stem
            try:
                meta = store.load(name)
                pages, chunks = meta.n_pages, len(meta.chunks)
            except Exception:  # 损坏的 PDF 不该让列表接口挂掉
                pages, chunks = None, None
            zh = store.zh_map(name)
            items.append(
                {
                    "name": name,
                    "pages": pages,
                    "chunks": chunks,
                    "uploaded": base == UPLOAD_DIR,
                    "indexed_zh": bool(zh),
                    "zh_coverage": round(len(zh) / max(1, chunks or 1), 3),
                }
            )
    return {"papers": items}


@app.post("/api/paper/ask")
def paper_ask(req: PaperAsk) -> dict:
    name = _safe_paper(req.paper)
    try:
        result = qa.ask(name, req.question, mode=req.mode, k=req.k)
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return result.to_dict()


@app.post("/api/paper/ask/stream")
def paper_ask_stream(req: PaperAsk) -> StreamingResponse:
    name = _safe_paper(req.paper)

    def event_source():
        try:
            # 门控、升级、锚点、校验全部在 PaperQA.ask_stream 里，
            # 这一层只负责把事件转成 SSE——保证流式与非流式走同一套判据
            for event in qa.ask_stream(name, req.question, mode=req.mode, k=req.k):
                yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
        except ProviderError as exc:
            yield "data: " + json.dumps(
                {"type": "error", "detail": str(exc)}, ensure_ascii=False
            ) + "\n\n"
        except Exception as exc:  # 兜底：任何异常都要让前端知道，而不是静默断流
            yield "data: " + json.dumps(
                {"type": "error", "detail": f"服务端异常：{exc}"}, ensure_ascii=False
            ) + "\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/paper/{name}/page/{page}.png")
def paper_page(name: str, page: int, dpi: int = 110) -> Response:
    name = _safe_paper(name)
    pdf = _pdf_path(name)
    # 缓存目录可能只读（托管平台的镜像层），写不进去就每次现渲染，不能因此 500
    cached = IMAGE_CACHE / f"{name}_p{page}_{dpi}.png"
    try:
        IMAGE_CACHE.mkdir(parents=True, exist_ok=True)
        if cached.exists():
            return Response(content=cached.read_bytes(), media_type="image/png")
        cached.write_bytes(render_page_png(pdf, page, dpi=dpi))
        return Response(content=cached.read_bytes(), media_type="image/png")
    except OSError:
        return Response(content=render_page_png(pdf, page, dpi=dpi), media_type="image/png")


@app.get("/api/paper/{name}/meta")
def paper_meta(name: str) -> dict:
    name = _safe_paper(name)
    paper = store.load(name)
    pdf = _pdf_path(name)
    import pymupdf

    doc = pymupdf.open(str(pdf))
    sizes = [[round(p.rect.width, 2), round(p.rect.height, 2)] for p in doc]
    doc.close()
    return {
        "name": name,
        "pages": paper.n_pages,
        "page_sizes": sizes,
        "indexed_zh": bool(store.zh_map(name)),
        "chunks": [
            {"id": c.id, "page": c.page, "section": c.section, "bbox": list(c.bbox)}
            for c in paper.chunks
        ],
    }


@app.post("/api/paper/{name}/index")
def paper_index(name: str, background: BackgroundTasks, limit: int = 0) -> dict:
    """给论文补中文索引（上传的 PDF 靠这个才能用中文检索）。

    分钟级任务放到后台执行、前端轮询进度：一次 HTTP 请求撑不到 1–2 分钟，
    托管平台的网关会先断掉连接。失败不影响英文提问（BM25 仍能命中英文原文）。
    """
    name = _safe_paper(name)
    if not settings.cloud_ready:
        raise HTTPException(status_code=400, detail="未配置云端密钥，无法生成中文索引")

    zh = load_zh(PAPER_DIR, name)
    pending = [c["id"] for c in load_chunks(PAPER_DIR, name) if not zh.get(c["id"])]
    total = len(pending[: limit or None]) if limit else len(pending)
    if total == 0:
        return {"status": "done", "name": name, "total": 0, "done": 0, "cost_cny": 0.0}

    INDEX_JOBS[name] = {"status": "running", "done": 0, "total": total, "cost_cny": 0.0}

    def run() -> None:
        try:
            result = build_zh_index(
                PAPER_DIR, name, pipeline.cloud, limit=limit or None,
                progress=lambda done, tot: INDEX_JOBS[name].update(done=done, total=tot),
            )
            store.invalidate(name)
            INDEX_JOBS[name] = {"status": "done", **result}
        except Exception as exc:  # 后台任务异常不能让整个服务挂掉
            INDEX_JOBS[name] = {"status": "error", "detail": str(exc),
                                **{k: v for k, v in INDEX_JOBS.get(name, {}).items() if k in ("done", "total")}}

    background.add_task(run)
    return {"status": "started", "name": name, "total": total}


@app.get("/api/paper/{name}/index/status")
def paper_index_status(name: str) -> dict:
    name = _safe_paper(name)
    return INDEX_JOBS.get(name, {"status": "idle"})


@app.post("/api/paper/upload")
async def paper_upload(file: UploadFile = File(...)) -> dict:
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只支持 PDF")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_.\-]+", "_", Path(file.filename).stem)[:60] or "upload"
    target = UPLOAD_DIR / f"{stem}.pdf"
    target.write_bytes(await file.read())
    paper = parse_pdf(target, name=stem)
    (UPLOAD_DIR / f"{stem}.chunks.json").write_text(
        json.dumps(paper.to_dict(), ensure_ascii=False), encoding="utf-8"
    )
    store._papers.pop(stem, None)
    return {
        "name": stem,
        "pages": paper.n_pages,
        "chunks": len(paper.chunks),
        "note": "已解析。中文检索需要建库（云端逐块翻译），在此之前英文提问可直接用。",
    }
