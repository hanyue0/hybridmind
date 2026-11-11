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

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import __version__
from ..config import ROOT, settings
from ..pipeline import HybridPipeline, Ledger
from ..providers import OpenAICompatProvider, ProviderError, build_local_provider
from ..rag.anchor import extract_anchors, verify_anchors
from ..rag.parser import parse_pdf, render_page_png
from ..rag.prompt import build_messages
from ..rag.qa import PaperQA, PaperStore
from ..schemas import ChatRequest, ChatResponse

STATIC_DIR = Path(__file__).parent / "static"
PAPER_DIR = ROOT / "data" / "papers"
UPLOAD_DIR = PAPER_DIR / "uploads"
IMAGE_CACHE = PAPER_DIR / ".cache"

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
            items.append(
                {
                    "name": name,
                    "pages": pages,
                    "chunks": chunks,
                    "uploaded": base == UPLOAD_DIR,
                    "indexed_zh": bool(store.zh_map(name)),
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
    IMAGE_CACHE.mkdir(parents=True, exist_ok=True)
    cached = IMAGE_CACHE / f"{name}_p{page}_{dpi}.png"
    if not cached.exists():
        cached.write_bytes(render_page_png(pdf, page, dpi=dpi))
    return Response(content=cached.read_bytes(), media_type="image/png")


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
