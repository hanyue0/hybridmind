"""Web 层：演示界面 + 只读的统计接口。

刻意保持极简——所有花哨逻辑都在 pipeline 里，这一层只负责
把「路由决策」这件事可视化成评委一眼能看懂的界面。
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..config import settings
from ..pipeline import HybridPipeline, Ledger
from ..providers import OllamaProvider, OpenAICompatProvider, ProviderError
from ..schemas import ChatRequest, ChatResponse

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="HybridMind", version=__version__)
ledger = Ledger()

pipeline = HybridPipeline(
    local=OllamaProvider(settings.ollama_host, settings.local_model, timeout_s=settings.timeout_s),
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

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


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
