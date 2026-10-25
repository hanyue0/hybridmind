"""进程内本地推理（llama.cpp），用于单容器部署。

为什么需要它：开发时本地推理走 Ollama 守护进程，但多数免费托管平台只接受
**一个**容器、且不给常驻进程管理能力。把 Ollama 塞进镜像既笨重又容易起不来，
因此部署形态改为「模型文件 + 进程内推理」——容器里只有 FastAPI 和一个 GGUF。

接口与 OllamaProvider 完全一致，编排层不需要知道后端换了。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator, List, Sequence

from .base import Generation, ProviderError


class LlamaCppProvider:
    name = "local"

    def __init__(self, model_path: str | Path, n_ctx: int = 4096, n_threads: int | None = None) -> None:
        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.model_name = self.model_path.stem
        self._llm = None

    # ---------- 惰性加载：容器启动快，首个请求付一次加载成本 ----------

    def _ensure(self):
        if self._llm is not None:
            return self._llm
        if not self.model_path.exists():
            raise ProviderError(f"找不到模型文件：{self.model_path}")
        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover
            raise ProviderError("未安装 llama-cpp-python，无法使用进程内推理") from exc
        self._llm = Llama(
            model_path=str(self.model_path),
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            verbose=False,
        )
        return self._llm

    def generate(
        self,
        messages: Sequence[dict],
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> Generation:
        llm = self._ensure()
        started = time.perf_counter()
        try:
            result = llm.create_chat_completion(
                messages=list(messages),
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
            )
        except Exception as exc:
            raise ProviderError(f"本地推理失败：{exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000.0
        usage = result.get("usage") or {}
        return Generation(
            text=result["choices"][0]["message"]["content"],
            model=self.model_name,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            latency_ms=latency_ms,
            cost_cny=0.0,
        )

    def stream(
        self,
        messages: Sequence[dict],
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> Iterator[dict]:
        llm = self._ensure()
        started = time.perf_counter()
        prompt_tokens = completion_tokens = 0
        try:
            for chunk in llm.create_chat_completion(
                messages=list(messages),
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            ):
                delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                piece = delta.get("content")
                if piece:
                    completion_tokens += 1
                    yield {"type": "delta", "text": piece}
                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = int(usage.get("prompt_tokens") or 0)
                    completion_tokens = int(usage.get("completion_tokens") or 0)
        except Exception as exc:
            raise ProviderError(f"本地流式推理失败：{exc}") from exc

        if not prompt_tokens:
            prompt_tokens = sum(len(str(m.get("content", ""))) for m in messages) // 2
        yield {
            "type": "done",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
        }
