"""Ollama 本地推理后端：零边际成本、数据不出本机。

同时提供一次性生成（评测用）与逐 token 流式（演示用）两种接口。
流式在这个项目里不是装饰：本地 CPU 推理只有个位数 tok/s，
一次性返回会让界面冻住十几秒，逐字输出才让"正在思考"可感知。
"""

from __future__ import annotations

import json
import time
from typing import Iterator, Sequence

import httpx

from .base import Generation, ProviderError


class OllamaProvider:
    name = "local"

    def __init__(self, host: str, model: str, timeout_s: float = 120.0) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def _payload(self, messages: Sequence[dict], max_tokens: int, temperature: float, stream: bool) -> dict:
        return {
            "model": self.model,
            "messages": list(messages),
            "stream": stream,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }

    def generate(
        self,
        messages: Sequence[dict],
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> Generation:
        payload = self._payload(messages, max_tokens, temperature, stream=False)
        started = time.perf_counter()
        try:
            resp = httpx.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout_s)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"本地推理调用失败：{exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000.0
        data = resp.json()
        return Generation(
            text=(data.get("message") or {}).get("content", ""),
            model=self.model,
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            completion_tokens=int(data.get("eval_count") or 0),
            latency_ms=latency_ms,
            cost_cny=0.0,
        )

    def stream(
        self,
        messages: Sequence[dict],
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> Iterator[dict]:
        """逐 token 产出事件：{"type": "delta", "text": ...} / {"type": "done", ...}。"""
        payload = self._payload(messages, max_tokens, temperature, stream=True)
        started = time.perf_counter()
        try:
            with httpx.stream(
                "POST", f"{self.host}/api/chat", json=payload, timeout=self.timeout_s
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    message = chunk.get("message") or {}
                    if message.get("content"):
                        yield {"type": "delta", "text": message["content"]}
                    if chunk.get("done"):
                        yield {
                            "type": "done",
                            "prompt_tokens": int(chunk.get("prompt_eval_count") or 0),
                            "completion_tokens": int(chunk.get("eval_count") or 0),
                            "latency_ms": (time.perf_counter() - started) * 1000.0,
                        }
        except httpx.HTTPError as exc:
            raise ProviderError(f"本地流式推理失败：{exc}") from exc
