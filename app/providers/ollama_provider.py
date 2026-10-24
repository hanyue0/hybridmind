"""Ollama 本地推理后端：零边际成本、数据不出本机。"""

from __future__ import annotations

import time
from typing import Sequence

import httpx

from .base import Generation, ProviderError


class OllamaProvider:
    name = "local"

    def __init__(self, host: str, model: str, timeout_s: float = 120.0) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def generate(
        self,
        messages: Sequence[dict],
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> Generation:
        payload = {
            "model": self.model,
            "messages": list(messages),
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        started = time.perf_counter()
        try:
            resp = httpx.post(
                f"{self.host}/api/chat", json=payload, timeout=self.timeout_s
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:  # 网络/服务不可用统一转成 ProviderError
            raise ProviderError(f"本地推理调用失败：{exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000.0
        data = resp.json()
        return Generation(
            text=(data.get("message") or {}).get("content", ""),
            model=self.model,
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            completion_tokens=int(data.get("eval_count") or 0),
            latency_ms=latency_ms,
            cost_cny=0.0,  # 本地推理无 API 费用，电费不计入 token 账本
        )
