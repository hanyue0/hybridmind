"""云端推理后端：任何 OpenAI 兼容协议的服务都能直接接入。"""

from __future__ import annotations

import json
import time
from typing import Sequence

import httpx

from ..config import Price
from .base import Generation, ProviderError


class OpenAICompatProvider:
    name = "cloud"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        price: Price,
        timeout_s: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.price = price
        self.timeout_s = timeout_s

    def stream(
        self,
        messages: Sequence[dict],
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> Iterator[dict]:
        """逐 token 产出事件，供演示界面使用。

        这个方法同时服务两种后端：云端 API，以及部署形态下由 llama-server 提供的
        本地 OpenAI 兼容服务——两者协议一致，所以不需要各写一套。
        """
        payload = {
            "model": self.model,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        headers = {"Authorization": f"Bearer {self.api_key or 'local'}"}
        started = time.perf_counter()
        prompt_tokens = completion_tokens = 0
        try:
            with httpx.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout_s,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if body == "[DONE]":
                        break
                    chunk = json.loads(body)
                    choice = (chunk.get("choices") or [{}])[0]
                    piece = (choice.get("delta") or {}).get("content")
                    if piece:
                        completion_tokens += 1
                        yield {"type": "delta", "text": piece}
                    usage = chunk.get("usage")
                    if usage:
                        prompt_tokens = int(usage.get("prompt_tokens") or 0)
                        completion_tokens = int(usage.get("completion_tokens") or 0)
        except httpx.HTTPError as exc:
            raise ProviderError(f"流式推理失败：{exc}") from exc

        latency_ms = (time.perf_counter() - started) * 1000.0
        if not prompt_tokens:
            prompt_tokens = sum(len(str(m.get("content", ""))) for m in messages) // 2
        yield {
            "type": "done",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_ms": latency_ms,
        }

    def generate(
        self,
        messages: Sequence[dict],
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> Generation:
        if not self.api_key:
            raise ProviderError(
                "未配置 CLOUD_API_KEY：请在 .env 中填入云端密钥，或改用 --mode local。"
            )
        payload = {
            "model": self.model,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        started = time.perf_counter()
        try:
            resp = httpx.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout_s,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"云端返回 {exc.response.status_code}：{exc.response.text[:200]}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"云端推理调用失败：{exc}") from exc

        latency_ms = (time.perf_counter() - started) * 1000.0
        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content", "")
        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        cost = (
            prompt_tokens * self.price.input + completion_tokens * self.price.output
        ) / 1_000_000.0

        return Generation(
            text=text,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            cost_cny=cost,
        )
