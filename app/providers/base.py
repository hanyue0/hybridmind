"""推理后端抽象：本地与云端对上层完全同构，路由层才可能无感切换。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass
class Generation:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    cost_cny: float = 0.0


class Provider(Protocol):
    name: str

    def generate(
        self,
        messages: Sequence[dict],
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> Generation: ...


class ProviderError(RuntimeError):
    """统一的推理异常，便于路由层做降级与兜底。"""
