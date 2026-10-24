"""编排层：把「判难度 → 选后端 → 兜底重答 → 记账」串成一次请求。

这一层刻意保持薄——它只做决策与记账，不碰具体推理细节，
这样换模型、加后端都不需要动编排逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from .config import Settings
from .providers.base import Generation, Provider, ProviderError
from .router import confidence as conf_mod
from .router.policy import decide
from .schemas import ChatResponse, RouteTrace, Usage

SYSTEM_PROMPT = (
    "你是一个简洁、准确的中文助手。"
    "回答尽量精炼：先给结论，再给必要的步骤或依据，不要重复题干。"
)


@dataclass
class Ledger:
    """进程内的成本与流量账本，前端看板与评测报告都从这里取数。"""

    requests: int = 0
    local_requests: int = 0
    cloud_requests: int = 0
    escalations: int = 0
    cost_cny: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_latency_ms: float = 0.0

    def record(self, route: str, generation: Generation, escalated: bool) -> None:
        self.requests += 1
        self.total_latency_ms += generation.latency_ms
        self.prompt_tokens += generation.prompt_tokens
        self.completion_tokens += generation.completion_tokens
        if route == "cloud":
            self.cloud_requests += 1
            self.cost_cny += generation.cost_cny
        else:
            self.local_requests += 1
        if escalated:
            self.escalations += 1

    def snapshot(self) -> dict:
        local_ratio = self.local_requests / self.requests if self.requests else 0.0
        return {
            "requests": self.requests,
            "local_requests": self.local_requests,
            "cloud_requests": self.cloud_requests,
            "escalations": self.escalations,
            "local_ratio": round(local_ratio, 4),
            "cost_cny": round(self.cost_cny, 6),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "avg_latency_ms": round(self.total_latency_ms / self.requests, 1)
            if self.requests
            else 0.0,
        }


class HybridPipeline:
    def __init__(
        self,
        local: Provider,
        cloud: Provider,
        settings: Settings,
        ledger: Optional[Ledger] = None,
    ) -> None:
        self.local = local
        self.cloud = cloud
        self.settings = settings
        self.ledger = ledger or Ledger()

    # ---------- 内部工具 ----------

    @staticmethod
    def _with_system(messages: Sequence[dict]) -> List[dict]:
        normalized = [dict(m) for m in messages]
        if not normalized or normalized[0].get("role") != "system":
            normalized.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        return normalized

    def _ask_local(self, messages: List[dict]) -> Generation:
        return self.local.generate(
            messages,
            max_tokens=self.settings.max_tokens,
            temperature=self.settings.temperature,
        )

    def _ask_cloud(self, messages: List[dict]) -> Generation:
        return self.cloud.generate(
            messages,
            max_tokens=self.settings.max_tokens,
            temperature=self.settings.temperature,
        )

    # ---------- 主流程 ----------

    def answer(self, messages: Sequence[dict], mode: str = "auto") -> ChatResponse:
        chat = self._with_system(messages)
        question = chat[-1].get("content", "")
        decision = decide(question, self.settings.theta_local, force=mode if mode != "auto" else None)
        reasons = list(decision.reasons)
        route = decision.route
        escalated = False
        confidence: Optional[float] = None

        if route == "cloud" and not self.settings.cloud_ready:
            route = "local"
            reasons.append("云端密钥未配置 → 降级本地推理")

        try:
            generation = self._ask_cloud(chat) if route == "cloud" else self._ask_local(chat)
        except ProviderError as exc:
            fallback = "local" if route == "cloud" else "cloud"
            reasons.append(f"{route} 推理失败（{exc}），已降级 {fallback}")
            if fallback == "cloud" and not self.settings.cloud_ready:
                raise
            route = fallback
            generation = self._ask_cloud(chat) if route == "cloud" else self._ask_local(chat)

        # 置信度兜底：只在本地作答且开启开关时触发
        if (
            route == "local"
            and self.settings.enable_escalation
            and self.settings.cloud_ready
            and mode == "auto"
        ):
            confidence = conf_mod.estimate(self.local, chat, generation.text)
            if confidence is not None:
                reasons.append(f"本地自评置信度 {confidence:.2f} / 阈值 {self.settings.tau_confidence:.2f}")
                if confidence < self.settings.tau_confidence:
                    try:
                        generation = self._ask_cloud(chat)
                        route = "cloud"
                        escalated = True
                        reasons.append("置信度不足 → 升级云端重答")
                    except ProviderError as exc:
                        reasons.append(f"升级失败（{exc}），保留本地结果")

        self.ledger.record(route, generation, escalated)

        return ChatResponse(
            answer=generation.text.strip(),
            model=generation.model,
            trace=RouteTrace(
                route=route,  # type: ignore[arg-type]
                difficulty=decision.difficulty,
                threshold=decision.threshold,
                reasons=reasons,
                escalated=escalated,
                confidence=confidence,
                latency_ms=round(generation.latency_ms, 1),
            ),
            usage=Usage(
                prompt_tokens=generation.prompt_tokens,
                completion_tokens=generation.completion_tokens,
                cost_cny=round(generation.cost_cny, 6),
            ),
        )
