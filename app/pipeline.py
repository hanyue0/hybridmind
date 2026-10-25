"""编排层：把「判难度 → 选后端 → 兜底重答 → 记账」串成一次请求。

这一层刻意保持薄——它只做决策与记账，不碰具体推理细节，
这样换模型、加后端都不需要动编排逻辑。

提供两个入口：
- `answer()`：一次性返回，评测脚本走这条，便于统计与复现；
- `answer_stream()`：逐 token 产出事件，演示界面走这条，
  因为本地 CPU 推理只有个位数 tok/s，一次性返回会让界面冻住十几秒。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence

from .config import Settings
from .providers.base import Generation, Provider, ProviderError
from .router import confidence as conf_mod
from .router import build_decider
from .schemas import ChatResponse, RouteTrace, Usage

SYSTEM_PROMPT = (
    "你是简洁的中文助手。默认不超过三句话。"
    "数学题只给最终数值和一行算式，不复述题干、不解释规则。"
    "被明确要求论证时才展开步骤。"
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
        self.decider = build_decider(settings)

    # ---------- 内部工具 ----------

    @staticmethod
    def _with_system(messages: Sequence[dict]) -> List[dict]:
        normalized = [dict(m) for m in messages]
        if not normalized or normalized[0].get("role") != "system":
            normalized.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        return normalized

    def _ask_local(self, messages: List[dict]) -> Generation:
        return self.local.generate(
            messages, max_tokens=self.settings.max_tokens, temperature=self.settings.temperature
        )

    def _ask_cloud(self, messages: List[dict]) -> Generation:
        return self.cloud.generate(
            messages, max_tokens=self.settings.max_tokens, temperature=self.settings.temperature
        )

    def _escalation_applies(self, decision) -> bool:
        """是否值得再问一次"你确定吗"。

        学习型路由已经给出"本地答对概率"，只有落在**不确定区间**时才值得再花一次
        自评开销。否则会出现「1+1 概率 0.99 却被自评拖去云端」这种明明有先验
        还要多花钱的情况——实测里真实发生过。
        """
        if not (self.settings.enable_escalation and self.settings.cloud_ready):
            return False
        if decision.score_kind == "p_local_correct":
            return decision.score < decision.threshold + self.settings.escalation_margin
        return True

    def _prepare(self, messages: Sequence[dict], mode: str):
        """决策与降级判断，两个入口共用。"""
        chat = self._with_system(messages)
        question = chat[-1].get("content", "")
        decision = self.decider(question, mode if mode != "auto" else None)
        reasons = list(decision.reasons)
        route = decision.route
        if route == "cloud" and not self.settings.cloud_ready:
            route = "local"
            reasons.append("云端密钥未配置 → 降级本地推理")
        return chat, decision, reasons, route

    # ---------- 一次性返回（评测） ----------

    def answer(self, messages: Sequence[dict], mode: str = "auto") -> ChatResponse:
        chat, decision, reasons, route = self._prepare(messages, mode)
        escalated = False
        confidence: Optional[float] = None

        try:
            generation = self._ask_cloud(chat) if route == "cloud" else self._ask_local(chat)
        except ProviderError as exc:
            fallback = "local" if route == "cloud" else "cloud"
            reasons.append(f"{route} 推理失败（{exc}），已降级 {fallback}")
            if fallback == "cloud" and not self.settings.cloud_ready:
                raise
            route = fallback
            generation = self._ask_cloud(chat) if route == "cloud" else self._ask_local(chat)

        if (
            route == "local"
            and self._escalation_applies(decision)
            and mode == "auto"
        ):
            confidence = conf_mod.estimate(self.local, chat, generation.text)
            if confidence is not None:
                reasons.append(
                    f"本地自评置信度 {confidence:.2f} / 阈值 {self.settings.tau_confidence:.2f}"
                )
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
                score_kind=decision.score_kind,
            ),
            usage=Usage(
                prompt_tokens=generation.prompt_tokens,
                completion_tokens=generation.completion_tokens,
                cost_cny=round(generation.cost_cny, 6),
            ),
        )

    # ---------- 流式（演示） ----------

    def answer_stream(self, messages: Sequence[dict], mode: str = "auto") -> Iterator[dict]:
        """产出事件流。前端按 type 分别处理：trace / delta / escalate / done / error。"""
        chat, decision, reasons, route = self._prepare(messages, mode)

        yield {
            "type": "trace",
            "route": route,
            "difficulty": decision.difficulty,
            "threshold": decision.threshold,
            "reasons": reasons,
        }

        start = 0.0
        text_parts: List[str] = []
        generation: Optional[Generation] = None
        escalated = False
        confidence: Optional[float] = None

        if route == "local":
            try:
                for event in self.local.stream(  # type: ignore[attr-defined]
                    chat,
                    max_tokens=self.settings.max_tokens,
                    temperature=self.settings.temperature,
                ):
                    if event["type"] == "delta":
                        text_parts.append(event["text"])
                        yield event
                    elif event["type"] == "done":
                        generation = Generation(
                            text="".join(text_parts),
                            model=self.settings.local_model,
                            prompt_tokens=event["prompt_tokens"],
                            completion_tokens=event["completion_tokens"],
                            latency_ms=event["latency_ms"],
                            cost_cny=0.0,
                        )
                        start = generation.latency_ms
            except ProviderError as exc:
                reasons.append(f"本地流式失败（{exc}），降级云端")
                yield {"type": "reasons", "reasons": reasons}
                route = "cloud"

        if route == "local" and generation is not None:
            if self._escalation_applies(decision) and mode == "auto":
                yield {"type": "status", "text": "本地作答完成，正在自评置信度…"}
                confidence = conf_mod.estimate(self.local, chat, generation.text, max_tokens=12)
                if confidence is not None:
                    reasons.append(
                        f"本地自评置信度 {confidence:.2f} / 阈值 {self.settings.tau_confidence:.2f}"
                    )
                if confidence is not None and confidence < self.settings.tau_confidence:
                    yield {
                        "type": "escalate",
                        "text": f"置信度 {confidence:.2f} 低于阈值 {self.settings.tau_confidence:.2f}，升级云端重答",
                    }
                    try:
                        cloud_gen = self._ask_cloud(chat)
                        text_parts = [cloud_gen.text]
                        yield {"type": "reset"}
                        yield {"type": "delta", "text": cloud_gen.text}
                        generation = cloud_gen
                        route = "cloud"
                        escalated = True
                        reasons.append("置信度不足 → 已升级云端重答")
                    except ProviderError as exc:
                        reasons.append(f"升级失败（{exc}），保留本地结果")

        if route == "cloud" and (generation is None or escalated):
            if escalated:
                pass  # 上面已生成
            else:
                try:
                    cloud_gen = self._ask_cloud(chat)
                except ProviderError as exc:
                    yield {"type": "error", "detail": str(exc)}
                    return
                text_parts = [cloud_gen.text]
                yield {"type": "delta", "text": cloud_gen.text}
                generation = cloud_gen

        if generation is None:
            yield {"type": "error", "detail": "推理未产出结果"}
            return

        generation.text = "".join(text_parts) if text_parts else generation.text
        self.ledger.record(route, generation, escalated)

        yield {
            "type": "done",
            "answer": generation.text.strip(),
            "model": generation.model,
            "trace": {
                "route": route,
                "difficulty": decision.difficulty,
                "threshold": decision.threshold,
                "reasons": reasons,
                "escalated": escalated,
                "confidence": confidence,
                "latency_ms": round(generation.latency_ms, 1),
                "score_kind": decision.score_kind,
            },
            "usage": {
                "prompt_tokens": generation.prompt_tokens,
                "completion_tokens": generation.completion_tokens,
                "cost_cny": round(generation.cost_cny, 6),
            },
        }
