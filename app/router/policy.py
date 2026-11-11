"""路由决策：两种可切换的策略。

- `decide()`：规则策略。难度分 vs 阈值 theta，零成本、可解释、可复现。
- `decide_learned()`：学习型策略。预测「本地答对概率」p，p 低于阈值就上云。

两者返回同一个 `Decision` 结构，`score_kind` 说明分值语义——
规则的分数是"难度"（越高越难），学习型是"本地答对概率"（越低越该上云）。
两者方向相反，界面必须区分显示，否则会误导使用者。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from . import features as feat


def _rag_reasons(rag: Optional[dict]) -> tuple[float, List[str]]:
    """检索特征 → 可解释的难度增量。

    这三条规则是「路由能读检索结果」的最小实现，刻意保持线性可读：
    证据越分散（跨页多、覆盖率低），越该交给大模型综合。
    """
    if not rag:
        return 0.0, []
    top1 = float(rag.get("top1_score", 0.0))
    pages = float(rag.get("n_pages_hit", 0.0))
    cover = float(rag.get("coverage", 0.0))
    delta = 0.0
    reasons: List[str] = []
    if top1 < 0.35:
        delta += 0.18
        reasons.append(f"检索置信低（top1={top1:.2f}）")
    if pages >= 4:
        delta += 0.10
        reasons.append(f"证据跨 {int(pages)} 页，需综合")
    if cover < 0.5:
        delta += 0.12
        reasons.append(f"问题词覆盖率 {cover:.0%}，可能检索不到")
    if not reasons:
        reasons.append(f"检索置信高（top1={top1:.2f}，top1-top2={float(rag.get('score_gap', 0.0)):.2f}）")
    return delta, reasons


@dataclass
class Decision:
    route: str
    score: float
    threshold: float
    reasons: List[str] = field(default_factory=list)
    score_kind: str = "difficulty"

    @property
    def difficulty(self) -> float:
        """兼容旧字段名：评测脚本与前端早期版本读的是 difficulty。"""
        return self.score


def decide(text: str, theta: float, force: Optional[str] = None,
           rag: Optional[dict] = None) -> Decision:
    score, reasons = feat.difficulty(feat.extract(text))
    delta, rag_reasons = _rag_reasons(rag)
    if delta:
        score = round(min(max(score + delta, 0.0), 1.0), 4)
    reasons = reasons + rag_reasons

    if force in ("local", "cloud"):
        reasons = [f"手动指定 {'本地' if force == 'local' else '云端'}"] + reasons
        return Decision(route=force, score=score, threshold=theta, reasons=reasons)

    route = "local" if score < theta else "cloud"
    verdict = f"难度 {score:.2f} {'<' if route == 'local' else '>='} 阈值 {theta:.2f} → {route}"
    return Decision(route=route, score=score, threshold=theta, reasons=[verdict] + reasons)


def decide_learned(text: str, model, force: Optional[str] = None,
                   rag: Optional[dict] = None) -> Decision:
    """用训练好的模型判断本地是否会答错。

    额外附一行规则视角的难度分，作为可解释的旁证——
    评委可以据此看出两个策略在同一题上的判断差异。
    """
    # 没有检索特征时按旧签名调用：既兼容只实现 predict_proba(text) 的模型，
    # 也保证算术题这条路径一个字节都没变
    probability = model.predict_proba(text, rag) if rag else model.predict_proba(text)
    reference, ref_reasons = feat.difficulty(feat.extract(text))
    _, rag_reasons = _rag_reasons(rag)

    if force in ("local", "cloud"):
        return Decision(
            route=force,
            score=probability,
            threshold=model.threshold,
            reasons=[f"手动指定 {'本地' if force == 'local' else '云端'}", f"预测本地答对概率 {probability:.2f}"]
            + rag_reasons + ref_reasons,
            score_kind="p_local_correct",
        )

    route = "local" if probability >= model.threshold else "cloud"
    verdict = (
        f"预测本地答对概率 {probability:.2f} "
        f"{'≥' if route == 'local' else '<'} 阈值 {model.threshold:.2f} → {route}"
    )
    return Decision(
        route=route,
        score=probability,
        threshold=model.threshold,
        reasons=[verdict, f"参考难度分 {reference:.2f}（规则视角）"] + rag_reasons + ref_reasons,
        score_kind="p_local_correct",
    )
