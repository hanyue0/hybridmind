"""路由决策：难度分 vs 阈值。

阈值 theta 是整份方案里最关键的旋钮——
theta 越大，越多请求留在本地（省钱但可能答错）；theta 越小，越多请求升级云端
（答得准但花钱）。评测部分会扫一遍 theta 取值，画出成本-准确率曲线。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from . import features as feat


@dataclass
class Decision:
    route: str
    difficulty: float
    threshold: float
    reasons: List[str] = field(default_factory=list)


def decide(text: str, theta: float, force: Optional[str] = None) -> Decision:
    score, reasons = feat.difficulty(feat.extract(text))

    if force in ("local", "cloud"):
        reasons = [f"手动指定 {'本地' if force == 'local' else '云端'}"] + reasons
        return Decision(route=force, difficulty=score, threshold=theta, reasons=reasons)

    route = "local" if score < theta else "cloud"
    verdict = f"难度 {score:.2f} {'<' if route == 'local' else '>='} 阈值 {theta:.2f} → {route}"
    return Decision(route=route, difficulty=score, threshold=theta, reasons=[verdict] + reasons)
