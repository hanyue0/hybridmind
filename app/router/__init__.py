"""路由层：特征、策略、置信度，以及两种可切换的决策器工厂。"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from . import confidence, features, policy
from .classifier import LocalCorrectnessModel
from .policy import Decision, decide

Decider = Callable[[str, Optional[str]], Decision]


def build_decider(settings) -> Decider:
    """按配置返回决策函数。

    学习型模型文件缺失时安全回退到规则策略——部署时少一个文件不该导致服务起不来，
    但要在日志里说清楚，不能静默改变行为。
    """
    if getattr(settings, "router_mode", "rule") == "learned":
        path = Path(settings.router_model_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent.parent.parent / path
        if path.exists():
            model = LocalCorrectnessModel.load(path)
            return lambda text, force=None, _m=model: policy.decide_learned(text, _m, force)
        print(f"[router] 未找到模型文件 {path}，回退规则策略")
    theta = settings.theta_local
    return lambda text, force=None: decide(text, theta, force)


__all__ = [
    "confidence",
    "features",
    "policy",
    "Decision",
    "decide",
    "build_decider",
    "LocalCorrectnessModel",
    "Decider",
]
