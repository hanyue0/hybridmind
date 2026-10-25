"""学习型路由：预测「本地模型能不能答对」，而不是猜「这题难不难」。

为什么换掉规则路由器：D0 的实测显示，字符串规则在 160 条评测集上给出的难度分
只在 0.13–0.33 之间浮动，且 tier4 的均值反而低于 tier3——它描述的是"题目怎么写"，
不是"模型会不会做错"。而后者才是路由真正需要的信息。

做法：把本地模型的真实作答结果当作标签（答对=1，答错=0），
用可解释的浅层特征训练逻辑回归，输出「本地答对概率 p」。
p 低于阈值就升级云端。这样阈值不再靠拍脑袋，而是对应一个可解释的风险水位：
"我愿意接受多少比例的本地答错"。

运行期推理不依赖 scikit-learn：训练完把系数导出成 JSON，服务端只用纯 Python 做点积。
scikit-learn 只在训练脚本里需要。
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from . import features as rule_features

FEATURE_NAMES: Sequence[str] = (
    "length",
    "num_numbers",
    "num_operators",
    "num_clauses",
    "num_sentences",
    "log_max_number",
    "has_percent",
    "question_marks",
    "math_terms",
    "word_problem_terms",
    "code_terms",
    "reasoning_terms",
    "multi_hop_terms",
    "arithmetic",
)

_OPERATOR = re.compile(r"[\+\-\*/×÷%]")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def extract_vector(text: str) -> List[float]:
    """提取特征向量。顺序必须与 FEATURE_NAMES 严格一致。"""
    rule = rule_features.extract(text)
    numbers = [float(m.group(0)) for m in _NUMBER.finditer(text)]
    return [
        float(rule.length),
        float(len(numbers)),
        float(len(_OPERATOR.findall(text))),
        float(text.count("，") + text.count(",")),
        float(text.count("。") + text.count("；") + text.count(";")),
        math.log1p(max(numbers)) if numbers else 0.0,
        1.0 if ("%" in text or "％" in text) else 0.0,
        float(rule.question_marks),
        float(rule.math_terms),
        float(rule.word_problem_terms),
        float(rule.code_terms),
        float(rule.reasoning_terms),
        float(rule.multi_hop_terms),
        1.0 if rule.arithmetic else 0.0,
    ]


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass
class LocalCorrectnessModel:
    """导出的逻辑回归：p(本地答对) = sigmoid( ((x-μ)/σ)·w + b )。"""

    feature_names: Sequence[str]
    mean: List[float]
    scale: List[float]
    coef: List[float]
    intercept: float
    threshold: float = 0.5
    auc: float = 0.0
    n_samples: int = 0

    def predict_proba(self, text: str) -> float:
        vector = extract_vector(text)
        total = self.intercept
        for value, mu, sigma, weight in zip(vector, self.mean, self.scale, self.coef):
            standardized = (value - mu) / sigma if sigma else 0.0
            total += standardized * weight
        return _sigmoid(total)

    def route(self, text: str) -> tuple[str, float]:
        probability = self.predict_proba(text)
        return ("local" if probability >= self.threshold else "cloud"), probability

    def to_dict(self) -> dict:
        return {
            "feature_names": list(self.feature_names),
            "mean": self.mean,
            "scale": self.scale,
            "coef": self.coef,
            "intercept": self.intercept,
            "threshold": self.threshold,
            "auc": self.auc,
            "n_samples": self.n_samples,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LocalCorrectnessModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**payload)
