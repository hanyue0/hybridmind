"""查询难度特征提取。

全部使用显式规则特征，不依赖外部模型，理由有三：

1. **零成本、零延迟**——路由器本身不能成为新瓶颈。若用大模型判断难度，等于每次请求
   先付一次云端费用，这个方案的立论就不成立了。
2. **可解释性由设计保证**——每个特征都有名字，路由理由可以直接展示给用户，而不是
   事后用另一个模型去"解释"一个黑箱决策。
3. **可复现**——同一输入在任何机器上得到同一分数，评测结果可被评委逐条复算。

v0.1 的一个教训（记录在 docs/ai-logs）：初版把「计算」「多少」当成数学特征，
结果「计算 87+8 是多少」和「计算这个积分」得到同样的分。这类**问句措辞**不是难度信号，
必须和**领域术语**分开，否则路由会把所有算术题都送去云端，方案的成本优势直接归零。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

# 领域术语：出现即说明问题本身有数学/算法结构
MATH_TERMS = (
    "方程", "求解", "积分", "导数", "概率", "排列", "组合", "面积",
    "体积", "比率", "余数", "最大公约", "最小公倍", "定理", "数列",
)
# 应用题语境：不直接等于难，但往往意味着多步
WORD_PROBLEM_TERMS = (
    "单价", "折扣", "涨价", "降价", "工程", "工人", "效率", "周长",
    "速度", "公里", "平均", "总价", "成本", "利润",
)
# 推理要求：任何语言风格下都强烈指向"小模型搞不定"
REASONING_TERMS = (
    "证明", "推导", "反证", "为什么", "解释", "分析", "比较", "评价",
    "方案", "设计", "步骤", "原理", "权衡", "优缺点", "论证", "说明", "阐述",
)
CODE_TERMS = (
    "代码", "函数", "报错", "调试", "实现", "算法", "复杂度", "重构",
    "python", "java", "sql", "接口",
)
# 并列/条件结构：多步推理的语言标记
MULTI_HOP_TERMS = ("然后", "并且", "接着", "再", "分别", "最后", "同时", "如果", "假设")

_ARITHMETIC = re.compile(r"\d+\s*[\+\-\*/×÷%]\s*\d+")


@dataclass(frozen=True)
class Features:
    length: int
    question_marks: int
    math_terms: int
    word_problem_terms: int
    code_terms: int
    reasoning_terms: int
    multi_hop_terms: int
    arithmetic: bool


def _count(text: str, words: tuple[str, ...]) -> int:
    lowered = text.lower()
    return sum(lowered.count(word.lower()) for word in words)


def extract(text: str) -> Features:
    return Features(
        length=len(text),
        question_marks=text.count("?") + text.count("？"),
        math_terms=_count(text, MATH_TERMS),
        word_problem_terms=_count(text, WORD_PROBLEM_TERMS),
        code_terms=_count(text, CODE_TERMS),
        reasoning_terms=_count(text, REASONING_TERMS),
        multi_hop_terms=_count(text, MULTI_HOP_TERMS),
        arithmetic=bool(_ARITHMETIC.search(text)),
    )


def difficulty(features: Features) -> Tuple[float, List[str]]:
    """把特征折算成 0~1 的难度分，并同步产出人类可读的理由。"""
    score = 0.10
    reasons: List[str] = []

    if features.math_terms:
        score += 0.28 + 0.05 * min(features.math_terms - 1, 2)
        reasons.append(f"命中数学领域术语 ×{features.math_terms}")

    if features.reasoning_terms:
        # 推理要求是强信号：篇幅短的证明题一样很难
        score += 0.20 + 0.08 * min(features.reasoning_terms - 1, 3)
        reasons.append(f"命中推理/论证要求 ×{features.reasoning_terms}")

    if features.code_terms:
        score += 0.25 + 0.05 * min(features.code_terms - 1, 2)
        reasons.append(f"命中代码/算法特征 ×{features.code_terms}")

    if features.word_problem_terms:
        score += 0.12 + 0.03 * min(features.word_problem_terms - 1, 2)
        reasons.append(f"命中应用题语境 ×{features.word_problem_terms}")

    if features.arithmetic:
        # 单步算式本身不是难题，只给弱信号
        score += 0.08
        reasons.append("含算式表达式")

    length_ratio = min(features.length / 300.0, 1.0)
    if length_ratio > 0.25:
        score += 0.20 * length_ratio
        reasons.append(f"文本长度 {features.length} 字")

    if features.multi_hop_terms:
        score += 0.10 * min(features.multi_hop_terms, 2)
        reasons.append(f"并列/条件结构 ×{features.multi_hop_terms}")

    if features.question_marks >= 2:
        score += 0.05
        reasons.append(f"多问句 ×{features.question_marks}")

    score = round(min(max(score, 0.0), 1.0), 4)
    if not reasons:
        reasons.append("未命中高难度特征，判为轻量问答")
    return score, reasons
