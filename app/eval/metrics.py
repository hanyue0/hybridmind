"""答案判分。

刻意**不用「大模型当裁判」**：那会让评测本身引入一个不可控变量，还额外产生费用，
评委也无法逐条复算。本作品的评测集全部是可机器判定的封闭答案
（数值题比数值，短答题比别名），保证「同一份数据 + 同一条命令 = 同一张表」。

数值题的判定口径：取预测文本中**最后一个**数字作为最终答案。
理由是大模型回答的惯例是先写步骤、最后给结论，取最后一个数比取第一个更接近
「模型认为的答案」；这条口径写进技术文档，避免评审时被质疑"挑规则"。
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List, Optional

_PUNCT = re.compile(r"[\s，,。.！!？?；;：:、\"'“”‘’（）()\[\]【】·—_/]+")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_PURE_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
# 「商 18，余数 4」这类回答里，余数是题面之外的附带量，不该被当成最终答案
_REMAINDER = re.compile(r"[，,；;]?\s*(?:余数|余|剩下)[^0-9]{0,6}\d+.*$")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower()
    return _PUNCT.sub("", text)


def strip_remainder(text: str) -> str:
    """去掉"余数/余…"从句。

    动机：问「商是多少」而模型答「商 18，余数 4」时，商是对的；
    若直接取最后一个数字会把 4 当成答案，造成假阴性。
    注意：只剥离从句，不改变商本身的判定——商答错时仍会判错。
    """
    return _REMAINDER.sub("", text or "")


def numbers_in(text: str) -> List[float]:
    if not text:
        return []
    cleaned = text.replace(",", "").replace("，", "")
    return [float(m.group(0)) for m in _NUMBER.finditer(cleaned)]


def last_number(text: str) -> Optional[float]:
    values = numbers_in(text)
    return values[-1] if values else None


def is_numeric_gold(gold: str) -> bool:
    return bool(_PURE_NUMBER.fullmatch((gold or "").strip()))


def is_correct(
    prediction: str,
    gold: str,
    aliases: Iterable[str] = (),
    tol: float = 1e-6,
) -> bool:
    """数值题比最后一个数值；短答题做别名包含匹配（任一别名命中即算对）。"""
    if is_numeric_gold(gold):
        predicted = last_number(strip_remainder(prediction))
        if predicted is None:
            return False
        return abs(predicted - float(gold.strip())) <= tol

    haystack = normalize(prediction)
    if not haystack:
        return False
    candidates = [normalize(gold)] + [normalize(a) for a in aliases if a]
    return any(candidate and candidate in haystack for candidate in candidates)


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight
