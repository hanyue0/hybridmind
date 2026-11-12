"""论文问答的评测指标。

与 ``metrics.py`` 的关系：答案对错沿用同一套确定性判分（数字比最后一位、
短答比别名），**不引入任何第二个模型当裁判**。本模块只增加三件论文场景才有的事：

1. **证据召回**：答案引用的锚点有没有落在 gold page 上；
2. **引用有效性**：答案里出现的 [S编号] 是否真的存在于检索结果里（幻觉引用）；
3. **数字/术语可溯性**：结论里的数字、英文术语能不能在引用的原文块里找到。

判分前先剥掉 [S3] 这类编号，否则「最后一个数字」会把编号当成答案——
这是实测踩到的坑，写在这里防止复现。
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence

from .metrics import is_correct, is_numeric_gold, normalize

_CITE = re.compile(r"\[\s*S\d+(?:\s*[,，]\s*S\d+)*\s*\]")
_REFUSE = ("原文未提及", "未提及", "无法回答", "不足以回答", "没有提到", "片段中没有")


def strip_citations(text: str) -> str:
    return _CITE.sub("", text or "")


def numeric_present(prediction: str, gold: str) -> bool:
    """数值题判分：gold 数值是否作为一个完整数词出现在答案里。

    为什么不用 ``metrics.py`` 的「取最后一个数字」：那条规则的依据是"算式题的
    结论写在最后"，对**抽取式**问答不成立。实测踩到的反例——问"哪一年 12 月的
    dump"，正确答案是「2016年12月的…」，取最后一个数字会得到 12，
    把对的判成错的。改用子串匹配（前后不能紧挨数字）后该题正确。

    千分位要先抹掉：原文写 12,288、答案写 12288 是同一件事。
    """
    body = re.sub(r"(?<=\d)[,，](?=\d{3}(?!\d))", "", prediction)
    pattern = r"(?<![\d.])" + re.escape(gold.strip()) + r"(?![\d])"
    return re.search(pattern, body) is not None


def answer_correct(prediction: str, gold: str, aliases: Iterable[str] = ()) -> bool:
    body = strip_citations(prediction)
    if is_numeric_gold(gold):
        return numeric_present(body, gold)
    return is_correct(body, gold, aliases)


def evidence_hit(anchor_pages: Sequence[int], gold_pages: Sequence[int], tol: int = 0) -> bool:
    if not anchor_pages or not gold_pages:
        return False
    return any(abs(a - g) <= tol for a in anchor_pages for g in gold_pages)


def is_refusal(answer: str) -> bool:
    return any(m in (answer or "") for m in _REFUSE)


def _rate(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0


def summarize(records: Sequence[dict]) -> Dict[str, object]:
    """把逐条结果汇总成一张可比对的表。"""
    n = len(records)
    if not n:
        return {}

    answerable = [r for r in records if r.get("answerable", True)]
    unanswerable = [r for r in records if not r.get("answerable", True)]

    by_type: Dict[str, dict] = {}
    for r in records:
        t = r.get("type", "unknown")
        bucket = by_type.setdefault(t, {"n": 0, "correct": 0, "evidence": 0})
        bucket["n"] += 1
        bucket["correct"] += 1 if r.get("correct") else 0
        bucket["evidence"] += 1 if r.get("evidence_hit") else 0
    for t, b in by_type.items():
        b["accuracy"] = _rate(b["correct"], b["n"])
        b["evidence_recall"] = _rate(b["evidence"], b["n"])

    costs = [float(r.get("cost_cny", 0.0)) for r in records]
    latencies = [float(r.get("latency_ms", 0.0)) for r in records]
    routes = [r.get("route", "local") for r in records]

    # 证据召回只在「论文里确实有答案」的题上算：
    # 不可答题的 gold_pages 只是占位，算进去会把分母弄脏
    return {
        "n": n,
        "accuracy": _rate(sum(1 for r in records if r.get("correct")), n),
        "answerable_n": len(answerable),
        "evidence_recall": _rate(
            sum(1 for r in answerable if r.get("evidence_hit")), len(answerable)
        ),
        "citation_valid_rate": _rate(sum(1 for r in records if r.get("citation_valid")), n),
        "number_consistency": _rate(sum(1 for r in records if r.get("number_ok")), n),
        "term_consistency": _rate(sum(1 for r in records if r.get("term_ok")), n),
        "anchored_rate": _rate(sum(1 for r in records if r.get("n_anchors", 0) > 0), n),
        "refusal_rate": _rate(sum(1 for r in records if r.get("refused")), n),
        "false_refusal_rate": _rate(
            sum(1 for r in answerable if r.get("refused")), len(answerable)
        ),
        "correct_refusal_rate": _rate(
            sum(1 for r in unanswerable if r.get("refused")), len(unanswerable)
        ),
        "cloud_ratio": _rate(sum(1 for x in routes if x == "cloud"), n),
        "total_cost_cny": round(sum(costs), 6),
        "cost_per_1k_cny": round(sum(costs) / n * 1000, 4),
        "avg_latency_ms": round(sum(latencies) / n, 1),
        "by_type": by_type,
    }


def group_by(records: Sequence[dict], key: str) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for r in records:
        out.setdefault(str(r.get(key, "")), []).append(r)
    return out
