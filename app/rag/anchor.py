"""锚点层：把答案里的每一条结论，钉回原文的具体句子。

方案分两段，这是文档里要重点解释的设计选择：

**第一段 · 约束生成**：prompt 里要求模型用 ``[S3]`` 标注出处，片段不足就直说
"原文未提及"。这是第一选择，成本为零。

**第二段 · 确定性对齐兜底**：实测本地 1.5B 经常不给编号（它会把话说圆，
但不写 [S]）。这时**不让模型当裁判**，而是用三类与语言无关的信号做对齐：

1. **数字**：论文里的数字是检索答案的主线索，且中英一致（"44.1" 在两种语言里都是 44.1）；
2. **英文术语**：答案里出现的 F1 / BM25 / GPT-3 这类词，直接和原文做词面匹配；
3. **译文重叠**：建库期做过逐块中文译文，中文答案句与译文做字 bigram 重合度。

三类信号加权后取最优块，再把高亮框收紧到命中的那几行。
这样模型的输出格式好坏不再决定产品能不能用——这是「可评测」的前提。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .parser import Chunk
from .retriever import tokenize_en, tokenize_zh

_REF_RE = re.compile(r"\[\s*(S\d+(?:\s*[,，]\s*S\d+)*)\s*\]")
_NUM_RE = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)?%?)(?![\w])")
_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}")
_SENT_SPLIT = re.compile(r"(?<=[.!?。！？])\s+")
_TERM_STOP = {
    "the", "and", "for", "with", "this", "that", "from", "are", "was", "were",
    "have", "has", "not", "but", "you", "can", "our", "which", "what", "how",
    "http", "https", "www", "com", "org", "pdf", "et", "al",
}
REFUSE_MARKERS = ("原文未提及", "未提及", "无法回答", "不足以回答", "片段中没有", "没有提到")


@dataclass
class Anchor:
    """一条证据锚点。"""

    chunk_id: str
    page: int
    quote: str
    bbox: Tuple[float, float, float, float]
    method: str = "cited"       # cited=模型自己标了编号；aligned=兜底对齐
    score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "page": self.page,
            "quote": self.quote,
            "bbox": list(self.bbox),
            "method": self.method,
            "score": round(self.score, 3),
        }


@dataclass
class AnchorReport:
    """一次问答的证据健康度。"""

    anchors: List[Anchor] = field(default_factory=list)
    invalid_refs: List[str] = field(default_factory=list)
    missing_numbers: List[str] = field(default_factory=list)
    missing_terms: List[str] = field(default_factory=list)
    refused: bool = False

    @property
    def number_ok(self) -> bool:
        return not self.missing_numbers

    @property
    def term_ok(self) -> bool:
        return not self.missing_terms

    @property
    def valid(self) -> bool:
        """三条规则全过、且至少有一条锚点。"""
        return bool(self.anchors) and not self.invalid_refs and self.number_ok and self.term_ok

    @property
    def citation_valid(self) -> bool:
        """引用编号是否都真实存在于检索结果里（衡量幻觉引用）。"""
        return not self.invalid_refs

    @property
    def gate_pass(self) -> bool:
        """校验门控：本地答案要不要保留。

        只用「与语言无关」的两条：有锚点、编号有效、数字可溯。
        术语一致性（term_ok）不计入——中文作答时英文术语会被翻译，
        拿它当门控会把正确答案误判为不通过（实测差异见技术文档 6.3 节）。
        """
        return bool(self.anchors) and self.citation_valid and self.number_ok

    def to_dict(self) -> dict:
        return {
            "anchors": [a.to_dict() for a in self.anchors],
            "invalid_refs": self.invalid_refs,
            "missing_numbers": self.missing_numbers,
            "missing_terms": self.missing_terms,
            "refused": self.refused,
            "valid": self.valid,
            "citation_valid": self.citation_valid,
            "gate_pass": self.gate_pass,
            "number_ok": self.number_ok,
            "term_ok": self.term_ok,
        }


def is_refusal(answer: str) -> bool:
    return any(m in answer for m in REFUSE_MARKERS)


def strip_refs(text: str) -> str:
    """去掉 [S3] 这类编号。

    不做这一步，校验会把编号里的 "S10" 当成英文术语、把 "3" 当成数字，
    造成「自己引用自己」的假阴性——实测踩过，写在测试里了。
    """
    return _REF_RE.sub("", text or "")


def _numbers(text: str) -> List[str]:
    """抽出需要溯源的数字：小数 / 百分数 / 两位以上的整数。"""
    out = []
    for m in _NUM_RE.finditer(text):
        v = m.group(1).replace(",", "")
        if "." in v or "%" in v or len(v) >= 2:
            out.append(v.rstrip("%"))
    return out


def _terms(text: str) -> List[str]:
    out = []
    for m in _TERM_RE.finditer(text):
        w = m.group(0)
        if w.lower() in _TERM_STOP:
            continue
        if w.islower() and len(w) < 4:
            continue
        out.append(w)
    return out


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if len(s.strip()) >= 8]


def _quote_from_chunk(chunk: Chunk, answer_piece: str) -> Tuple[str, Tuple[float, float, float, float]]:
    """在片段里选出最该高亮的句子，并把框收紧到这些行。"""
    nums = set(_numbers(answer_piece))
    terms = {t.lower() for t in _terms(answer_piece)}
    best, best_score = "", -1.0
    for sent in _sentences(chunk.text) or [chunk.text]:
        low = sent.lower()
        s = 0.0
        s += 2.0 * sum(1 for n in nums if n in sent)
        s += 1.0 * sum(1 for t in terms if t in low)
        if s > best_score:
            best, best_score = sent, s
    quote = best or chunk.text[:200]
    return quote, _bbox_for_quote(chunk, quote)


def _bbox_for_quote(chunk: Chunk, quote: str) -> Tuple[float, float, float, float]:
    """把引文映射回行坐标：命中词最多的那几行并起来。"""
    if not chunk.lines:
        return chunk.bbox
    q_tokens = set(tokenize_en(quote)) | {n for n in _numbers(quote)}
    if not q_tokens:
        return chunk.bbox
    hits = []
    for ln in chunk.lines:
        text = ln["t"]
        low = text.lower()
        score = sum(1 for t in q_tokens if t in low)
        if score:
            hits.append((score, tuple(ln["b"])))
    if not hits:
        return chunk.bbox
    hits.sort(key=lambda x: -x[0])
    keep = [b for s, b in hits if s >= max(1, hits[0][0] // 2)][:6]
    xs0 = min(b[0] for b in keep); ys0 = min(b[1] for b in keep)
    xs1 = max(b[2] for b in keep); ys1 = max(b[3] for b in keep)
    return (round(xs0, 2), round(ys0, 2), round(xs1, 2), round(ys1, 2))


def _align_score(answer_piece: str, chunk: Chunk, zh: Dict[str, str]) -> float:
    """与语言无关的对齐打分。"""
    nums = set(_numbers(answer_piece))
    terms = {t.lower() for t in _terms(answer_piece)}
    low = chunk.text.lower()
    s = 0.0
    s += 3.0 * sum(1 for n in nums if n in chunk.text)
    s += 1.5 * sum(1 for t in terms if t in low)
    zh_text = zh.get(chunk.id, "")
    if zh_text:
        a = set(tokenize_zh(answer_piece))
        b = set(tokenize_zh(zh_text))
        if a:
            s += 2.0 * len(a & b) / len(a)
    return s


def extract_anchors(
    answer: str,
    chunks: Sequence[Chunk],
    zh: Optional[Dict[str, str]] = None,
    threshold: float = 1.2,
) -> List[Anchor]:
    """先认模型标的编号，认不到再用确定性对齐兜底。"""
    zh = zh or {}
    by_id = {c.id: c for c in chunks}
    anchors: List[Anchor] = []
    seen = set()

    cited: List[str] = []
    for m in _REF_RE.finditer(answer):
        for part in re.split(r"[,，]", m.group(1)):
            cid = part.strip().upper()
            if cid in by_id and cid not in cited:
                cited.append(cid)

    if cited:
        for cid in cited:
            chunk = by_id[cid]
            quote, bbox = _quote_from_chunk(chunk, strip_refs(answer))
            anchors.append(Anchor(cid, chunk.page, quote, bbox, "cited", 1.0))
            seen.add(cid)
        return anchors

    # ---- 兜底：逐句对齐 ----
    for sent in _sentences(strip_refs(answer)) or [strip_refs(answer)]:
        scored = sorted(
            ((_align_score(sent, c, zh), c) for c in chunks), key=lambda x: -x[0]
        )
        if not scored or scored[0][0] < threshold:
            continue
        score, chunk = scored[0]
        if chunk.id in seen:
            continue
        quote, bbox = _quote_from_chunk(chunk, sent)
        anchors.append(Anchor(chunk.id, chunk.page, quote, bbox, "aligned", score))
        seen.add(chunk.id)
        if len(anchors) >= 4:
            break
    return anchors


def verify_anchors(
    answer: str,
    anchors: Sequence[Anchor],
    chunks: Sequence[Chunk],
    zh: Optional[Dict[str, str]] = None,
) -> AnchorReport:
    """三条确定性规则 + 拒答判定，输出可统计的证据健康度。"""
    zh = zh or {}
    by_id = {c.id: c for c in chunks}
    report = AnchorReport(anchors=list(anchors), refused=is_refusal(answer))

    referenced = {a.chunk_id for a in anchors}
    for m in _REF_RE.finditer(answer):
        for part in re.split(r"[,，]", m.group(1)):
            cid = part.strip().upper()
            if cid not in by_id:
                report.invalid_refs.append(cid)
    # 模型只引用了 A 段，但结论里的数字在 B 段：只校验「被引用的片段」，
    # 没被引用到的片段不能替它背书
    scope = " ".join(
        c.text + " " + zh.get(c.id, "") for c in chunks if c.id in referenced
    ).lower()

    body = strip_refs(answer)
    for n in _numbers(body):
        if n not in scope and n not in " ".join(a.quote for a in anchors):
            report.missing_numbers.append(n)
    for t in _terms(body):
        if t.lower() not in scope:
            report.missing_terms.append(t)

    return report
