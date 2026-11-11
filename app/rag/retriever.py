"""检索层：纯 CPU 的 BM25，外加一层「离线双语索引」。

为什么不用向量库：
- 目标环境 2 vCPU / 8 GB、无 GPU，BM25 建索引是毫秒级，向量化一篇 20 页论文要几十秒；
- 评委需要能逐条复算检索结果，BM25 的分数可以手算，稠密向量不行。

为什么要双语：
- 场景是「中文提问、英文论文」。BM25 是词面匹配，中文问句和英文段落的重合度是 0，
  直接检索必然全错。
- 解法不是运行时调云端翻译（那等于每个问题都上云，路由就失去意义），
  而是**建库期一次性**把每块译文写进索引（``data/papers/<name>.zh.json``），
  运行期中文问句命中译文、锚点仍回指英文原文的 chunk。运行期零额外调用。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .parser import Chunk, Paper

_WORD = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")
_CJK = re.compile(r"[\u4e00-\u9fff]+")

# 只挡最高频的功能词；剩下的交给 IDF
_EN_STOP = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are",
    "was", "were", "be", "been", "by", "with", "as", "that", "this", "these",
    "those", "it", "its", "we", "our", "they", "their", "which", "what", "how",
    "does", "do", "did", "from", "at", "than", "then", "also", "can", "may",
    "such", "not", "no", "has", "have", "had", "but", "if", "when", "where",
    "many", "much", "used", "uses", "using", "would", "could", "should", "there",
}
# 中文问句里的疑问词二元组，命中它们对定位证据没有帮助
_ZH_STOP = {
    "的是", "什么", "怎么", "如何", "为什", "么样", "这个", "那个", "多少",
    "请问", "这篇", "文章", "论文", "作者", "我们", "他们", "可以", "是否",
    "以及", "并且", "但是", "因为", "所以", "如果", "已经", "进行", "通过",
}

# 极简词干化：只处理最高频的屈折变化，不引入 nltk。
# 索引与查询走同一个函数，所以即使规则不完美，两侧仍是一致的。
_SUFFIX_RULES = (
    ("ization", "ize"),
    ("ations", "ate"),
    ("ation", "ate"),
    ("ings", ""),
    ("ing", ""),
    ("edly", ""),
    ("ed", ""),
    ("ies", "y"),
    ("es", ""),
    ("s", ""),
)
_NO_S = ("ss", "us", "is", "as")


def _stem(word: str) -> str:
    if len(word) <= 4:
        return word
    for suffix, repl in _SUFFIX_RULES:
        if not word.endswith(suffix):
            continue
        if suffix == "s" and word.endswith(_NO_S):
            continue
        if suffix == "ed" and word.endswith("eed"):
            continue
        stem = word[: -len(suffix)]
        if len(stem) < 3:
            continue
        return stem + repl
    return word


def tokenize_en(text: str) -> List[str]:
    out = []
    for m in _WORD.finditer(text.lower()):
        w = m.group(0).strip(".-")
        if len(w) < 2 or w in _EN_STOP:
            continue
        out.append(_stem(w))
    return out


def tokenize_zh(text: str) -> List[str]:
    """中文按字 bigram 切：不依赖分词器，术语也能命中。"""
    out: List[str] = []
    for run in _CJK.findall(text):
        if len(run) == 1:
            continue
        for i in range(len(run) - 1):
            big = run[i : i + 2]
            if big in _ZH_STOP:
                continue
            out.append(big)
    return out


def tokenize(text: str) -> List[str]:
    return tokenize_en(text) + tokenize_zh(text)


@dataclass
class RetrievalHit:
    chunk: Chunk
    score: float
    rank: int


class Retriever:
    """BM25（k1=1.5, b=0.75），索引英文原文 + 建库期译文。"""

    def __init__(self, chunks: Sequence[Chunk], zh: Optional[Dict[str, str]] = None,
                 k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = list(chunks)
        self._pos = {c.id: i for i, c in enumerate(self.chunks)}
        self.zh = zh or {}
        self.k1 = k1
        self.b = b
        self._tf: List[Counter] = []
        self._len: List[int] = []
        df: Counter = Counter()
        for c in self.chunks:
            toks = tokenize(c.text) + tokenize_zh(self.zh.get(c.id, ""))
            tf = Counter(toks)
            self._tf.append(tf)
            self._len.append(max(1, len(toks)))
            df.update(tf.keys())
        self._df = df
        self._n = max(1, len(self.chunks))
        self._avgdl = max(1.0, sum(self._len) / self._n)
        self._idf = {
            t: math.log(1 + (self._n - d + 0.5) / (d + 0.5)) for t, d in df.items()
        }

    # ---- 建索引时的最大可得分数，用于把分数归一化到 [0,1] ----
    def _norm(self, qterms: Iterable[str]) -> float:
        return max(1e-6, sum(self._idf.get(t, 0.0) * (self.k1 + 1) for t in set(qterms)))

    def search(self, query: str, k: int = 5) -> List[RetrievalHit]:
        qterms = tokenize(query)
        if not qterms:
            return []
        uniq = [t for t in qterms if t in self._idf]
        scores: List[Tuple[int, float]] = []
        for i, tf in enumerate(self._tf):
            s = 0.0
            dl = self._len[i]
            for t in uniq:
                f = tf.get(t, 0)
                if not f:
                    continue
                idf = self._idf[t]
                s += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / self._avgdl))
            if s > 0:
                scores.append((i, s))
        scores.sort(key=lambda x: -x[1])
        norm = self._norm(uniq)
        hits = []
        for rank, (i, s) in enumerate(scores[:k], start=1):
            hits.append(RetrievalHit(self.chunks[i], round(s / norm, 4), rank))
        return hits

    # ---- 路由要用的 4 维检索特征 ----
    def features(self, query: str, hits: Sequence[RetrievalHit]) -> Dict[str, float]:
        if not hits:
            return {
                "top1_score": 0.0,
                "score_gap": 0.0,
                "n_pages_hit": 0.0,
                "coverage": 0.0,
            }
        top1 = hits[0].score
        top2 = hits[1].score if len(hits) > 1 else 0.0
        pages = {h.chunk.page for h in hits}
        qterms = set(tokenize(query))
        covered = set()
        for h in hits:
            tf = self._tf[self._pos[h.chunk.id]]
            covered.update(t for t in qterms if t in tf)
        coverage = len(covered) / max(1, len(qterms))
        return {
            "top1_score": round(top1, 4),
            "score_gap": round(max(0.0, top1 - top2), 4),
            "n_pages_hit": float(len(pages)),
            "coverage": round(coverage, 4),
        }


def build_index(paper: Paper, zh: Optional[Dict[str, str]] = None) -> Retriever:
    return Retriever(paper.chunks, zh=zh)
