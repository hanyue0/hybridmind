"""论文精读层：解析 → 检索 → 证据锚点。

与 router 的分工：
- router 决定「这个问题该谁答」；
- rag 决定「凭什么这么答」，并把答案锚回原文的具体句子。
"""

from .parser import Chunk, Paper, parse_pdf, render_page_png
from .retriever import Retriever, RetrievalHit, build_index
from .anchor import Anchor, extract_anchors, verify_anchors, AnchorReport
from .qa import PaperQA, AnswerResult

__all__ = [
    "Chunk",
    "Paper",
    "parse_pdf",
    "render_page_png",
    "Retriever",
    "RetrievalHit",
    "build_index",
    "Anchor",
    "extract_anchors",
    "verify_anchors",
    "AnchorReport",
    "PaperQA",
    "AnswerResult",
]
