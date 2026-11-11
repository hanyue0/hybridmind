"""解析层：把 PDF 切成带「页码 + bbox + 章节」的片段。

设计要点（也是文档里要讲清楚的三件事）：

1. **保留 bbox**。只存纯文本的 RAG 前端只能贴一段引用文字，
   我们存 block 级坐标，前端才能把框画在原文那句话上。
2. **按页断开**。跨页的片段无法用一个矩形高亮，必须保证每块只属于一页。
3. **去页眉页脚**。arXiv 论文每页都有 "Published as a conference paper at ..."、
   页码、arXiv 侧栏，不去掉会被检索当成高相关片段反复命中。

依赖只有 pymupdf（已装），不引入向量库。
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pymupdf

# 连字与软连字符：arXiv 的 PDF 里 ffi/fl 经常是单个字形，检索时会漏命中
_LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\u00ad": "",
    "\u200b": "",
}

_PAGE_NOISE = re.compile(r"^(page\s*)?\d{1,3}$", re.IGNORECASE)
_WS = re.compile(r"\s+")
# 章节标题的常见写法：1 / 1.2 / 3.1.4 后面跟大写的标题词
_SECTION_RE = re.compile(r"^(\d{1,2}(\.\d{1,2}){0,2})\s+[A-Z\u4e00-\u9fff]")
# 参考文献之后的内容对问答没有价值，反而会被作者名命中
_REF_SECTION = re.compile(r"^(references|bibliography|参考文献)$", re.IGNORECASE)


def _normalize(text: str) -> str:
    for src, dst in _LIGATURES.items():
        text = text.replace(src, dst)
    text = unicodedata.normalize("NFKC", text)
    return _WS.sub(" ", text).strip()


@dataclass
class Chunk:
    """一个可检索、可高亮的原文片段。

    ``lines`` 存了片段内每一行的文本与坐标，用来把高亮框收到具体句子上——
    只有块级 bbox 的话，前端会框住整整半页，演示效果很差。
    """

    id: str                      # S1, S2, ...（prompt 里用它做引用编号）
    page: int                    # 页码，1 基
    text: str
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    section: str = ""            # 所属章节标题（尽力而为）
    lines: List[dict] = field(default_factory=list)  # [{"t": 行文本, "b": [x0,y0,x1,y1]}]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "page": self.page,
            "text": self.text,
            "bbox": list(self.bbox),
            "section": self.section,
            "lines": self.lines,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(
            id=d["id"],
            page=int(d["page"]),
            text=d["text"],
            bbox=tuple(d.get("bbox") or (0.0, 0.0, 0.0, 0.0)),
            section=d.get("section", ""),
            lines=list(d.get("lines") or []),
        )


@dataclass
class Paper:
    """一篇解析完的论文。"""

    name: str
    path: str
    n_pages: int
    chunks: List[Chunk] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "path": self.path,
            "n_pages": self.n_pages,
            "chunks": [c.to_dict() for c in self.chunks],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Paper":
        return cls(
            name=d["name"],
            path=d.get("path", ""),
            n_pages=int(d.get("n_pages", 0)),
            chunks=[Chunk.from_dict(c) for c in d.get("chunks", [])],
        )


@dataclass
class _Line:
    page: int
    text: str
    bbox: Tuple[float, float, float, float]
    size: float
    bold: bool


def _collect_lines(doc: "pymupdf.Document") -> Tuple[List[_Line], float]:
    """抽出所有文本行，同时统计正文字号（用于判定标题）。"""
    lines: List[_Line] = []
    sizes: Counter = Counter()
    for pno in range(doc.page_count):
        page = doc[pno]
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:  # 0=文本块，1=图片
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                text = _normalize("".join(s.get("text", "") for s in spans))
                if not text:
                    continue
                size = max((s.get("size", 0.0) for s in spans), default=0.0)
                bold = any("Bold" in (s.get("font") or "") for s in spans)
                bbox = tuple(round(v, 2) for v in line.get("bbox", (0, 0, 0, 0)))
                lines.append(_Line(pno + 1, text, bbox, size, bold))
                sizes[round(size, 1)] += len(text)
    body = sizes.most_common(1)[0][0] if sizes else 10.0
    return lines, body


def _repeated_lines(lines: Iterable[_Line], n_pages: int) -> set:
    """页眉页脚：在过半页面上出现过、且不长的行。"""
    counter: Counter = Counter()
    for ln in lines:
        if len(ln.text) <= 90:
            counter[ln.text] += 1
    threshold = max(3, int(n_pages * 0.5))
    return {t for t, c in counter.items() if c >= threshold}


def _is_heading(line: _Line, body: float) -> bool:
    if len(line.text) > 90 or len(line.text) < 3:
        return False
    if line.text.endswith((".", "。", "，", ",", ";", "；")):
        return False
    if line.size >= body + 1.0 and not line.text[0].islower():
        return True
    if _SECTION_RE.match(line.text) and line.size >= body - 0.2:
        return True
    return False


def _is_noise(line: _Line) -> bool:
    text = line.text
    if len(text) < 3:
        return True
    if _PAGE_NOISE.match(text):
        return True
    # 图表里大量的坐标轴标签/单字符：字母数字占比极低
    alnum = sum(ch.isalnum() for ch in text)
    if alnum == 0:
        return True
    return False


def _union(bboxes: List[Tuple[float, float, float, float]]) -> Tuple[float, float, float, float]:
    xs0 = [b[0] for b in bboxes]
    ys0 = [b[1] for b in bboxes]
    xs1 = [b[2] for b in bboxes]
    ys1 = [b[3] for b in bboxes]
    return (round(min(xs0), 2), round(min(ys0), 2), round(max(xs1), 2), round(max(ys1), 2))


def parse_pdf(
    path: str | Path,
    max_chars: int = 420,
    min_chars: int = 150,
    name: Optional[str] = None,
) -> Paper:
    """把 PDF 解析成 Chunk 列表。

    分块策略：按阅读顺序累积文本，攒到 ``max_chars`` 就断开；
    遇到标题行先落盘当前块，并把它记为后续块的 ``section``；
    跨页必断（保证一个 bbox 只落在一页里）。
    """
    path = Path(path)
    doc = pymupdf.open(str(path))
    lines, body = _collect_lines(doc)
    repeated = _repeated_lines(lines, doc.page_count)

    chunks: List[Chunk] = []
    section = ""
    buf: List[_Line] = []
    buf_len = 0

    def flush() -> None:
        nonlocal buf, buf_len
        if not buf:
            return
        text = _normalize(" ".join(ln.text for ln in buf))
        if len(text) >= min_chars:
            idx = len(chunks) + 1
            chunks.append(
                Chunk(
                    id=f"S{idx}",
                    page=buf[0].page,
                    text=text,
                    bbox=_union([ln.bbox for ln in buf]),
                    section=section,
                    lines=[{"t": ln.text, "b": list(ln.bbox)} for ln in buf],
                )
            )
        buf = []
        buf_len = 0

    for ln in lines:
        if ln.text in repeated or _is_noise(ln):
            continue
        if _is_heading(ln, body):
            if _REF_SECTION.match(ln.text.strip()):
                break
            flush()
            section = ln.text
            continue
        if buf and ln.page != buf[-1].page:
            flush()
        buf.append(ln)
        buf_len += len(ln.text) + 1
        if buf_len >= max_chars:
            flush()
    flush()

    paper = Paper(name=name or path.stem, path=str(path), n_pages=doc.page_count, chunks=chunks)
    doc.close()
    return paper


def render_page_png(path: str | Path, page: int, dpi: int = 110) -> bytes:
    """把某一页渲染成 PNG（前端左侧原图 + 高亮框用，避免引入 pdf.js）。"""
    doc = pymupdf.open(str(path))
    try:
        pix = doc[page - 1].get_pixmap(dpi=dpi)
        return pix.tobytes("png")
    finally:
        doc.close()


def page_size(path: str | Path, page: int) -> Tuple[float, float]:
    """页面原始尺寸（pt），前端按比例把 bbox 换算成像素。"""
    doc = pymupdf.open(str(path))
    try:
        rect = doc[page - 1].rect
        return (rect.width, rect.height)
    finally:
        doc.close()
