"""把 docs/tech/part*.html 拼成完整文档，解析目录页码，导出 PDF。

为什么用 HTML + 无头浏览器而不是 Word/LaTeX：
- 排版用 CSS 精确控制（A4、页边距、分页、表格、图注），改一行就能重排；
- 不依赖 pandoc / LaTeX 等重型工具链，任何装了 Edge/Chrome 的机器都能复现；
- 浏览器打印引擎对中文与表格的支持稳定。

目录页码靠两遍渲染解决：浏览器打印不支持跨页引用，
因此先渲染一遍，在生成的 PDF 里定位每个标题所在页，再回填页码重排一次。

用法：
    python scripts/build_doc.py
输出：
    docs/HybridMind_技术文档.pdf
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TECH = ROOT / "docs" / "tech"
OUT_PDF = ROOT / "docs" / "HybridMind_技术文档.pdf"

DOC_TITLE = "HybridMind：端云协同的自适应大模型路由"
DOC_AUTHOR = "张云迪"
DOC_SUBJECT = "2026 第九届传智杯 · AI 大模型创新应用挑战赛（B 组）"

EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
]

# (层级, 标题) —— 标题文本必须与 HTML 中的 <h1>/<h2> 完全一致（含全角空格），
# 因为它同时被用作在 PDF 中定位该标题的检索串。
TOC_ENTRIES = [
    (1, "1　问题与动机"),
    (2, "1.1　端侧推理为何再次成为焦点"),
    (2, "1.2　一个被误判的前提"),
    (2, "1.3　本文要回答的三个问题"),
    (2, "1.4　主要贡献"),
    (1, "2　相关工作"),
    (2, "2.1　模型级联与预算感知推理"),
    (2, "2.2　大模型路由"),
    (2, "2.3　端侧小模型"),
    (2, "2.4　与本文的差异"),
    (1, "3　系统设计"),
    (2, "3.1　总体架构"),
    (2, "3.2　推理后端抽象"),
    (2, "3.3　编排与降级"),
    (2, "3.4　可观测性与成本账本"),
    (2, "3.5　交互与可视化"),
    (1, "4　路由方法"),
    (2, "4.1　问题形式化"),
    (2, "4.2　规则基线"),
    (2, "4.3　学习型路由"),
    (2, "4.4　置信度兜底与触发条件"),
    (1, "5　实验设置"),
    (2, "5.1　评测集构造"),
    (2, "5.2　判分口径"),
    (2, "5.3　对照与指标"),
    (2, "5.4　硬件与超参"),
    (1, "6　结果与分析"),
    (2, "6.1　本机推理能力"),
    (2, "6.2　全本地与全云端基线"),
    (2, "6.3　路由质量"),
    (2, "6.4　分布外检验"),
    (2, "6.5　失败分析"),
    (1, "7　讨论"),
    (2, '7.1　为什么"省钱"不是好理由'),
    (2, "7.2　阈值的业务含义"),
    (2, "7.3　局限性"),
    (2, "7.4　改进方向"),
    (1, "8　工程实现与复现"),
    (2, "8.1　代码结构"),
    (2, "8.2　部署"),
    (2, "8.3　复现步骤"),
    (2, "8.4　开发过程留痕"),
    (1, "参考文献"),
]


def find_browser() -> str:
    for path in EDGE_CANDIDATES:
        if Path(path).exists():
            return path
    found = shutil.which("chrome") or shutil.which("msedge")
    if found:
        return found
    raise SystemExit("未找到 Edge/Chrome，无法导出 PDF")


def render_toc(pages: dict[str, int] | None) -> str:
    rows = []
    for level, title in TOC_ENTRIES:
        cls = "toc-l1" if level == 1 else "toc-l2"
        page = pages.get(title, 0) if pages else 0
        label = str(page) if page else "—"
        rows.append(
            f'<div class="toc-row {cls}">'
            f'<span class="toc-title">{title}</span>'
            f'<span class="toc-dots"></span>'
            f'<span class="toc-page">{label}</span>'
            f"</div>"
        )
    return "\n".join(rows)


def build_html(toc_pages: dict[str, int] | None) -> Path:
    parts = sorted(TECH.glob("part*.html"))
    if not parts:
        raise SystemExit("没有找到 part*.html")
    shell = parts[0].read_text(encoding="utf-8")
    bodies = [p.read_text(encoding="utf-8") for p in parts[1:]]
    merged = shell.replace("</body>", "\n".join(bodies) + "\n</body>")
    merged = merged.replace("<!--TOC-->", render_toc(toc_pages))
    merged = merged.replace("__AUTHOR__", DOC_AUTHOR)
    out = TECH / "tech_doc.html"
    out.write_text(merged, encoding="utf-8")
    return out


def print_pdf(html: Path, browser: str) -> None:
    cmd = [
        browser,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--virtual-time-budget=8000",
        f"--print-to-pdf={OUT_PDF}",
        html.resolve().as_uri(),
    ]
    subprocess.run(cmd, capture_output=True, timeout=180)
    if not OUT_PDF.exists():
        raise SystemExit("PDF 生成失败")


def _is_toc_page(text: str) -> bool:
    """判断一页是否仍是目录。

    目录的特征是「标题行 + 纯数字页码」交替出现，纯数字行占比很高。
    早期版本用「一页里出现 ≥15 个 N.N 章节号」来判断，阈值过高：
    目录续页只有 9 个章节号，被误判成正文，导致该页之后所有标题
    都在目录页本身被检索到，页码全印成了该页的页码。
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 8:
        return False
    numeric = sum(1 for ln in lines if ln.isdigit())
    return numeric / len(lines) >= 0.3


def _body_start(doc) -> int:
    """返回正文起始页索引，跳过封面、摘要与目录（含续页）。"""
    for index in range(min(doc.page_count, 10)):
        if "目 录" in doc[index].get_text():
            cursor = index
            while cursor + 1 < doc.page_count and _is_toc_page(doc[cursor + 1].get_text()):
                cursor += 1
            return cursor + 1
    return 0


def locate_pages() -> dict[str, int]:
    """在渲染出的 PDF 里定位各标题所在页（1 基）。

    标题里的全角空格在浏览器里会被当作可折叠空白，PDF 中可能变成半角空格、
    也可能被保留，因此每个标题都用三种写法各试一次。
    """
    import fitz

    def variants(title: str) -> list[str]:
        return [title, title.replace("\u3000", " "), title.replace("\u3000", "")]

    found: dict[str, int] = {}
    with fitz.open(OUT_PDF) as doc:
        start = _body_start(doc)
        for _, title in TOC_ENTRIES:
            for index in range(start, doc.page_count):
                page = doc[index]
                if any(page.search_for(v) for v in variants(title)):
                    found[title] = index + 1
                    break
    return found


def stamp_pages(title: str) -> None:
    """给除封面外的每页加盖页脚：左标题、右页码。"""
    import fitz

    with fitz.open(OUT_PDF) as doc:
        total = doc.page_count
        for index in range(1, total):
            page = doc[index]
            width, height = page.rect.width, page.rect.height
            gray = (0.35, 0.38, 0.42)
            page.insert_text((width * 0.09, height - 26), title, fontsize=7, color=gray, fontname="helv")
            label = f"{index + 1} / {total}"
            text_width = fitz.get_text_length(label, fontname="helv", fontsize=8)
            page.insert_text(
                (width - width * 0.09 - text_width, height - 26),
                label,
                fontsize=8,
                color=gray,
                fontname="helv",
            )
            page.draw_line(
                (width * 0.09, height - 34), (width * 0.91, height - 34), color=(0.78, 0.81, 0.85), width=0.6
            )
        doc.saveIncr()


def set_metadata() -> None:
    import fitz

    with fitz.open(OUT_PDF) as doc:
        doc.set_metadata(
            {
                "title": DOC_TITLE,
                "author": DOC_AUTHOR,
                "subject": DOC_SUBJECT,
                "creator": "HybridMind build_doc.py",
            }
        )
        doc.saveIncr()


def main() -> None:
    browser = find_browser()

    # 第一遍：占位目录，用于测页
    print_pdf(build_html(None), browser)
    pages = locate_pages()

    # 第二遍：回填页码
    print_pdf(build_html(pages), browser)
    verify = locate_pages()

    shifted = [n for n in pages if pages[n] != verify.get(n)]
    print_pdf(build_html(verify), browser)
    final = locate_pages()

    stamp_pages("HybridMind — Adaptive LLM Routing (2026 Chuanzhibei Cup)")
    set_metadata()

    import fitz

    with fitz.open(OUT_PDF) as doc:
        count = doc.page_count

    print(f"PDF: {OUT_PDF}")
    print(f"大小: {OUT_PDF.stat().st_size // 1024} KB")
    print(f"页数: {count}")
    if count > 30:
        print(f"⚠ 超出 30 页限制 {count - 30} 页")
    if shifted:
        print(f"提示：{len(shifted)} 个标题页码在两遍之间发生位移，已用第三遍结果")
    missing = [title for _, title in TOC_ENTRIES if title not in final]
    if missing:
        print(f"⚠ 有 {len(missing)} 个目录项未在 PDF 中定位到：{missing[:3]}")


if __name__ == "__main__":
    main()
