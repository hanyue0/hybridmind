"""把 docs/tech/part*.html 拼成完整文档并导出 PDF。

为什么用 HTML + 无头浏览器而不是 Word/LaTeX：
- 排版用 CSS 精确控制（A4、页边距、分页、表格、图注），改一行就能重排；
- 不依赖 pandoc / LaTeX 等重型工具链，任何装了 Edge/Chrome 的机器都能复现；
- 浏览器打印引擎对中文与表格的支持稳定。

用法：
    python scripts/build_doc.py
输出：
    docs/HybridMind_技术文档.pdf
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TECH = ROOT / "docs" / "tech"
OUT_PDF = ROOT / "docs" / "HybridMind_技术文档.pdf"

EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
]


def find_browser() -> str:
    for path in EDGE_CANDIDATES:
        if Path(path).exists():
            return path
    found = shutil.which("chrome") or shutil.which("msedge")
    if found:
        return found
    raise SystemExit("未找到 Edge/Chrome，无法导出 PDF")


def build_html() -> Path:
    parts = sorted(TECH.glob("part*.html"))
    if not parts:
        raise SystemExit("没有找到 part*.html")
    shell = parts[0].read_text(encoding="utf-8")
    bodies = [p.read_text(encoding="utf-8") for p in parts[1:]]
    marker = "</body>"
    if marker not in shell:
        raise SystemExit("part1 缺少 </body>")
    merged = shell.replace(marker, "\n".join(bodies) + "\n" + marker)
    out = TECH / "tech_doc.html"
    out.write_text(merged, encoding="utf-8")
    return out


def count_pages(pdf: Path) -> int:
    try:
        import fitz  # PyMuPDF

        with fitz.open(pdf) as doc:
            return doc.page_count
    except Exception:
        return -1


def stamp_pages(pdf: Path, title: str = "HybridMind — Adaptive LLM Routing (2026 Chuanzhibei Cup)") -> int:
    """给除封面外的每页加页脚：左标题、右页码。

    浏览器打印引擎不支持 CSS 的 @page 边距盒，因此页码在导出后补盖，
    这样正文版式不受影响。
    """
    try:
        import fitz
    except Exception:
        return -1

    with fitz.open(pdf) as doc:
        total = doc.page_count
        for index in range(1, total):
            page = doc[index]
            width, height = page.rect.width, page.rect.height
            gray = (0.35, 0.38, 0.42)
            page.insert_text((width * 0.09, height - 26), title, fontsize=7, color=gray, fontname="helv")
            label = f"{index + 1} / {total}"
            text_width = fitz.get_text_length(label, fontname="helv", fontsize=8)
            page.insert_text((width - width * 0.09 - text_width, height - 26), label, fontsize=8, color=gray, fontname="helv")
            page.draw_line((width * 0.09, height - 34), (width * 0.91, height - 34), color=(0.78, 0.81, 0.85), width=0.6)
        doc.saveIncr()
    return total


def main() -> None:
    html = build_html()
    browser = find_browser()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
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

    pages = count_pages(OUT_PDF)
    if pages > 1:
        stamp_pages(OUT_PDF)
    size_kb = OUT_PDF.stat().st_size // 1024
    print(f"PDF: {OUT_PDF}")
    print(f"大小: {size_kb} KB")
    print(f"页数: {pages if pages > 0 else '未统计'}")
    if pages > 30:
        print(f"⚠ 超出 30 页限制 {pages - 30} 页，需要压缩")


if __name__ == "__main__":
    main()
