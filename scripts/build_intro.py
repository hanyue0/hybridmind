"""把 docs/intro/intro.html 打成 PDF（作品介绍文档）。

与技术文档不同，这份没有目录与页码引用，所以不需要两遍渲染，直接打印即可。
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "docs" / "intro" / "intro.html"
OUT = ROOT / "docs" / "HybridMind_作品介绍.pdf"

EDGE = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
]


def find_browser() -> str:
    for path in EDGE:
        if Path(path).exists():
            return path
    for name in ("msedge", "chrome"):
        found = shutil.which(name)
        if found:
            return found
    raise SystemExit("未找到 Edge/Chrome")


def main() -> int:
    browser = find_browser()
    cmd = [
        browser,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--virtual-time-budget=8000",
        f"--print-to-pdf={OUT}",
        SRC.resolve().as_uri(),
    ]
    subprocess.run(cmd, capture_output=True, timeout=180)
    if not OUT.exists():
        raise SystemExit("PDF 生成失败")

    import pymupdf

    with pymupdf.open(OUT) as doc:
        doc.set_metadata(
            {
                "title": "HybridMind 作品介绍：论文精读问答与证据锚点",
                "author": "张云迪",
                "subject": "2026 第九届传智杯 · AI 大模型创新应用挑战赛（B 组）",
                "creator": "HybridMind build_intro.py",
            }
        )
        pages = doc.page_count
        doc.saveIncr()

    print(f"PDF: {OUT}")
    print(f"大小: {OUT.stat().st_size // 1024} KB")
    print(f"页数: {pages}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
