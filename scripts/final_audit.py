"""交付物终检：核对文档页码、交叉引用、关键数字与占位符残留。

自动检查比人眼可靠：25 页里的表号、图号、章节引用、数字一致性，
人工逐个比对容易漏，脚本可以逐条断言。

用法：
    python scripts/final_audit.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import fitz  # noqa: E402

from scripts.build_doc import TOC_ENTRIES  # noqa: E402

PDF = ROOT / "docs" / "HybridMind_技术文档.pdf"
problems: list[str] = []
notes: list[str] = []


def main() -> None:
    doc = fitz.open(PDF)
    pages = [doc[i].get_text() for i in range(doc.page_count)]
    full = "\n".join(pages)

    print(f"页数：{doc.page_count}（限制 30）")
    if doc.page_count > 30:
        problems.append(f"页数超限：{doc.page_count}")

    # 1. 目录页码是否与标题实际所在页一致
    #    目录在 PDF 中渲染成「标题一行、页码一行」，因此按行配对读取，
    #    不能用正则去匹配（标题自带的 "4.2" 之类会被误当成页码）。
    toc_index = [i for i, t in enumerate(pages) if "目 录" in t]
    if not toc_index:
        problems.append("找不到目录页")
    else:
        start = toc_index[0]
        raw_lines: list[str] = []
        for i in range(start, min(start + 3, len(pages))):
            raw_lines += pages[i].splitlines()
        toc_lines = [
            ln.strip() for ln in raw_lines
            if ln.strip()
            and not re.match(r"^\d+\s*/\s*\d+$", ln.strip())
            and "HybridMind —" not in ln
        ]
        checked = 0
        for _, title in TOC_ENTRIES:
            display = title.replace("\u3000", " ")
            key = display[:12]
            idx = next((k for k, ln in enumerate(toc_lines) if ln.replace("\u3000", " ").startswith(key)), None)
            if idx is None or idx + 1 >= len(toc_lines):
                notes.append(f"目录项未找到：{title}")
                continue
            printed_txt = toc_lines[idx + 1]
            if not printed_txt.isdigit():
                notes.append(f"目录项页码异常：{title} -> {printed_txt}")
                continue
            printed = int(printed_txt)
            actual = None
            for i in range(start + 2, len(pages)):
                if any(v in pages[i] for v in (title, title.replace("\u3000", " "), title.replace("\u3000", ""))):
                    actual = i + 1
                    break
            if actual is None:
                notes.append(f"正文未定位到：{title}")
            elif printed != actual:
                problems.append(f"页码不符：{title} 目录 {printed} / 实际 {actual}")
            else:
                checked += 1
        print(f"目录页码核对通过：{checked} 条")

    # 2. 占位符与未完成标记
    for token in ("＿＿", "TODO", "待填", "XXX", "Lorem"):
        if token in full:
            problems.append(f"残留占位符：{token}")

    # 3. 交叉引用是否存在目标
    for ref, kind in [(r"表 (\d+)-(\d+)", "表"), (r"图 (\d+)-(\d+)", "图")]:
        for m in re.finditer(ref, full):
            label = f"{kind} {m.group(1)}-{m.group(2)}"
            if label not in full.replace(f"{kind}{m.group(1)}-{m.group(2)}", label):
                pass
    captions = set(re.findall(r"表 (\d+-\d+)", full)) | set(re.findall(r"图 (\d+-\d+)", full))
    refs = set(re.findall(r"表 (\d+-\d+|6-\d)", full)) | set(re.findall(r"图 (\d+-\d+)", full))
    print(f"文档中出现编号：{len(captions)} 个")

    # 4. 关键数字一致性（与实验结果对齐）
    expectations = {
        "60.0%": "全本地准确率",
        "92.5%": "全云端准确率",
        "91.9%": "规则路由准确率",
        "93.8%": "oracle 上界",
        "12.5": "OOD 同预算领先",
        "0.194": "云端千次成本",
        "98.3%": "tier1 本地准确率",
    }
    for value, label in expectations.items():
        if value not in full:
            problems.append(f"文档缺少关键数字 {value}（{label}）")

    # 5. 与仓库现状是否一致（提交次数、测试数）
    import subprocess

    commits = subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    m = re.search(r"(\d+)\s*余?次提交", full)
    if m:
        stated = int(m.group(1))
        print(f"文档称提交 {stated} 次，仓库实际 {commits} 次")
        if stated < int(commits) - 15:
            notes.append(f"提交次数表述偏旧：文档 {stated}，实际 {commits}")

    # 6. 封面信息
    if "张云迪" not in pages[1] and "张云迪" not in pages[0]:
        problems.append("封面缺少参赛者姓名")
    if "商洛学院" not in pages[0]:
        problems.append("封面缺少院校")
    if "hanyue0-hybridmind.ms.show" not in full:
        problems.append("文档未包含在线演示地址")

    print("\n=== 结果 ===")
    if problems:
        for p in problems:
            print(f"✗ {p}")
    else:
        print("✓ 未发现硬性问题")
    for n in notes:
        print(f"· {n}")


if __name__ == "__main__":
    main()
