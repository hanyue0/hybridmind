"""建库：把每个片段译成中文，写进离线双语索引。

三个地方共用同一套逻辑，所以抽成一个模块：
- ``scripts/build_paper_index.py``（命令行，批量建库）；
- ``POST /api/paper/{name}/index``（演示界面里给新上传的 PDF 建索引）；
- 评测脚本复现时也走这里。

翻译只做一次，结果落在 ``{name}.zh.json``；运行期不再调用云端。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

BATCH = 5
_LINE = re.compile(r"^\s*(S\d+)\s*\|{3}\s*(.+?)\s*$")

PROMPT = (
    "把下面这些英文学术片段逐条翻译成简洁中文，用于中文检索匹配。要求：\n"
    "1. 数字、百分比、模型名、指标名（如 F1、EM、BM25、RAG）原样保留；\n"
    "2. 公式与符号不要改写；\n"
    "3. 严格逐条输出，格式为 `S编号|||译文`，每条一行，不要输出其它任何内容。"
)


def _resolve(root: Path, name: str, suffix: str) -> Path:
    for base in (root, root / "uploads"):
        candidate = base / f"{name}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{name}{suffix}")


def load_chunks(root: Path, name: str) -> List[dict]:
    try:
        path = _resolve(root, name, ".chunks.json")
    except FileNotFoundError:
        from .parser import parse_pdf

        pdf = _resolve(root, name, ".pdf")
        paper = parse_pdf(pdf, name=name)
        path = pdf.with_suffix(".chunks.json")
        path.write_text(json.dumps(paper.to_dict(), ensure_ascii=False), encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))["chunks"]


def load_zh(root: Path, name: str) -> Dict[str, str]:
    try:
        path = _resolve(root, name, ".zh.json")
    except FileNotFoundError:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_zh(root: Path, name: str, zh: Dict[str, str]) -> None:
    try:
        target = _resolve(root, name, ".zh.json")
    except FileNotFoundError:
        # 还没建过索引：与对应 PDF 放在一起（上传的论文就落在 uploads/ 下）
        target = _resolve(root, name, ".pdf").with_suffix(".zh.json")
    target.write_text(json.dumps(zh, ensure_ascii=False, indent=1), encoding="utf-8")


def translate_batch(provider, chunks: Sequence[dict]):
    body = "\n".join(f"[{c['id']}] {c['text']}" for c in chunks)
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": body},
    ]
    gen = provider.generate(messages, max_tokens=len(chunks) * 260 + 80, temperature=0.0)
    out: Dict[str, str] = {}
    for line in gen.text.splitlines():
        m = _LINE.match(line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out, gen


def build_zh_index(
    root: Path,
    name: str,
    provider,
    limit: Optional[int] = None,
    progress=None,
) -> Dict[str, object]:
    """给一篇论文补齐中文译文。已译过的块跳过，可反复调用（增量）。"""
    chunks = load_chunks(root, name)
    zh = load_zh(root, name)
    todo = [c for c in chunks if c["id"] not in zh or not zh.get(c["id"])]
    if limit:
        todo = todo[:limit]

    cost = 0.0
    done = 0
    missing: List[str] = []
    for i in range(0, len(todo), BATCH):
        batch = todo[i : i + BATCH]
        got: Dict[str, str] = {}
        for _ in range(3):
            try:
                got, gen = translate_batch(provider, batch)
                cost += gen.cost_cny
                break
            except Exception:  # 网络抖动时重试，仍失败就记下来继续
                continue
        for c in batch:
            if got.get(c["id"]):
                zh[c["id"]] = got[c["id"]]
            else:
                missing.append(c["id"])
        done += len(batch)
        save_zh(root, name, zh)  # 每批落盘，进程中断也不丢进度
        if progress:
            progress(done, len(todo))

    save_zh(root, name, zh)
    return {
        "name": name,
        "chunks": len(chunks),
        "translated": len([v for v in zh.values() if v]),
        "done_now": done,
        "missing": missing,
        "cost_cny": round(cost, 6),
    }
