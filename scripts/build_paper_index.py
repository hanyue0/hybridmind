"""建库脚本：解析 PDF + 生成逐块中文译文（离线双语索引）。

为什么译文要落盘而不是运行时翻译：
- 中文问句和英文段落做 BM25 的必要前提是「两侧有共同词」；
- 若每个问题都先调一次云端翻译，等于所有请求都上云，路由就失去意义；
- 建库期一次性翻译，运行期零额外调用，成本只在建库时发生一次。

成本：3 篇论文约 400 块，按 deepseek-chat 价格约 1 元。脚本会打印实际花费。

用法：
    python scripts/build_paper_index.py                # 增量：已译过的块不重复翻
    python scripts/build_paper_index.py --force        # 全部重译
    python scripts/build_paper_index.py --limit 12     # 只翻前 12 块（试跑）
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.providers import OpenAICompatProvider  # noqa: E402
from app.rag.parser import parse_pdf  # noqa: E402

BATCH = 5
_LINE = re.compile(r"^\s*(S\d+)\s*\|{3}\s*(.+?)\s*$")

PROMPT = (
    "把下面这些英文学术片段逐条翻译成简洁中文，用于中文检索匹配。要求：\n"
    "1. 数字、百分比、模型名、指标名（如 F1、EM、BM25、RAG）原样保留；\n"
    "2. 公式与符号不要改写；\n"
    "3. 严格逐条输出，格式为 `S编号|||译文`，每条一行，不要输出其它任何内容。"
)


def translate_batch(provider: OpenAICompatProvider, chunks: list[dict]):
    """返回 (id→译文, Generation)。"""
    body = "\n".join(f"[{c['id']}] {c['text']}" for c in chunks)
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": body},
    ]
    gen = provider.generate(messages, max_tokens=len(chunks) * 260 + 80, temperature=0.0)
    out: dict = {}
    for line in gen.text.splitlines():
        m = _LINE.match(line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out, gen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers", default=str(ROOT / "data" / "papers"))
    parser.add_argument("--force", action="store_true", help="重解析 PDF 并全部重译")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 块（调试用）")
    parser.add_argument("--skip-zh", action="store_true", help="只建分块，不翻译")
    args = parser.parse_args()

    root = Path(args.papers)
    provider = OpenAICompatProvider(
        settings.cloud_base_url,
        settings.cloud_api_key,
        settings.cloud_model,
        settings.price_of(settings.cloud_model),
        timeout_s=120,
    )

    total_cost = 0.0
    for pdf in sorted(root.glob("*.pdf")):
        name = pdf.stem
        cache = root / f"{name}.chunks.json"
        if cache.exists() and not args.force:
            paper = json.loads(cache.read_text(encoding="utf-8"))
            print(f"[{name}] 复用解析缓存：{len(paper['chunks'])} 块 / {paper['n_pages']} 页")
        else:
            p = parse_pdf(pdf, name=name)
            paper = p.to_dict()
            cache.write_text(json.dumps(paper, ensure_ascii=False), encoding="utf-8")
            print(f"[{name}] 解析完成：{len(paper['chunks'])} 块 / {paper['n_pages']} 页")

        if args.skip_zh:
            continue

        zh_path = root / f"{name}.zh.json"
        zh = {} if args.force else (
            json.loads(zh_path.read_text(encoding="utf-8")) if zh_path.exists() else {}
        )
        todo = [c for c in paper["chunks"] if c["id"] not in zh]
        if args.limit:
            todo = todo[: args.limit]
        if not todo:
            print(f"[{name}] 译文已齐全（{len(zh)} 块）")
            continue

        print(f"[{name}] 待翻译 {len(todo)} 块 …")
        for i in range(0, len(todo), BATCH):
            batch = todo[i : i + BATCH]
            for attempt in range(3):
                try:
                    got, gen = translate_batch(provider, batch)
                    total_cost += gen.cost_cny
                    break
                except Exception as exc:  # 网络抖动重试
                    print(f"    批次失败（{exc}），第 {attempt + 1} 次重试")
                    time.sleep(2)
            else:
                got = {}
            for c in batch:
                zh[c["id"]] = got.get(c["id"], "")
            missing = [c["id"] for c in batch if not zh.get(c["id"])]
            if missing:
                print(f"    警告：{missing} 未取到译文")
            zh_path.write_text(json.dumps(zh, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"    {min(i + BATCH, len(todo))}/{len(todo)} 块，累计 ¥{total_cost:.3f}")

        empty = sum(1 for k, v in zh.items() if not v)
        print(f"[{name}] 译文完成：{len(zh)} 块，空译文 {empty} 块")

    print(f"\n总花费 ≈ ¥{total_cost:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
