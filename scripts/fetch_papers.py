"""下载演示语料（3 篇 arXiv 论文）。

为什么不把 PDF 直接放进仓库：
三篇论文都是 arXiv 的 **non-exclusive license**（作者只授权 arXiv 分发，
并未授权第三方再分发）。仓库是公开的，直接提交 PDF 会越过许可边界。
所以这里只提交「论文清单 + 下载脚本」，任何人一条命令就能把语料还原，
构建镜像时也在构建期下载，不进入 git 历史。

用法：
    python scripts/fetch_papers.py
输出：
    data/papers/{name}.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "papers"

PAPERS = [
    {
        "name": "rag_2005.11401",
        "arxiv": "2005.11401",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "authors": "P. Lewis, E. Perez, A. Piktus, et al.",
        "why": "本系统的技术路线来源；含 21M 文档索引、EM 等可核对的数字",
    },
    {
        "name": "lora_2106.09685",
        "arxiv": "2106.09685",
        "title": "LoRA: Low-Rank Adaptation of Large Language Models",
        "authors": "E. J. Hu, Y. Shen, P. Wallis, et al.",
        "why": "参数效率方向；表格密集，适合数字定位类问题",
    },
    {
        "name": "lost_in_middle_2307.03172",
        "arxiv": "2307.03172",
        "title": "Lost in the Middle: How Language Models Use Long Contexts",
        "authors": "N. F. Liu, K. Lin, J. Hewitt, et al.",
        "why": "直接论证「长上下文不等于读得准」，是本作品「检索+锚点」的动机",
    },
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ok = 0
    for meta in PAPERS:
        target = OUT / f"{meta['name']}.pdf"
        if target.exists() and target.stat().st_size > 100_000:
            print(f"[skip] {meta['name']} 已存在（{target.stat().st_size // 1024} KB）")
            ok += 1
            continue
        url = f"https://arxiv.org/pdf/{meta['arxiv']}"
        try:
            resp = httpx.get(url, follow_redirects=True, timeout=90)
            resp.raise_for_status()
            target.write_bytes(resp.content)
            print(f"[ok]   {meta['name']} ← {url}（{len(resp.content) // 1024} KB）")
            ok += 1
        except Exception as exc:  # 网络不可达时不要中断构建
            print(f"[fail] {meta['name']}：{exc}", file=sys.stderr)
    print(f"\n{ok}/{len(PAPERS)} 篇就绪，目录 {OUT}")
    return 0 if ok == len(PAPERS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
