"""本机推理基准：把「估算速度」换成可写进文档的实测值。

用法：
    python scripts/bench_local.py
输出：
    终端表格 + docs/figures/local_bench.md
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.providers import OllamaProvider, ProviderError  # noqa: E402

PROMPTS = [
    "计算 87 + 8 的值。",
    "一件商品单价 45 元，买 7 件，打 9 折，需要支付多少元？",
    "请证明根号 2 是无理数，并说明反证法的步骤与原理。",
]


def main() -> None:
    provider = OllamaProvider(settings.ollama_host, settings.local_model, timeout_s=180)
    rows = []
    for index, prompt in enumerate(PROMPTS, 1):
        try:
            generation = provider.generate(
                [{"role": "user", "content": prompt}], max_tokens=256, temperature=0.2
            )
        except ProviderError as exc:
            print(f"[{index}] 失败：{exc}")
            return
        throughput = (
            generation.completion_tokens / (generation.latency_ms / 1000)
            if generation.latency_ms
            else 0.0
        )
        rows.append(
            {
                "prompt": prompt,
                "prompt_tokens": generation.prompt_tokens,
                "completion_tokens": generation.completion_tokens,
                "latency_ms": generation.latency_ms,
                "tok_s": throughput,
            }
        )
        print(
            f"[{index}/{len(PROMPTS)}] {generation.completion_tokens} tokens "
            f"in {generation.latency_ms / 1000:.1f}s = {throughput:.1f} tok/s"
        )

    avg_tok_s = statistics.mean(row["tok_s"] for row in rows)
    avg_latency = statistics.mean(row["latency_ms"] for row in rows)
    print(f"\n平均 {avg_tok_s:.1f} tok/s，平均延迟 {avg_latency / 1000:.2f}s（模型 {settings.local_model}）")

    out = ROOT / "docs" / "figures" / "local_bench.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 本机推理基准（实测）",
        "",
        f"- 模型：`{settings.local_model}`（Q4_K_M，约 1 GB）",
        f"- 环境：i5-13420H / 16 GB 内存 / Intel UHD 集显（无 CUDA，纯 CPU 推理）",
        f"- 平均吞吐：**{avg_tok_s:.1f} tok/s**",
        f"- 平均端到端延迟：**{avg_latency / 1000:.2f} s**",
        "",
        "| # | 提示 | 输入 tokens | 输出 tokens | 延迟 (s) | 吞吐 (tok/s) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for index, row in enumerate(rows, 1):
        lines.append(
            f"| {index} | {row['prompt']} | {row['prompt_tokens']} | "
            f"{row['completion_tokens']} | {row['latency_ms'] / 1000:.2f} | "
            f"{row['tok_s']:.1f} |"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已写入 {out}")


if __name__ == "__main__":
    main()
