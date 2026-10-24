"""评测入口。

一条命令跑一组对照，产出可直接进文档的数据：
    python -m app.eval.run_eval --mode local
    python -m app.eval.run_eval --mode cloud
    python -m app.eval.run_eval --mode auto
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

from ..config import ROOT, settings
from ..pipeline import HybridPipeline, Ledger
from ..providers import OllamaProvider, OpenAICompatProvider, ProviderError
from . import metrics

DEFAULT_DATASET = ROOT / "data" / "eval_set.jsonl"


def load_dataset(path: Path) -> List[dict]:
    items: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def build_pipeline(theta: float | None = None) -> HybridPipeline:
    if theta is not None:
        settings.theta_local = theta
    local = OllamaProvider(settings.ollama_host, settings.local_model)
    cloud = OpenAICompatProvider(
        settings.cloud_base_url,
        settings.cloud_api_key,
        settings.cloud_model,
        settings.price_of(settings.cloud_model),
        timeout_s=settings.timeout_s,
    )
    return HybridPipeline(local, cloud, settings, Ledger())


def run(mode: str, dataset: Path, limit: int | None, theta: float | None) -> Dict:
    items = load_dataset(dataset)
    if limit:
        items = items[:limit]
    pipeline = build_pipeline(theta)

    records: List[dict] = []
    for index, item in enumerate(items, 1):
        started = time.perf_counter()
        try:
            response = pipeline.answer([{"role": "user", "content": item["question"]}], mode=mode)
            prediction = response.answer
            trace = response.trace
            usage = response.usage
        except ProviderError as exc:
            records.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "gold": item["answer"],
                    "prediction": "",
                    "correct": False,
                    "error": str(exc),
                }
            )
            print(f"[{index}/{len(items)}] {item['id']} ERROR {exc}", file=sys.stderr)
            continue

        correct = metrics.is_correct(prediction, item["answer"], item.get("aliases", []))
        records.append(
            {
                "id": item["id"],
                "question": item["question"],
                "gold": item["answer"],
                "prediction": prediction,
                "correct": correct,
                "route": trace.route,
                "difficulty": trace.difficulty,
                "escalated": trace.escalated,
                "confidence": trace.confidence,
                "latency_ms": trace.latency_ms,
                "cost_cny": usage.cost_cny,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "wall_ms": round((time.perf_counter() - started) * 1000, 1),
                "error": None,
            }
        )
        print(
            f"[{index}/{len(items)}] {item['id']} {'✔' if correct else '✘'} "
            f"{trace.route} {trace.latency_ms:.0f}ms",
            file=sys.stderr,
        )

    scored = [r for r in records if r.get("error") is None]
    accuracy = sum(1 for r in scored if r["correct"]) / len(scored) if scored else 0.0
    latencies = [r["latency_ms"] for r in scored]
    summary = {
        "mode": mode,
        "theta": settings.theta_local,
        "dataset": dataset.name,
        "n": len(records),
        "n_scored": len(scored),
        "accuracy": round(accuracy, 4),
        "cost_cny": round(sum(r["cost_cny"] for r in scored), 6),
        "cost_per_1k_cny": round(
            sum(r["cost_cny"] for r in scored) / max(len(scored), 1) * 1000, 4
        ),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        "p50_latency_ms": round(metrics.percentile(latencies, 0.5), 1),
        "p95_latency_ms": round(metrics.percentile(latencies, 0.95), 1),
        "local_ratio": round(
            sum(1 for r in scored if r["route"] == "local") / len(scored), 4
        )
        if scored
        else 0.0,
        "escalations": sum(1 for r in scored if r["escalated"]),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return {"summary": summary, "records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description="HybridMind 评测脚本")
    parser.add_argument("--mode", choices=["local", "cloud", "auto"], default="auto")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条，控制云端花费")
    parser.add_argument("--theta", type=float, default=None, help="覆盖本地阈值，用于扫阈值")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    result = run(args.mode, args.dataset, args.limit, args.theta)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = args.out or (ROOT / "data" / "results" / f"{args.mode}_{stamp}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = result["summary"]
    print("\n| 模式 | 样本 | 准确率 | 单条成本(元) | 千次成本(元) | 平均延迟(ms) | P95(ms) | 本地占比 |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    print(
        f"| {summary['mode']} | {summary['n_scored']} | {summary['accuracy']:.1%} | "
        f"{summary['cost_cny'] / max(summary['n_scored'], 1):.6f} | "
        f"{summary['cost_per_1k_cny']:.2f} | {summary['avg_latency_ms']:.0f} | "
        f"{summary['p95_latency_ms']:.0f} | {summary['local_ratio']:.0%} |"
    )
    print(f"\n结果已写入 {out}")


if __name__ == "__main__":
    main()
