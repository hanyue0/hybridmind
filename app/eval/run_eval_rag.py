"""论文问答评测入口。

与 ``run_eval`` 并列而不是替换：算术集验证「路由值不值得做」，
论文集验证「路由在真实场景里能不能同时保住准确率和证据质量」。
两组数据、两张表，都进技术文档。

    python -m app.eval.run_eval_rag --mode local
    python -m app.eval.run_eval_rag --mode cloud
    python -m app.eval.run_eval_rag --mode auto --theta 0.30 --k 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

from ..config import ROOT, settings
from ..pipeline import HybridPipeline, Ledger
from ..providers import OpenAICompatProvider, ProviderError, build_local_provider
from ..rag.qa import PaperQA, PaperStore
from . import metrics_rag

DEFAULT_DATASET = ROOT / "data" / "qa_paper.jsonl"


def load_dataset(path: Path) -> List[dict]:
    items: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def build_pipeline() -> HybridPipeline:
    """论文评测用 18 维（含检索特征）的模型文件。"""
    paper_settings = replace(settings, router_model_path=settings.router_model_path_paper)
    local = build_local_provider(paper_settings)
    cloud = OpenAICompatProvider(
        paper_settings.cloud_base_url,
        paper_settings.cloud_api_key,
        paper_settings.cloud_model,
        paper_settings.price_of(paper_settings.cloud_model),
        timeout_s=max(paper_settings.timeout_s, 120),
    )
    return HybridPipeline(local, cloud, paper_settings, Ledger())


def run(mode: str, dataset: Path, limit: Optional[int], theta: Optional[float],
        k: int, router: Optional[str]) -> Dict:
    if theta is not None:
        settings.theta_local = theta
    if router:
        settings.router_mode = router

    items = load_dataset(dataset)
    if limit:
        items = items[:limit]

    pipeline = build_pipeline()          # 决策器在构造时读取 settings，故先改 settings
    store = PaperStore(ROOT / "data" / "papers")
    qa = PaperQA(pipeline, store, top_k=k)

    records: List[dict] = []
    for index, item in enumerate(items, 1):
        started = time.perf_counter()
        try:
            result = qa.ask(item["paper"], item["question"], mode=mode, k=k)
        except ProviderError as exc:
            records.append({**item, "prediction": "", "correct": False, "error": str(exc)})
            print(f"[{index}/{len(items)}] {item['id']} ERROR {exc}", file=sys.stderr)
            continue

        correct = metrics_rag.answer_correct(
            result.answer, item["answer"], item.get("aliases", [])
        )
        pages = [a.page for a in result.anchors]
        hit = metrics_rag.evidence_hit(pages, item.get("gold_pages", []))
        report = result.report

        records.append(
            {
                "id": item["id"],
                "paper": item["paper"],
                "type": item.get("type", "unknown"),
                "question": item["question"],
                "gold": item["answer"],
                "gold_pages": item.get("gold_pages", []),
                "answerable": item.get("answerable", True),
                "prediction": result.answer,
                "correct": correct,
                "anchor_pages": pages,
                "n_anchors": len(result.anchors),
                "evidence_hit": hit,
                "citation_valid": report.citation_valid,
                "gate_pass": report.gate_pass,
                "number_ok": report.number_ok,
                "term_ok": report.term_ok,
                "refused": report.refused,
                "escalated": result.escalated,
                "gate_reason": result.gate_reason,
                "invalid_refs": report.invalid_refs,
                "missing_numbers": report.missing_numbers,
                "retrieval_features": result.retrieval_features,
                "top_chunk": result.hits[0].chunk.id if result.hits else None,
                "route": result.trace.route,
                "difficulty": result.trace.difficulty,
                "escalated": result.trace.escalated,
                "latency_ms": result.trace.latency_ms,
                "cost_cny": result.usage.cost_cny,
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "wall_ms": round((time.perf_counter() - started) * 1000, 1),
                "error": None,
            }
        )
        print(
            f"[{index}/{len(items)}] {item['id']} {'✔' if correct else '✘'} "
            f"证据{'✔' if hit else '✘'} {result.trace.route} "
            f"{result.trace.latency_ms:.0f}ms",
            file=sys.stderr,
        )

    scored = [r for r in records if r.get("error") is None]
    summary = metrics_rag.summarize(scored)
    summary.update(
        {
            "mode": mode,
            "router": settings.router_mode,
            "theta": settings.theta_local,
            "dataset": dataset.name,
            "top_k": k,
            "n_total": len(records),
            "n_scored": len(scored),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    return {"summary": summary, "records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description="论文问答评测")
    parser.add_argument("--mode", choices=["local", "cloud", "auto", "gated"], default="gated")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--theta", type=float, default=None)
    parser.add_argument("--k", type=int, default=5, help="检索返回的片段数")
    parser.add_argument("--router", choices=["rule", "learned"], default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    result = run(args.mode, args.dataset, args.limit, args.theta, args.k, args.router)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = f"{settings.router_mode}_{args.mode}"
    out = args.out or (ROOT / "data" / "results" / f"rag_{tag}_{stamp}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    s = result["summary"]
    print("\n| 指标 | 值 |")
    print("| --- | --- |")
    for key in ("n_scored", "accuracy", "evidence_recall", "citation_valid_rate",
                "number_consistency", "anchored_rate", "false_refusal_rate",
                "cloud_ratio", "cost_per_1k_cny", "avg_latency_ms"):
        print(f"| {key} | {s.get(key)} |")
    print("\n分题型：")
    for t, b in (s.get("by_type") or {}).items():
        print(f"  {t}: n={b['n']} 准确率={b['accuracy']:.1%} 证据召回={b['evidence_recall']:.1%}")
    print(f"\n结果已写入 {out}")


if __name__ == "__main__":
    main()
