"""用最新判分口径重算已有评测结果——不重跑模型，只重判。

判分口径会随评审反馈调整（例如"商 18，余数 4"应判对），
重新跑一遍模型要十几分钟且结果受温漂影响，重判则秒级且可复现。

用法：
    python scripts/rescore.py data/results/local_20261024-223351.json
    python scripts/rescore.py            # 默认重判最新的一个结果文件
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.eval import metrics  # noqa: E402

DATASET = ROOT / "data" / "eval_set.jsonl"


def load_items() -> dict:
    items = {}
    with DATASET.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                record = json.loads(line)
                items[record["id"]] = record
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", nargs="?", type=Path, default=None)
    args = parser.parse_args()

    if args.result is None:
        candidates = sorted((ROOT / "data" / "results").glob("*.json"))
        if not candidates:
            raise SystemExit("没有可重判的结果文件")
        args.result = candidates[-1]

    items = load_items()
    data = json.loads(args.result.read_text(encoding="utf-8"))

    by_tier = collections.defaultdict(lambda: [0, 0])
    changed = 0
    for record in data["records"]:
        item = items[record["id"]]
        new_verdict = metrics.is_correct(
            record["prediction"], item["answer"], item.get("aliases", [])
        )
        if new_verdict != record["correct"]:
            changed += 1
        record["correct"] = new_verdict
        bucket = by_tier[item["tier"]]
        bucket[1] += 1
        if new_verdict:
            bucket[0] += 1

    scored = [r for r in data["records"] if not r.get("error")]
    accuracy = sum(1 for r in scored if r["correct"]) / len(scored) if scored else 0.0
    data["summary"]["accuracy"] = round(accuracy, 4)
    data["summary"]["rescored"] = True

    print(f"文件：{args.result.name}")
    print(f"样本 {len(scored)} 条，判分口径变更导致 {changed} 条改判")
    print("\n按难度档：")
    for tier in sorted(by_tier):
        ok, total = by_tier[tier]
        print(f"  tier{tier}: {ok}/{total} = {ok / total:.1%}")
    print(f"\n总体准确率：{accuracy:.1%}")

    out = args.result.with_name(args.result.stem + "_rescored.json")
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {out}")


if __name__ == "__main__":
    main()
