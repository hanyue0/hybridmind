"""离线重判：不重跑模型，只按新的判分口径重算准确率与汇总。

用途与旧版 ``scripts/rescore.py`` 相同：判分口径一旦调整，必须能在**已有落盘结果**
上重算，而不是重跑一遍模型——否则"同一批复现实验"就变成了"每次跑都换一组数"。

本次调整的原因：论文问答是抽取式任务，gold 是原文里的一个数词，答案常把它写在
句子中间（"2016年12月的 DrQA dump"）。旧口径「取最后一个数字」会得到 12，
把正确答案判错。改为「gold 数值作为完整数词出现」后重新判分。

用法：
    python scripts/rescore_rag.py                 # 重判所有 rag_*.json
    python scripts/rescore_rag.py --pattern "rag_rule_gated_*.json"
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.eval import metrics_rag  # noqa: E402


def load_aliases() -> dict:
    """别名写在数据集里，重判时必须带上——否则「前缀调优」这类中文别名会丢。"""
    path = ROOT / "data" / "qa_paper.jsonl"
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                out[item["id"]] = item.get("aliases", [])
    return out


def rescore(path: Path) -> Path:
    payload = json.loads(path.read_text(encoding="utf-8"))
    aliases = load_aliases()
    records = payload["records"]
    changed = 0
    for r in records:
        if r.get("error") is not None or "prediction" not in r:
            continue
        before = bool(r.get("correct"))
        after = metrics_rag.answer_correct(r["prediction"], r["gold"], aliases.get(r["id"], []))
        r["correct"] = after
        r["correct_before_rescore"] = before
        if before != after:
            changed += 1

    scored = [r for r in records if r.get("error") is None]
    summary = metrics_rag.summarize(scored)
    summary.update({k: v for k, v in payload["summary"].items() if k not in summary})
    summary["rescored_from"] = path.name
    summary["rescored_changed"] = changed
    summary["rescore_rule"] = "numeric: token-boundary substring; text: alias containment"

    out = path.with_name(path.stem + "_rescored.json")
    out.write_text(json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{path.name} → {out.name}：准确率 {summary['accuracy']:.1%}（改动 {changed} 条）")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="rag_*.json")
    args = ap.parse_args()
    files = [f for f in glob.glob(str(ROOT / "data" / "results" / args.pattern))
             if "_rescored" not in f]
    if not files:
        raise SystemExit("没有找到结果文件")
    for f in sorted(files):
        rescore(Path(f))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
