"""在「同等云端预算」下对比规则路由与学习型路由。

两条曲线的构造方式一致：把 160 条题目按各自的打分排序，
取出最应该上云的 B% 送去云端，其余留本地，计算准确率。
这样比较的是「花同样多的云端调用，谁更准」，而不是各自挑自己的阈值。

输出：
    docs/figures/router_curve.png
    docs/figures/router_budget.md
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.router import features as rule_features  # noqa: E402
from app.router.classifier import LocalCorrectnessModel  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_latest(pattern: str) -> dict:
    files = sorted(glob.glob(str(ROOT / "data" / "results" / pattern)))
    return {r["id"]: r for r in json.loads(Path(files[-1]).read_text(encoding="utf-8"))["records"]}


def curve(scores: np.ndarray, local_ok: np.ndarray, cloud_ok: np.ndarray, budgets: np.ndarray):
    """scores 越大表示越应该上云。返回每个预算下的准确率。"""
    order = np.argsort(-scores)  # 最该上云的在前
    accuracies = []
    for budget in budgets:
        k = int(round(budget * len(scores)))
        to_cloud = np.zeros(len(scores), dtype=bool)
        to_cloud[order[:k]] = True
        accuracies.append(float(np.where(to_cloud, cloud_ok, local_ok).mean()))
    return np.array(accuracies)


def main() -> None:
    items = [json.loads(line) for line in (ROOT / "data" / "eval_set.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    local = load_latest("local_*_rescored.json")
    cloud = load_latest("cloud_*.json")
    model = LocalCorrectnessModel.load(ROOT / "data" / "router_model.json")

    local_ok = np.array([1 if local[i["id"]]["correct"] else 0 for i in items])
    cloud_ok = np.array([1 if cloud[i["id"]]["correct"] else 0 for i in items])

    rule_scores = np.array([rule_features.difficulty(rule_features.extract(i["question"]))[0] for i in items])
    learned_scores = np.array([1.0 - model.predict_proba(i["question"]) for i in items])  # 越不该留本地越该上云

    budgets = np.linspace(0, 1, 101)  # 每 1% 一个点，便于按百分比直接索引
    rule_acc = curve(rule_scores, local_ok, cloud_ok, budgets)
    learned_acc = curve(learned_scores, local_ok, cloud_ok, budgets)
    oracle = float(np.where(local_ok.astype(bool) | cloud_ok.astype(bool), 1, 0).mean())

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=160)
    ax.plot(budgets * 100, rule_acc * 100, "--", color="#f5a524", linewidth=2, label="规则路由（题面关键词）")
    ax.plot(budgets * 100, learned_acc * 100, "-", color="#4f8cff", linewidth=2.4, label="学习型路由（本地答对概率）")
    ax.axhline(oracle * 100, color="#22c55e", linewidth=1.2, linestyle=":", label=f"oracle 上界 {oracle:.1%}")
    ax.scatter([0, 100], [local_ok.mean() * 100, cloud_ok.mean() * 100],
               color="#e6edf3", edgecolor="#0a0e14", zorder=5, s=38)
    ax.annotate(f"全本地 {local_ok.mean():.1%}", (0, local_ok.mean() * 100), textcoords="offset points",
                xytext=(8, -16), fontsize=9, color="#b8c6d6")
    ax.annotate(f"全云端 {cloud_ok.mean():.1%}", (100, cloud_ok.mean() * 100), textcoords="offset points",
                xytext=(-72, 8), fontsize=9, color="#b8c6d6")
    ax.set_xlabel("云端调用占比（%）")
    ax.set_ylabel("端到端准确率（%）")
    ax.set_title("同等云端预算下，两种路由策略的准确率对比（n=160）")
    ax.grid(alpha=0.18)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_ylim(55, 100)
    fig.tight_layout()

    out = ROOT / "docs" / "figures" / "router_curve.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)

    rows = []
    for budget in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        idx = int(round(budget * 100))
        rows.append((budget, rule_acc[idx], learned_acc[idx], learned_acc[idx] - rule_acc[idx]))

    lines = [
        "# 同等云端预算下的路由质量对比",
        "",
        "| 云端预算 | 规则路由 | 学习型路由 | 提升 |",
        "| --- | --- | --- | --- |",
    ]
    for budget, r, l, delta in rows:
        lines.append(f"| {budget:.0%} | {r:.1%} | {l:.1%} | {delta:+.1f} pt |")
    lines += ["", f"oracle 上界：{oracle:.1%}", "", f"![曲线](router_curve.png)"]
    (ROOT / "docs" / "figures" / "router_budget.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n| 云端预算 | 规则路由 | 学习型路由 | 提升 |")
    print("| --- | --- | --- | --- |")
    for budget, r, l, delta in rows:
        print(f"| {budget:.0%} | {r:.1%} | {l:.1%} | {delta:+.1f} pt |")
    mean_gain = float(np.mean(learned_acc - rule_acc))
    print(f"\n全预算平均提升：{mean_gain:+.2f} pt")
    print(f"图已保存 {out}")


if __name__ == "__main__":
    main()
