"""生成技术文档与答辩用的对比图（全部基于已落盘的真实结果，可复现）。

输出：
    docs/figures/fig_router_budget.png   同预算下两种路由的准确率曲线
    docs/figures/fig_ood.png             分布外检验的柱状对比
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
plt.rcParams["figure.dpi"] = 160

BG = "#0a0e14"
FG = "#e6edf3"
GRID = "#1e2a38"
BLUE = "#4f8cff"
GREEN = "#22c55e"
AMBER = "#f5a524"
GREY = "#8494a7"

FIGS = ROOT / "docs" / "figures"


def load(pattern: str, candidates: Path) -> dict:
    files = sorted(glob.glob(str(candidates / "results" / pattern)))
    if not files:
        raise SystemExit(f"缺少结果：{pattern}")
    return {r["id"]: r for r in json.loads(Path(files[-1]).read_text(encoding="utf-8"))["records"]}


def style(ax) -> None:
    ax.set_facecolor(BG)
    ax.tick_params(colors=GREY, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(color=GRID, alpha=0.5, linewidth=0.7)


def budget_curve(scores: np.ndarray, local_ok: np.ndarray, cloud_ok: np.ndarray, budgets):
    order = np.argsort(-scores)
    out = []
    for b in budgets:
        k = int(round(b * len(scores)))
        mask = np.zeros(len(scores), dtype=bool)
        mask[order[:k]] = True
        out.append(float(np.where(mask, cloud_ok, local_ok).mean()))
    return np.array(out)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    model = LocalCorrectnessModel.load(ROOT / "data" / "router_model.json")

    # ---------- 图 1：训练分布内的等预算曲线 ----------
    items = [json.loads(l) for l in (ROOT / "data" / "eval_set.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    local = load("local_20261024*_rescored.json", ROOT / "data")
    cloud = load("cloud_20261024*.json", ROOT / "data")
    y = np.array([1 if local[i["id"]]["correct"] else 0 for i in items])
    c_ok = np.array([1 if cloud[i["id"]]["correct"] else 0 for i in items])
    rule = np.array([rule_features.difficulty(rule_features.extract(i["question"]))[0] for i in items])
    learned = np.array([1.0 - model.predict_proba(i["question"]) for i in items])
    oracle = float(np.where(y.astype(bool) | c_ok.astype(bool), 1, 0).mean())

    budgets = np.linspace(0, 1, 101)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    fig.patch.set_facecolor(BG)
    ax.plot(budgets * 100, budget_curve(rule, y, c_ok, budgets) * 100, "--", color=AMBER, lw=2, label="规则路由")
    ax.plot(budgets * 100, budget_curve(learned, y, c_ok, budgets) * 100, "-", color=BLUE, lw=2.4, label="学习型路由")
    ax.axhline(oracle * 100, color=GREEN, ls=":", lw=1.2, label=f"oracle 上界 {oracle:.1%}")
    ax.set_xlabel("云端调用占比（%）", color=FG, fontsize=10)
    ax.set_ylabel("准确率（%）", color=FG, fontsize=10)
    ax.set_title("训练分布内：两种路由的准确率—预算曲线（n=160）", color=FG, fontsize=11.5, pad=12)
    style(ax)
    leg = ax.legend(loc="lower right", facecolor="#10161f", edgecolor=GRID, labelcolor=FG, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_router_budget.png", facecolor=BG)
    plt.close(fig)

    # ---------- 图 2：分布外检验 ----------
    ood = [json.loads(l) for l in (ROOT / "data" / "eval_ood.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    o_local = load("local_20261025*.json", ROOT / "data")
    o_cloud = load("cloud_20261025*.json", ROOT / "data")
    oy = np.array([1 if o_local[i["id"]]["correct"] else 0 for i in ood])
    oc = np.array([1 if o_cloud[i["id"]]["correct"] else 0 for i in ood])
    o_rule = np.array([rule_features.difficulty(rule_features.extract(i["question"]))[0] for i in ood])
    o_prob = np.array([model.predict_proba(i["question"]) for i in ood])
    rule_route = o_rule >= 0.20
    learned_route = o_prob < model.threshold

    labels = ["全本地\n(0% 云端)", "全云端\n(100%)", "规则路由\nθ=0.20", "学习型路由\np<0.85"]
    values = [
        oy.mean() * 100,
        oc.mean() * 100,
        float(np.where(rule_route, oc, oy).mean()) * 100,
        float(np.where(learned_route, oc, oy).mean()) * 100,
    ]
    ratios = [0, 100, rule_route.mean() * 100, learned_route.mean() * 100]
    colors = [GREY, BLUE, AMBER, GREEN]

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    fig.patch.set_facecolor(BG)
    bars = ax.bar(labels, values, color=colors, width=0.62, alpha=0.9)
    for bar, value, ratio in zip(bars, values, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.8, f"{value:.1f}%",
                ha="center", color=FG, fontsize=11, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width() / 2, 2, f"云端 {ratio:.0f}%",
                ha="center", color="#0a0e14", fontsize=8.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("准确率（%）", color=FG, fontsize=10)
    ax.set_title("分布外检验：40 条全新句式，两端均不重训", color=FG, fontsize=11.5, pad=12)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_ood.png", facecolor=BG)
    plt.close(fig)

    print("已生成：")
    for name in ("fig_router_budget.png", "fig_ood.png"):
        p = FIGS / name
        print(f"  {p}  {p.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
