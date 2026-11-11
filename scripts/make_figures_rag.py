"""论文场景的核心图：三条策略的质量-成本对比，以及门控信号的判别力。

全部数字来自 data/results/rag_*_*.json，脚本只读结果、不重跑模型，
所以评委可以用同一批落盘结果重画这两张图。
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures"
C_LOCAL = "#22c55e"
C_CLOUD = "#4f8cff"
C_GATE = "#f5a524"
C_MUTED = "#8494a7"


def latest(pattern: str) -> dict:
    files = sorted(glob.glob(str(ROOT / "data" / "results" / pattern)))
    if not files:
        raise SystemExit(f"缺少结果：{pattern}")
    # 有重判版本就用重判版本（见 scripts/rescore_rag.py）
    rescored = [f for f in files if "_rescored" in f]
    return json.loads(Path((rescored or files)[-1]).read_text(encoding="utf-8"))["summary"]


def main() -> None:
    local = latest("rag_rule_local_*.json")
    cloud = latest("rag_rule_cloud_*.json")
    gated = latest("rag_rule_gated_*.json")

    names = ["全本地 1.5B", "全云端 DeepSeek", "证据门控路由"]
    colors = [C_LOCAL, C_CLOUD, C_GATE]
    acc = [local["accuracy"], cloud["accuracy"], gated["accuracy"]]
    ev = [local["evidence_recall"], cloud["evidence_recall"], gated["evidence_recall"]]
    cost = [local["cost_per_1k_cny"], cloud["cost_per_1k_cny"], gated["cost_per_1k_cny"]]
    cloud_ratio = [0.0, 1.0, gated["cloud_ratio"]]

    # ---------- 图 1：准确率 / 证据召回 ----------
    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=200)
    x = np.arange(len(names))
    w = 0.34
    b1 = ax.bar(x - w / 2, acc, w, label="答案准确率", color=colors, alpha=0.95)
    b2 = ax.bar(x + w / 2, ev, w, label="证据召回率", color=colors, alpha=0.45, hatch="//")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.015,
                    f"{r.get_height():.1%}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("比例")
    ax.set_title("三类策略：答案准确率与证据召回（n=68，含 6 道不可答题）", fontsize=10, pad=26)
    # 图例放到坐标区上方：柱子占满下半区，放里面会压住数据
    ax.legend(fontsize=8, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.02), ncol=2)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "fig_paper_strategies.png", bbox_inches="tight")
    plt.close(fig)

    # ---------- 图 2：成本—质量 ----------
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
    for i, name in enumerate(names):
        ax.scatter(cost[i], acc[i], s=140, color=colors[i], zorder=3, edgecolor="white", linewidth=1.2)
        ax.annotate(
            f"{name}\n{acc[i]:.1%} / ¥{cost[i]:.2f}",
            (cost[i], acc[i]),
            textcoords="offset points",
            xytext=(10, -4 if i == 1 else 8),
            fontsize=8.5,
            color="#333333",
        )
    ax.axhline(max(acc), color=C_MUTED, linestyle=":", linewidth=0.9)
    ax.set_xlabel("千次问答的云端成本（元）")
    ax.set_ylabel("答案准确率")
    ax.set_xlim(-0.12, max(cost) * 1.45)
    ax.set_ylim(0.55, 0.92)
    ax.set_title("成本—质量：门控路由在更低的成本下取得更高准确率", fontsize=10)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "fig_paper_cost.png", bbox_inches="tight")
    plt.close(fig)

    # ---------- 图 3：门控信号的判别力 ----------
    files = sorted(glob.glob(str(ROOT / "data" / "results" / "rag_rule_local_*.json")))
    rescored = [f for f in files if "_rescored" in f]
    records = [r for r in json.loads(Path((rescored or files)[-1]).read_text(encoding="utf-8"))["records"]
               if r.get("error") is None and r.get("answerable", True)]

    def passed_gate(r: dict) -> bool:
        # 早期结果文件没有 gate_pass 字段，用同样口径重算一遍
        return bool(r.get("gate_pass", (
            r.get("n_anchors", 0) > 0
            and r.get("citation_valid", True)
            and r.get("number_ok", True)
        )))

    passed = [r for r in records if passed_gate(r)]
    failed = [r for r in records if not passed_gate(r)]
    rate_pass = sum(1 for r in passed if r["correct"]) / max(1, len(passed))
    rate_fail = sum(1 for r in failed if r["correct"]) / max(1, len(failed))

    fig, ax = plt.subplots(figsize=(5.6, 3.2), dpi=200)
    bars = ax.bar(
        ["校验通过\n（保留本地）", "校验未通过\n（升级云端）"],
        [rate_pass, rate_fail],
        color=[C_GATE, C_MUTED],
        width=0.5,
    )
    for r, v, n in zip(bars, [rate_pass, rate_fail], [len(passed), len(failed)]):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.02, f"{v:.1%}\n(n={n})",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("本地答案确实正确的比例")
    ax.set_title("门控信号有效：校验通过时本地正确率显著更高", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "fig_paper_gate.png", bbox_inches="tight")
    plt.close(fig)

    print("fig_paper_strategies.png / fig_paper_cost.png / fig_paper_gate.png 已生成")
    print(f"门控通过率 {len(passed) / len(records):.1%}；"
          f"通过时本地正确率 {rate_pass:.1%}，未通过时 {rate_fail:.1%}")
    print(f"云端占比 {cloud_ratio}；成本 {cost}")


if __name__ == "__main__":
    main()
