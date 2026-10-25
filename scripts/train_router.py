"""训练学习型路由器，并与规则路由器在同一评测集上对比。

数据来源：
- 题目与标准答案：data/eval_set.jsonl（160 条，封闭答案）
- 本地模型的真实作答：data/results/local_*_rescored.json
- 云端模型的真实作答：data/results/cloud_*.json

关键设计：**用 5 折交叉验证的 out-of-fold 预测做阈值选择**，
而不是拿训练集自己的预测来调参——否则报出来的数字会虚高。

用法：
    python scripts/train_router.py
输出：
    data/router_model.json      运行时模型（纯 Python 推理）
    docs/router_report.md       对比报告
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.router import features as rule_features  # noqa: E402
from app.router.classifier import FEATURE_NAMES, LocalCorrectnessModel, extract_vector  # noqa: E402

DATA = ROOT / "data"
TARGET_ACCURACY = 0.92
SEED = 20261024


def load_latest(pattern: str) -> dict:
    files = sorted(glob.glob(str(DATA / "results" / pattern)))
    if not files:
        raise SystemExit(f"缺少结果文件：{pattern}")
    return {r["id"]: r for r in json.loads(Path(files[-1]).read_text(encoding="utf-8"))["records"]}


def main() -> None:
    items = [json.loads(line) for line in (DATA / "eval_set.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    local = load_latest("local_*_rescored.json")
    cloud = load_latest("cloud_*.json")

    X = np.array([extract_vector(item["question"]) for item in items], dtype=float)
    y = np.array([1 if local[item["id"]]["correct"] else 0 for item in items], dtype=int)
    print(f"样本 {len(items)} 条，本地答对 {y.sum()} 条（{y.mean():.1%}）")

    # ---------- 5 折交叉验证，拿 out-of-fold 概率 ----------
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")),
    ])
    oof = np.zeros(len(items))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for train_idx, test_idx in skf.split(X, y):
        pipeline.fit(X[train_idx], y[train_idx])
        oof[test_idx] = pipeline.predict_proba(X[test_idx])[:, 1]

    auc = roc_auc_score(y, oof)
    print(f"5 折交叉验证 AUC = {auc:.4f}")

    # ---------- 在 OOF 上扫阈值 ----------
    n = len(items)
    cloud_cost = np.array([cloud[item["id"]]["cost_cny"] for item in items])
    local_ok = np.array([1 if local[item["id"]]["correct"] else 0 for item in items])
    cloud_ok = np.array([1 if cloud[item["id"]]["correct"] else 0 for item in items])

    rows = []
    for threshold in np.arange(0.05, 0.96, 0.05):
        to_cloud = oof < threshold          # 本地答对概率低 → 上云
        acc = float(np.where(to_cloud, cloud_ok, local_ok).mean())
        ratio = float(to_cloud.mean())
        cost = float(cloud_cost[to_cloud].sum() / n * 1000)
        rows.append({"threshold": round(float(threshold), 2), "cloud_ratio": ratio,
                     "accuracy": acc, "cost_per_1k": cost})

    feasible = [r for r in rows if r["accuracy"] >= TARGET_ACCURACY]
    if feasible:
        best = min(feasible, key=lambda r: (r["cloud_ratio"], -r["accuracy"]))
    else:
        best = max(rows, key=lambda r: r["accuracy"])

    # ---------- 规则路由基线（同一评测集） ----------
    theta = 0.20
    rule_cloud = np.array([
        1 if rule_features.difficulty(rule_features.extract(item["question"]))[0] >= theta else 0
        for item in items
    ], dtype=bool)
    rule_acc = float(np.where(rule_cloud, cloud_ok, local_ok).mean())
    rule_ratio = float(rule_cloud.mean())
    rule_cost = float(cloud_cost[rule_cloud].sum() / n * 1000)

    oracle = float(np.where(local_ok.astype(bool) | cloud_ok.astype(bool), 1, 0).mean())

    # ---------- 全量训练并导出 ----------
    pipeline.fit(X, y)
    scaler: StandardScaler = pipeline.named_steps["scaler"]
    clf: LogisticRegression = pipeline.named_steps["clf"]
    model = LocalCorrectnessModel(
        feature_names=list(FEATURE_NAMES),
        mean=scaler.mean_.tolist(),
        scale=scaler.scale_.tolist(),
        coef=clf.coef_[0].tolist(),
        intercept=float(clf.intercept_[0]),
        threshold=float(best["threshold"]),
        auc=float(auc),
        n_samples=int(n),
    )
    model.save(DATA / "router_model.json")

    # 校验导出的纯 Python 推理与 sklearn 一致
    drift = max(abs(model.predict_proba(item["question"]) - p) for item, p in zip(items, pipeline.predict_proba(X)[:, 1]))
    print(f"导出模型与 sklearn 的最大偏差：{drift:.6f}")

    # ---------- 报告 ----------
    lines = [
        "# 学习型路由器训练报告",
        "",
        f"- 样本：{n} 条（本地答对 {int(y.sum())} 条，占 {y.mean():.1%}）",
        f"- 特征：{len(FEATURE_NAMES)} 维（可解释的题面统计量，无嵌入）",
        f"- 模型：标准化 + L2 逻辑回归，5 折分层交叉验证",
        f"- **交叉验证 AUC = {auc:.4f}**",
        f"- 选择阈值：{model.threshold:.2f}（约束：准确率 ≥ {TARGET_ACCURACY:.0%}，最小化云端占比）",
        f"- 导出校验：与 sklearn 预测最大偏差 {drift:.6f}",
        "",
        "## 三条策略对比（同一评测集、同一判分口径）",
        "",
        "| 策略 | 云端占比 | 准确率 | 千次成本(元) |",
        "| --- | --- | --- | --- |",
        f"| 全本地 | 0% | {local_ok.mean():.1%} | 0.000 |",
        f"| 全云端 | 100% | {cloud_ok.mean():.1%} | {cloud_cost.sum() / n * 1000:.3f} |",
        f"| 规则路由 θ=0.20 | {rule_ratio:.0%} | {rule_acc:.1%} | {rule_cost:.3f} |",
        f"| **学习型路由** | **{best['cloud_ratio']:.0%}** | **{best['accuracy']:.1%}** | **{best['cost_per_1k']:.3f}** |",
        f"| oracle 上界 | — | {oracle:.1%} | — |",
        "",
        "## 阈值扫描（out-of-fold 预测）",
        "",
        "| 本地答对概率阈值 | 云端占比 | 准确率 | 千次成本(元) |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        mark = " **←**" if row["threshold"] == best["threshold"] else ""
        lines.append(
            f"| {row['threshold']:.2f} | {row['cloud_ratio']:.0%} | {row['accuracy']:.1%} | {row['cost_per_1k']:.3f}{mark} |"
        )

    lines += [
        "",
        "## 特征权重（标准化后，正=倾向于本地答对）",
        "",
        "| 特征 | 系数 |",
        "| --- | --- |",
    ]
    for name, weight in sorted(zip(FEATURE_NAMES, model.coef), key=lambda x: -abs(x[1])):
        lines.append(f"| `{name}` | {weight:+.3f} |")

    lines += [
        "",
        "## 已知局限",
        "",
        "1. **阈值在 OOF 上选取，再用于同一批 OOF 评估，存在轻度乐观偏差**；"
        "更严谨的做法是留出独立测试集，样本量足够后补做。",
        "2. 160 条样本偏少，AUC 的置信区间较宽（可用 bootstrap 估计，待补）。",
        "3. 特征全部来自题面文本，不含对话历史；多轮场景需要另加状态特征。",
    ]
    (ROOT / "docs" / "router_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n| 策略 | 云端占比 | 准确率 | 千次成本(元) |")
    print("| --- | --- | --- | --- |")
    print(f"| 全本地 | 0% | {local_ok.mean():.1%} | 0.000 |")
    print(f"| 全云端 | 100% | {cloud_ok.mean():.1%} | {cloud_cost.sum() / n * 1000:.3f} |")
    print(f"| 规则路由 θ=0.20 | {rule_ratio:.0%} | {rule_acc:.1%} | {rule_cost:.3f} |")
    print(f"| 学习型路由 | {best['cloud_ratio']:.0%} | {best['accuracy']:.1%} | {best['cost_per_1k']:.3f} |")
    print(f"| oracle | — | {oracle:.1%} | — |")
    print(f"\n模型已保存 {DATA / 'router_model.json'}")
    print(f"报告已保存 {ROOT / 'docs' / 'router_report.md'}")


if __name__ == "__main__":
    main()
