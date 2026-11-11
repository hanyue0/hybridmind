"""训练「论文场景」的学习型路由：14 维题面特征 + 4 维检索特征。

为什么单独一份模型，而不是把两批数据合并训练：
- 算术题的 160 条没有检索环节，4 维检索特征恒为 0，合在一起训会让这些维度的
  系数被「有没有检索」这个混杂因素带偏；
- 两个场景的阈值语义也不同（论文题的本地答对率天然更低）。
两份模型、两份报告，评测表也分开列，读者不会混淆。

数据来源：
- 题目与 gold：data/qa_paper.jsonl
- 本地真实作答：data/results/rag_*_local_*.json
- 云端真实作答：data/results/rag_*_cloud_*.json

用法：
    python scripts/train_router_rag.py
输出：
    data/router_model_paper.json
    docs/router_report_paper.md
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
from app.router import policy  # noqa: E402
from app.router.classifier import (  # noqa: E402
    FEATURE_NAMES,
    RAG_FEATURE_NAMES,
    LocalCorrectnessModel,
    extract_named,
)

DATA = ROOT / "data"
NAMES = list(FEATURE_NAMES) + list(RAG_FEATURE_NAMES)
SEED = 20261111
TARGET_ACCURACY = 0.90


def load_latest(pattern: str) -> dict:
    files = sorted(glob.glob(str(DATA / "results" / pattern)))
    if not files:
        raise SystemExit(f"缺少结果文件：{pattern}")
    payload = json.loads(Path(files[-1]).read_text(encoding="utf-8"))
    return {r["id"]: r for r in payload["records"] if r.get("error") is None}


def build_matrix(items: list[dict], local: dict, cloud: dict) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    rows, kept = [], []
    for item in items:
        rec_local = local.get(item["id"])
        rec_cloud = cloud.get(item["id"])
        if not rec_local or not rec_cloud:
            continue
        feats = extract_named(item["question"])
        feats.update(
            {k: float(rec_local.get("retrieval_features", {}).get(k, 0.0)) for k in RAG_FEATURE_NAMES}
        )
        rows.append([feats[n] for n in NAMES])
        kept.append({**item, "local_ok": bool(rec_local["correct"]), "cloud_ok": bool(rec_cloud["correct"]),
                     "cloud_cost": float(rec_cloud.get("cost_cny", 0.0)),
                     "rag": rec_local.get("retrieval_features", {})})
    X = np.array(rows, dtype=float)
    y = np.array([1 if r["local_ok"] else 0 for r in kept], dtype=int)
    return X, y, kept


def main() -> None:
    items = [
        json.loads(line)
        for line in (DATA / "qa_paper.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    local = load_latest("rag_*_local_*.json")
    cloud = load_latest("rag_*_cloud_*.json")
    X, y, kept = build_matrix(items, local, cloud)
    n = len(kept)
    if n < 20:
        raise SystemExit(f"样本只有 {n} 条，先跑完 local 与 cloud 两组评测")
    print(f"样本 {n} 条，本地答对 {int(y.sum())} 条（{y.mean():.1%}）")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced")),
    ])
    oof = np.zeros(n)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for train_idx, test_idx in skf.split(X, y):
        pipe.fit(X[train_idx], y[train_idx])
        oof[test_idx] = pipe.predict_proba(X[test_idx])[:, 1]
    auc = roc_auc_score(y, oof)
    print(f"5 折交叉验证 AUC = {auc:.4f}")

    local_ok = np.array([1 if r["local_ok"] else 0 for r in kept])
    cloud_ok = np.array([1 if r["cloud_ok"] else 0 for r in kept])
    cloud_cost = np.array([r["cloud_cost"] for r in kept])
    oracle = float(np.where(local_ok | cloud_ok, 1, 0).mean())

    # 阈值扫描：本地答对概率 < 阈值 → 上云
    rows = []
    for t in np.arange(0.05, 0.96, 0.05):
        to_cloud = oof < t
        rows.append(
            {
                "threshold": round(float(t), 2),
                "cloud_ratio": float(to_cloud.mean()),
                "accuracy": float(np.where(to_cloud, cloud_ok, local_ok).mean()),
                "cost_per_1k": float(cloud_cost[to_cloud].sum() / n * 1000),
            }
        )
    feasible = [r for r in rows if r["accuracy"] >= TARGET_ACCURACY]
    best = (min(feasible, key=lambda r: (r["cloud_ratio"], -r["accuracy"])) if feasible
            else max(rows, key=lambda r: r["accuracy"]))

    # 规则基线：带检索特征的规则路由（θ 扫一遍取最优工作点，保证对比公平）
    rule_rows = []
    for theta in np.arange(0.2, 0.8, 0.05):
        to_cloud = np.array(
            [1 if policy.decide(r["question"], float(theta), rag=r["rag"]).route == "cloud" else 0
             for r in kept], dtype=bool
        )
        rule_rows.append(
            {
                "theta": round(float(theta), 2),
                "cloud_ratio": float(to_cloud.mean()),
                "accuracy": float(np.where(to_cloud, cloud_ok, local_ok).mean()),
                "cost_per_1k": float(cloud_cost[to_cloud].sum() / n * 1000),
            }
        )
    rule_feasible = [r for r in rule_rows if r["accuracy"] >= TARGET_ACCURACY]
    rule_best = (min(rule_feasible, key=lambda r: (r["cloud_ratio"], -r["accuracy"])) if rule_feasible
                 else max(rule_rows, key=lambda r: r["accuracy"]))

    # 全量训练并导出
    pipe.fit(X, y)
    scaler: StandardScaler = pipe.named_steps["scaler"]
    clf: LogisticRegression = pipe.named_steps["clf"]
    model = LocalCorrectnessModel(
        feature_names=NAMES,
        mean=scaler.mean_.tolist(),
        scale=scaler.scale_.tolist(),
        coef=clf.coef_[0].tolist(),
        intercept=float(clf.intercept_[0]),
        threshold=float(best["threshold"]),
        auc=float(auc),
        n_samples=n,
    )
    model.save(DATA / "router_model_paper.json")

    # 导出模型与 sklearn 的一致性校验（用带检索特征的向量）
    drift = 0.0
    for i, rec in enumerate(kept):
        p = model.predict_proba(rec["question"], rec["rag"])
        drift = max(drift, abs(p - pipe.predict_proba(X[i : i + 1])[0, 1]))
    print(f"导出模型与 sklearn 的最大偏差：{drift:.6f}")

    lines = [
        "# 论文场景 · 学习型路由训练报告",
        "",
        f"- 样本：{n} 条（本地答对 {int(y.sum())} 条，占 {y.mean():.1%}）",
        f"- 特征：14 维题面 + 4 维检索（top1 分数 / top1-top2 差值 / 命中页数 / 问题词覆盖率）",
        f"- 模型：标准化 + L2 逻辑回归，5 折分层交叉验证",
        f"- **交叉验证 AUC = {auc:.4f}**",
        f"- 选择阈值：{model.threshold:.2f}（约束：准确率 ≥ {TARGET_ACCURACY:.0%}，最小化云端占比）",
        f"- 导出校验：与 sklearn 最大偏差 {drift:.6f}",
        "",
        "## 策略对比（同一批题、同一判分口径）",
        "",
        "| 策略 | 云端占比 | 准确率 | 千次成本(元) |",
        "| --- | --- | --- | --- |",
        f"| 全本地 | 0% | {local_ok.mean():.1%} | 0.000 |",
        f"| 全云端 | 100% | {cloud_ok.mean():.1%} | {cloud_cost.sum() / n * 1000:.3f} |",
        f"| 规则路由（含检索特征）θ={rule_best['theta']:.2f} | {rule_best['cloud_ratio']:.0%} | "
        f"{rule_best['accuracy']:.1%} | {rule_best['cost_per_1k']:.3f} |",
        f"| **学习型路由** | **{best['cloud_ratio']:.0%}** | **{best['accuracy']:.1%}** | "
        f"**{best['cost_per_1k']:.3f}** |",
        f"| oracle 上界 | — | {oracle:.1%} | — |",
        "",
        "## 阈值扫描（out-of-fold 预测）",
        "",
        "| 本地答对概率阈值 | 云端占比 | 准确率 | 千次成本(元) |",
        "| --- | --- | --- | --- |",
    ]
    for r in rows:
        mark = " **←**" if r["threshold"] == best["threshold"] else ""
        lines.append(f"| {r['threshold']:.2f} | {r['cloud_ratio']:.0%} | {r['accuracy']:.1%} | {r['cost_per_1k']:.3f}{mark} |")

    lines += ["", "## 特征权重（标准化后，正=本地更可能答对）", "", "| 特征 | 系数 |", "| --- | --- |"]
    for name, weight in sorted(zip(NAMES, model.coef), key=lambda x: -abs(x[1])):
        lines.append(f"| `{name}` | {weight:+.3f} |")

    lines += [
        "",
        "## 已知局限",
        "",
        "1. 样本 60 余条，AUC 置信区间较宽；阈值在 OOF 上选、又在同一批 OOF 上评估，存在轻度乐观偏差。",
        "2. 检索特征依赖固定的分块与 top-k，换分块策略需重训。",
        "3. 拒答题（原文未提及）的「本地是否答对」标签由拒答判定给出，不是人类标注。",
    ]
    (ROOT / "docs" / "router_report_paper.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n| 策略 | 云端占比 | 准确率 | 千次成本(元) |")
    print("| --- | --- | --- | --- |")
    print(f"| 全本地 | 0% | {local_ok.mean():.1%} | 0.000 |")
    print(f"| 全云端 | 100% | {cloud_ok.mean():.1%} | {cloud_cost.sum() / n * 1000:.3f} |")
    print(f"| 规则路由 θ={rule_best['theta']:.2f} | {rule_best['cloud_ratio']:.0%} | {rule_best['accuracy']:.1%} | {rule_best['cost_per_1k']:.3f} |")
    print(f"| 学习型路由 | {best['cloud_ratio']:.0%} | {best['accuracy']:.1%} | {best['cost_per_1k']:.3f} |")
    print(f"| oracle | — | {oracle:.1%} | — |")
    print(f"\n模型 {DATA / 'router_model_paper.json'}\n报告 {ROOT / 'docs' / 'router_report_paper.md'}")


if __name__ == "__main__":
    main()
