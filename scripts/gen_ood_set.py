"""生成分布外（OOD）评测集：40 条全新句式的封闭答案题目。

用途：检验路由策略的**泛化**能力。

动机：原评测集（160 条）的题目形态是我自己写的生成器产出的，
而规则路由的特征词表是照着这套形态调的——两者同源，
规则在它上面的表现存在"作者偏差"，不能代表真实流量。

因此单独造一批**句式与词汇都不重叠**的题目，用来回答一个问题：
换一种说法，谁还能判断对？模型不在这批数据上重训。
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "eval_ood.jsonl"
rng = random.Random(20261025)

records: list[dict] = []


def add(question: str, answer, kind: str = "numeric", aliases=None) -> None:
    records.append(
        {
            "id": f"o{len(records) + 1:03d}",
            "question": question,
            "answer": str(answer),
            "aliases": aliases or [],
            "tier": 0,
            "type": kind,
        }
    )


# 句式 1：购物找零（不含「计算」「多少元」等原词表高频词）
for _ in range(6):
    price, count = rng.randint(6, 19), rng.randint(2, 7)
    money = price * count + rng.randint(10, 80)  # 保证钱够花、答案为正
    add(f"小林带了 {money} 元，买了 {count} 本单价 {price} 元的练习册，付完钱后口袋里还剩几元？", money - price * count)

# 句式 2：分组平均
for _ in range(6):
    total, girls, groups = rng.randint(36, 60), rng.randint(10, 20), rng.randint(3, 6)
    boys = total - girls
    boys -= boys % groups
    add(f"一个班 {total} 人，男生 {boys} 人、女生 {girls} 人。把男生平均分成 {groups} 组，每组几人？", boys // groups)

# 句式 3：反向求解（问未知数）
for _ in range(6):
    a, b = rng.randint(7, 40), rng.randint(50, 120)
    add(f"有一个数，它和 {a} 相加的结果正好是 {b}，这个数是几？", b - a)

# 句式 4：时间跨度
for _ in range(6):
    h1, m1 = rng.randint(7, 10), rng.choice([5, 10, 15, 20, 30])
    h2, m2 = rng.randint(11, 16), rng.choice([0, 15, 30, 45])
    minutes = (h2 * 60 + m2) - (h1 * 60 + m1)
    add(f"上午 {h1} 点 {m1} 分到下午 {h2} 点 {m2} 分，中间过了多长时间？请用分钟数回答。", minutes)

# 句式 5：倍数关系
for _ in range(6):
    base, times = rng.randint(12, 40), rng.randint(2, 6)
    add(f"乙数是 {base}，甲数是乙数的 {times} 倍，甲数减掉 {base} 之后还剩？", base * times - base)

# 句式 6：和差问题
for _ in range(5):
    diff, total = rng.randint(4, 20), rng.randint(40, 120)
    if (total + diff) % 2:
        total += 1
    add(f"两个人一共收集了 {total} 张卡，其中一人比另一人多 {diff} 张，收集多的那人有几张？", (total + diff) // 2)

# 短答题：常识与定义
facts = [
    ("地球绕太阳转一圈大约需要多长时间？（用「天」作答）", "365", ["365"]),
    ("一个标准足球队同时在场的球员有几人？", "11", ["十一"]),
    ("农历八月十五是中国的哪个传统节日？", "中秋节", ["中秋"]),
    ("人体正常体温大约是多少摄氏度？", "37", ["三十七"]),
    ("一天有多少小时？", "24", ["二十四"]),
]
for question, answer, aliases in facts:
    add(question, answer, kind="short", aliases=aliases)

OUT.write_text(
    "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8"
)
print(f"写入 {len(records)} 条 OOD 题目 -> {OUT}")
print("样例如下：")
for r in records[:3] + records[-2:]:
    print(" ", r["question"], "=>", r["answer"])
