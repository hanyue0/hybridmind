"""生成评测集：150 条封闭答案题目，含四个难度档。

为什么用「程序化生成 + 精确计算」的题目：
封闭答案意味着判分不需要大模型当裁判，评委可以逐条复算，评测结论才站得住。
随机种子固定，任何人重跑都能得到同一份数据。

用法：
    python scripts/gen_eval_set.py
输出：
    data/eval_set.jsonl
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "eval_set.jsonl"
SEED = 20261024
rng = random.Random(SEED)


def item(idx: int, question: str, answer: str, tier: int, aliases: List[str] | None = None) -> dict:
    return {
        "id": f"q{idx:03d}",
        "question": question,
        "answer": str(answer),
        "aliases": aliases or [],
        "tier": tier,
        "type": "numeric",
    }


def t1_single_step(start: int) -> List[dict]:
    """一档：单步运算，小模型几乎必对。"""
    out = []
    for _ in range(20):
        a, b = rng.randint(12, 99), rng.randint(3, 19)
        out.append(item(0, f"计算 {a} + {b} 的值为多少？", a + b, 1))
    for _ in range(20):
        a, b = rng.randint(30, 99), rng.randint(2, 9)
        out.append(item(0, f"{a} 乘以 {b} 等于多少？", a * b, 1))
    for _ in range(20):
        b = rng.randint(3, 15)
        q = rng.randint(4, 30)
        a = b * q
        out.append(item(0, f"{a} 除以 {b} 的商是多少？", q, 1))
    return out


def t2_two_step() -> List[dict]:
    """二档：两步运算，需要保持中间结果。"""
    out = []
    for _ in range(20):
        a, b, c = rng.randint(10, 60), rng.randint(2, 9), rng.randint(3, 20)
        out.append(item(0, f"先算 {a} 乘以 {b}，再加上 {c}，结果是多少？", a * b + c, 2))
    for _ in range(10):
        price, qty, discount = rng.randint(20, 90), rng.randint(3, 12), rng.choice([5, 10, 15])
        total = price * qty
        answer = round(total * (100 - discount) / 100, 2)
        out.append(
            item(
                0,
                f"一件商品单价 {price} 元，买 {qty} 件，商家给 {discount}% 的折扣（按总价打折），"
                f"实际需要支付多少元？",
                answer,
                2,
            )
        )
    for _ in range(10):
        a, b = rng.randint(40, 200), rng.randint(5, 40)
        out.append(
            item(0, f"一个长方形的长是 {a} 厘米，宽是 {b} 厘米，它的周长是多少厘米？", 2 * (a + b), 2)
        )
    return out


def t3_three_step() -> List[dict]:
    """三档：三步以上，含百分数与单位换算。"""
    out = []
    for _ in range(15):
        base, up, down = rng.randint(100, 900), rng.choice([10, 20, 25]), rng.choice([10, 15, 20])
        final = round(base * (1 + up / 100) * (1 - down / 100), 4)
        out.append(
            item(
                0,
                f"某商品原价 {base} 元，先涨价 {up}%，之后再降价 {down}%（按涨价后的价格计算），"
                f"最终售价是多少元？保留两位小数。",
                round(final, 2),
                3,
            )
        )
    for _ in range(15):
        speed, hours, extra = rng.randint(40, 90), rng.randint(2, 5), rng.randint(10, 40)
        distance = speed * hours + extra
        out.append(
            item(
                0,
                f"一辆汽车以每小时 {speed} 公里的速度行驶 {hours} 小时，之后又行驶了 {extra} 公里，"
                f"总共行驶多少公里？",
                distance,
                3,
            )
        )
    return out


def t4_multi_hop() -> List[dict]:
    """四档：多步推理，含并列条件和「先…再…」结构，正是路由该升级到云端的区域。"""
    out = []
    for _ in range(10):
        workers, days, extra_workers = rng.randint(4, 9), rng.randint(6, 15), rng.randint(2, 6)
        total = workers * days
        new_days = round(total / (workers + extra_workers), 2)
        out.append(
            item(
                0,
                f"一项工程由 {workers} 名工人做，需要 {days} 天完成。如果增加 {extra_workers} 名工人，"
                f"且所有人效率相同，需要多少天完成？保留两位小数。",
                new_days,
                4,
            )
        )
    for _ in range(10):
        a, b = rng.randint(3, 9), rng.randint(3, 9)
        total = a * b
        out.append(
            item(
                0,
                f"有一个两位数，十位数字是 {a}，个位数字是 {b}。先把这个数乘以 2，"
                f"再减去它的各位数字之和，最终结果是多少？",
                total * 2 - (a + b),
                4,
            )
        )
    return out


KNOWLEDGE = [
    ("中国的首都是哪座城市？", "北京", ["北京"]),
    ("世界上最高的山峰叫什么名字？", "珠穆朗玛峰", ["珠峰", "珠穆朗玛"]),
    ("水的标准沸点（一个大气压下）是多少摄氏度？", "100", ["100"]),
    ("一年中哪个月份的天数最少（不考虑闰年）？", "二月", ["2月", "二月"]),
    ("太阳系中距离太阳最近的行星是哪一颗？", "水星", ["水星"]),
    ("光在真空中的传播速度约为每秒多少公里？", "30万", ["300000", "30万"]),
    ("中国农历中一年通常有多少个月？", "12", ["十二"]),
    ("人体最大的器官是什么？", "皮肤", ["皮肤"]),
    ("圆周率 π 保留两位小数是多少？", "3.14", ["3.14"]),
    ("中国的陆地面积约为多少万平方公里？", "960", ["960", "九百六十"]),
]


def knowledge_items(start_idx: int) -> List[dict]:
    out = []
    for idx, (question, answer, aliases) in enumerate(KNOWLEDGE):
        record = item(start_idx + idx, question, answer, 2, aliases)
        record["type"] = "short"
        out.append(record)
    return out


def main() -> None:
    records: List[dict] = []
    records += t1_single_step(0)
    records += t2_two_step()
    records += t3_three_step()
    records += t4_multi_hop()
    records += knowledge_items(len(records))

    for idx, record in enumerate(records, 1):
        record["id"] = f"q{idx:03d}"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    tiers = {t: sum(1 for r in records if r["tier"] == t) for t in (1, 2, 3, 4)}
    print(f"写入 {len(records)} 条 -> {OUT}")
    print(f"难度分布：{tiers}")


if __name__ == "__main__":
    main()
