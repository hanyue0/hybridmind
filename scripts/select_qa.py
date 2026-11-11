"""从候选草案里挑出正式评测集 ``data/qa_paper.jsonl``。

只做「筛选与去重」，不改写答案——答案的合法性已经由 gen_qa_draft 的
逐字校验保证过（答案必须在源段落里原样出现）。挑选规则：

1. 每题必须有唯一的 gold page（来自答案所在段落）；
2. 同一段落最多出一题，同一数值/术语最多出现一次，避免重复计分；
3. 数值题要求数字足够独特（≥3 字符或带小数/百分号），避免「3」「10」这种
   在任意长回答里都能蒙对的答案；
4. 事实题的答案必须是单一专有名词（不含空格或引文格式），便于确定性匹配。
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CANDIDATES = ROOT / "data" / "qa_candidates.jsonl"
OUT = ROOT / "data" / "qa_paper.jsonl"

PER_PAPER = 26          # 每篇论文取多少题（下面 DROP 再去掉一批）
NUM_RATIO = 0.6         # 其中数值题占比

# 人工复核后剔除的题。保留在这里而不是直接删数据，是为了让「为什么剔」
# 有据可查——每一条都是我在逐题看的时候发现的具体问题。
DROP = {
    "rag_2005.11401__S54": "论文里举的检索示例（海明威），不是论文自身内容",
    "lost_in_middle_2307.03172__S47": "答案 3.5 来自 GPT-3.5 型号名，语义错误",
    "lora_2106.09685__S71": "答案 175 缺单位，与 175 billion 混淆",
    "lora_2106.09685__S77": "问的是文献，抽出的却是年份 2020",
    "lost_in_middle_2307.03172__S49": "JSON 键值示例，不是论文结论",
    "lora_2106.09685__S34": "答案是英文数词 four，中文作答无法别名匹配",
    "rag_2005.11401__S34": "答案 2.1 是任务版本号，单独作答案无意义",
    "lost_in_middle_2307.03172__S10": "答案应为 Claude-1.3，抽出 1.3 不完整",
    "lost_in_middle_2307.03172__S90": "coarse 是英文副词，中文作答匹配不上",
    "rag_2005.11401__S32": "答案 Natural 被截断（应为四个数据集）",
    "lost_in_middle_2307.03172__S8": "问的是区间 512–2048，取单一数值会误判",
    "lora_2106.09685__S24": "non-monotonically 是英文副词，匹配不上",
    "lost_in_middle_2307.03172__S6": "答案是 middle，中文作答会写「中间」",
    "lost_in_middle_2307.03172__S70": "near-perfect 是形容性短语，判分不稳",
    "lost_in_middle_2307.03172__S138": "tokenizer 常被译成「分词器」",
    "lost_in_middle_2307.03172__S52": "UUIDs 常被译成「UUID」单数",
}

# 允许补充别名的地方：答案本身合法，但中文作答可能换说法
ALIAS_OVERRIDES = {
    "rag_2005.11401__S40": ["维基百科", "wikipedia"],
    "rag_2005.11401__S78": ["bart"],
    "lora_2106.09685__S12": ["prefix", "前缀"],
}

_NUM = re.compile(r"\d+(?:[.,]\d+)*%?")
_BAD = re.compile(r"[=|\\]|^https?://|et al\.|\(19\d\d\)|\(20\d\d\)")


def clean_number(ans: str):
    """把 '10,000 times' 这类答案收敛成一个可判分的最小数字串。"""
    m = _NUM.search(ans)
    if not m:
        return None
    token = m.group(0).replace(",", "").rstrip("%")
    if len(token) < 3:
        return None
    try:
        float(token)
    except ValueError:
        return None
    return token


def main() -> int:
    rows = [
        json.loads(line)
        for line in CANDIDATES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    random.seed(20261111)

    by_paper = defaultdict(list)
    for r in rows:
        by_paper[r["paper"]].append(r)

    picked: list[dict] = []
    for paper in sorted(by_paper):
        numeric: list[tuple[dict, str]] = []
        facts: list[dict] = []
        for r in by_paper[paper]:
            if _BAD.search(r["answer"]):
                continue
            num = clean_number(r["answer"])
            if num is not None:
                numeric.append((r, num))
            elif " " not in r["answer"] and 2 <= len(r["answer"]) <= 30:
                facts.append(r)

        random.shuffle(numeric)
        random.shuffle(facts)

        seen_ans, seen_chunk = set(), set()
        chosen: list[dict] = []

        def add(item: dict, key: str, chunk_id: str) -> bool:
            if key in seen_ans or chunk_id in seen_chunk:
                return False
            seen_ans.add(key)
            seen_chunk.add(chunk_id)
            chosen.append(item)
            return True

        want_num = int(PER_PAPER * NUM_RATIO)
        for r, num in numeric:
            if len(chosen) >= want_num:
                break
            add(
                {
                    "id": f"{paper}__{r['chunk_id']}",
                    "paper": paper,
                    "type": "number",
                    "question": r["question"],
                    "answer": num,
                    "aliases": [r["answer"]],
                    "gold_pages": [int(r["page"])],
                    "source_chunk": r["chunk_id"],
                    "answerable": True,
                },
                key=num,
                chunk_id=r["chunk_id"],
            )

        for r in facts:
            if len(chosen) >= PER_PAPER:
                break
            add(
                {
                    "id": f"{paper}__{r['chunk_id']}",
                    "paper": paper,
                    "type": "fact",
                    "question": r["question"],
                    "answer": r["answer"],
                    "aliases": [r["answer"].lower()],
                    "gold_pages": [int(r["page"])],
                    "source_chunk": r["chunk_id"],
                    "answerable": True,
                },
                key=r["answer"].lower(),
                chunk_id=r["chunk_id"],
            )

        picked.extend(chosen)
        print(
            f"{paper}: 候选 {len(by_paper[paper])} → 选中 {len(chosen)} "
            f"（数值 {sum(1 for c in chosen if c['type'] == 'number')}）"
        )

    # 六道「原文没写」的题，用于测拒答：答案明确不在论文里
    unanswerable = [
        ("lora_2106.09685", "LoRA 的作者有没有获得过图灵奖？", 26),
        ("lora_2106.09685", "这篇论文在 ImageNet 图像分类上的 Top-1 准确率是多少？", 26),
        ("lost_in_middle_2307.03172", "这篇论文提出的模型在 GLUE 基准上的平均分是多少？", 18),
        ("lost_in_middle_2307.03172", "作者有没有公开训练代码的 GitHub 仓库地址？", 18),
        ("rag_2005.11401", "RAG 在 ImageNet 上的 Top-5 准确率是多少？", 19),
        ("rag_2005.11401", "这篇论文的作者来自哪家公司的人力资源部门？", 19),
    ]
    for i, (paper, q, pages) in enumerate(unanswerable, 1):
        picked.append(
            {
                "id": f"{paper}__unans{i}",
                "paper": paper,
                "type": "unanswerable",
                "question": q,
                "answer": "原文未提及",
                "aliases": ["未提及", "没有提到", "无法回答"],
                "gold_pages": [pages],
                "source_chunk": None,
                "answerable": False,
            }
        )

    random.shuffle(picked)
    dropped = [p for p in picked if p["id"] in DROP]
    picked = [p for p in picked if p["id"] not in DROP]
    for item in picked:
        if item["id"] in ALIAS_OVERRIDES:
            item["aliases"] = list(dict.fromkeys(item["aliases"] + ALIAS_OVERRIDES[item["id"]]))
    print(f"复核剔除 {len(dropped)} 题：{', '.join(sorted(DROP))}")

    OUT.write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in picked), encoding="utf-8"
    )
    print(f"\n写入 {OUT}：{len(picked)} 题")
    print("题型分布:", dict(Counter(p["type"] for p in picked)))
    print("论文分布:", dict(Counter(p["paper"] for p in picked)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
