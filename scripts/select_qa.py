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

PER_PAPER = 40          # 每篇论文取多少题（下面 DROP 再去掉一批）
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
    # 中文作答把英文金标翻成了中文/换了写法，加别名避免假阴性（金标本身不改）
    "rag_2005.11401__S33": ["nq rag"],
    "rag_2005.11401__S60": ["表 5", "表5", "table 5"],
    "rag_2005.11401__S61": ["bm25"],
    "rag_2005.11401__S75": ["机器翻译"],
    "lora_2106.09685__S60": ["gpt-2 medium"],
    "lora_2106.09685__S89": ["grassmann"],
    "lora_2106.09685__S92": ["图4", "图 4"],
    "lost_in_middle_2307.03172__S19": ["u形", "u 形", "u-shaped"],
    "lost_in_middle_2307.03172__S76": ["u形", "u 形", "u-shaped"],
    "lost_in_middle_2307.03172__S126": ["干扰项的顺序", "随机化"],
    "rag_2005.11401__S7": ["realm", "orqa"],
    "rag_2005.11401__S40": ["维基百科", "wikipedia"],
    "rag_2005.11401__S78": ["bart"],
    "lora_2106.09685__S12": ["prefix", "前缀"],
}

_NUM = re.compile(r"\d+(?:[.,]\d+)*%?")
_BAD = re.compile(r"[=|\\]|^https?://|et al\.|\(19\d\d\)|\(20\d\d\)")

# 质量问题过滤（逐条看过新题之后总结出来的四类，写进脚本才能复现）
_BAD_QUESTION = re.compile(
    r"arXiv|论文编号|发表在哪个会议|期刊|诺贝尔|海明威|分别是|分别多少|各是多少|第几节"
    r"|作者来自|作者中|作者是谁|工作的作者|论文标题|哪所大学|哪个机构|感谢|图\s?\d|表\s?\d|论文主要聚焦"
    r"|多少.{0,14}多少"          # 一句话问两个数，取最后一个数的判分会误伤
)

# 泛化的多词答案：中文作答会换说法，别名匹配不可靠
_GENERIC_ANSWERS = {"encoder-decoder models", "transformer language models", "rag-s"}
_LOWERCASE_WORD = re.compile(r"[a-z][a-z\- ]*")


def bad_fact_answer(ans: str) -> bool:
    """全小写的英文普通词：中文作答会翻译（retriever→检索器），别名匹配必失败。"""
    return bool(_LOWERCASE_WORD.fullmatch(ans)) or ans.startswith("§")


def proper_noun_ok(ans: str) -> bool:
    """多词答案只放行「专有名词/缩写」：中文作答时模型通常保留英文原文。

    例如 Dense Passage Retriever、Wq and Wv、Kronecker products；
    而 a natural language query 这类小写短语一律不要。
    """
    words = ans.split()
    if not 1 < len(words) <= 4 or len(ans) > 30:
        return False
    if not all(re.fullmatch(r"[A-Za-z0-9\-\.\+/=]+", w) for w in words):
        return False
    return ans[0].isupper() or ans.isupper()


def bad_number_answer(ans: str, chunk_text: str) -> bool:
    """数字是更长标识符的一部分，例如 gold=100 其实来自 V100。"""
    return re.search(r"[A-Za-z0-9]" + re.escape(ans), chunk_text) is not None


def load_chunks(paper: str) -> dict:
    """读段落文件；缺失时给出可执行的修复步骤，而不是抛 FileNotFoundError。"""
    src = ROOT / "data" / "papers" / f"{paper}.chunks.json"
    if not src.exists():
        raise SystemExit(
            f"缺少 {src.relative_to(ROOT)}：请先运行\n"
            "  python scripts/fetch_papers.py\n"
            "  python scripts/build_paper_index.py\n"
            "生成段落文件后再重建评测集。"
        )
    data = json.loads(src.read_text(encoding="utf-8"))
    return {c["id"]: c for c in data["chunks"]}


def clean_number(ans: str):
    """把 '10,000 times' 这类答案收敛成一个可判分的最小数字串。

    整数要去掉多余的 `.0`：原文写 125.0M、模型答 125M，否则会被判错。
    """
    m = _NUM.search(ans)
    if not m:
        return None
    token = m.group(0).replace(",", "").rstrip("%")
    if len(token) < 3:
        return None
    try:
        value = float(token)
    except ValueError:
        return None
    if value.is_integer():
        token = str(int(value))
    return token


def main() -> int:
    # 两轮出题的候选池合并：qa_candidates.jsonl（第一轮）+ qa_candidates_cross.jsonl（跨句综合）
    pool_files = [CANDIDATES] + sorted(CANDIDATES.parent.glob("qa_candidates_*.jsonl"))
    rows: list[dict] = []
    seen: set = set()
    for path in pool_files:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            key = (item["chunk_id"], item["answer"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(item)
    if not rows:
        raise SystemExit(
            "候选池为空：请先运行 scripts/gen_qa_draft.py 生成 qa_draft*.jsonl 再用 --verify 校验，\n"
            "或确认 data/qa_candidates*.jsonl 存在（本仓库已附带）"
        )
    print(f"候选池：{len(rows)} 条（来自 {[p.name for p in pool_files if p.exists()]}）")
    random.seed(20261111)

    by_paper = defaultdict(list)
    for r in rows:
        by_paper[r["paper"]].append(r)

    picked: list[dict] = []
    for paper in sorted(by_paper):
        chunk_text = {cid: c["text"] for cid, c in load_chunks(paper).items()}

        numeric: list[tuple[dict, str]] = []
        facts: list[dict] = []
        for r in by_paper[paper]:
            if _BAD.search(r["answer"]) or _BAD_QUESTION.search(r["question"]):
                continue
            num = clean_number(r["answer"])
            if num is not None:
                if bad_number_answer(num, chunk_text.get(r["chunk_id"], "")):
                    continue
                numeric.append((r, num))
            elif " " not in r["answer"] and 2 <= len(r["answer"]) <= 30:
                if bad_fact_answer(r["answer"]):
                    continue
                facts.append(r)
            elif proper_noun_ok(r["answer"]):
                # 多词专有名词：答案仍然是逐字出现在原文里的，别名给大小写两种写法
                if r["answer"].lower() in _GENERIC_ANSWERS:
                    continue
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

    # 人工补充：多词专有名词题。机器筛选只收「答案无空格」的项，这批是逐条回原文看过之后补的；
    # 答案同样要求逐字出现在源段落里（下面有断言，不满足直接报错）。
    manual = [
        ("rag_2005.11401", "S11", "RAG 中作为非参数记忆的稠密向量索引，用什么检索器访问？",
         "Dense Passage Retriever", ["dpr", "稠密段落检索"]),
        ("rag_2005.11401", "S29", "为避免额外前向传播、更高效解码的近似方法被称为什么？",
         "Fast Decoding", ["快速解码"]),
        ("rag_2005.11401", "S36", "在非问答设定下评估生成能力时，论文让模型生成哪种格式的问题？",
         "Jeopardy questions", ["jeopardy", "危险边缘"]),
        ("rag_2005.11401", "S52", "BART 把《The Sun Also Rises》的作者关联到哪部作品？",
         "A Farewell to Arms", ["farewell to arms", "永别了武器"]),
        ("lora_2106.09685", "S52", "LoRA 在大多数实验中只应用于哪两个权重矩阵？",
         "Wq and Wv", ["wq", "wv"]),
        ("lora_2106.09685", "S73", "COMPACTER 用什么方式参数化适配器？",
         "Kronecker products", ["kronecker", "克罗内克"]),
        ("lora_2106.09685", "S35", "作者在下游任务中冻结了哪些模块，只适配注意力权重？",
         "MLP modules", ["mlp", "多层感知机"]),
        ("lora_2106.09685", "S23", "对模型做 sharding 时，adapter 增加的深度会要求更多哪类同步？",
         "AllReduce and Broadcast", ["allreduce", "broadcast"]),
        ("lost_in_middle_2307.03172", "S17", "一些模型在输入上下文中部检索匹配 token 时表现出什么形状的性能曲线？",
         "U-shaped performance curve", ["u-shaped", "u形"]),
        ("lost_in_middle_2307.03172", "S126", "作者用什么方式验证性能下降不只是因为无法识别相关文档？",
         "Randomizing Distractor Order", ["randomizing", "打乱干扰项顺序"]),
        ("lora_2106.09685", "S41", "所有实验使用的是哪种 GPU？",
         "NVIDIA Tesla V100", ["v100", "特斯拉 v100"]),
        ("rag_2005.11401", "S34", "测试 RAG 在知识密集型场景下自然语言生成能力用的是哪个任务？",
         "MSMARCO NLG task v2.1", ["msmarco", "v2.1"]),
    ]
    chunk_cache: dict = {}
    for paper, cid, question, answer, aliases in manual:
        if paper not in chunk_cache:
            chunk_cache[paper] = load_chunks(paper)
        chunk = chunk_cache[paper][cid]
        if answer.lower() not in chunk["text"].lower():
            raise SystemExit(f"人工题答案不在原文：{paper}/{cid}/{answer}")
        item_id = f"{paper}__{cid}"
        picked = [p for p in picked if p["id"] != item_id]  # 与机器选题去重
        picked.append(
            {
                "id": item_id,
                "paper": paper,
                "type": "fact",
                "question": question,
                "answer": answer,
                "aliases": [answer.lower()] + aliases,
                "gold_pages": [int(chunk["page"])],
                "source_chunk": cid,
                "answerable": True,
            }
        )

    # 「原文没写」的题，用于测拒答：答案明确不在论文里
    unanswerable = [
        ("lora_2106.09685", "LoRA 的作者有没有获得过图灵奖？", 26),
        ("lora_2106.09685", "这篇论文在 ImageNet 图像分类上的 Top-1 准确率是多少？", 26),
        ("lora_2106.09685", "这篇论文有没有给出在手机端部署的显存占用数据？", 26),
        ("lost_in_middle_2307.03172", "这篇论文提出的模型在 GLUE 基准上的平均分是多少？", 18),
        ("lost_in_middle_2307.03172", "作者有没有公开训练代码的 GitHub 仓库地址？", 18),
        ("lost_in_middle_2307.03172", "这篇论文用了多少张 A100 显卡做训练？", 18),
        ("rag_2005.11401", "RAG 在 ImageNet 上的 Top-5 准确率是多少？", 19),
        ("rag_2005.11401", "这篇论文的作者来自哪家公司的人力资源部门？", 19),
        ("rag_2005.11401", "论文有没有报告推理成本相比 BART 高出多少倍？", 19),
        ("lora_2106.09685", "论文有没有报告在中文语料上的实验结果？", 26),
        ("lost_in_middle_2307.03172", "作者有没有报告模型在语音任务上的表现？", 18),
        ("rag_2005.11401", "论文有没有给出与 GPT-4 的对比实验？", 19),
        ("rag_2005.11401", "论文有没有报告在中文数据集上的实验结果？", 19),
        ("lora_2106.09685", "这篇论文有没有给出量化推理的显存占用数据？", 26),
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
