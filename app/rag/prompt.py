"""受约束的 prompt：让模型「只根据片段回答、并标出处」。

两个细节是被实测逼出来的：

1. **片段编号必须写进正文格式**（``[S1]``），不能让模型自由发挥成「根据第一段」——
   自由发挥的引用无法解析，锚点层就得全靠兜底对齐。
2. **先说"片段不足就直说未提及"**，再说"简短"。顺序反过来的话，模型会为了简短
   硬凑一个答案，而不是承认检索失败。拒答率本身是评测指标之一。
"""

from __future__ import annotations

from typing import Sequence

from .parser import Chunk

SYSTEM = (
    "你是论文精读助手。只能依据给定的原文片段作答，不要使用片段以外的知识。"
    "用中文回答，不超过 80 字。每个结论后面用 [S编号] 标注出处。"
    "如果片段不足以回答问题，只回答「原文未提及」，不要猜测。"
)

_HEADER = "以下是论文原文片段（英文原文，编号 S1..Sn）：\n"


def build_user_prompt(question: str, chunks: Sequence[Chunk]) -> str:
    parts = [_HEADER]
    for c in chunks:
        sec = f" {c.section}" if c.section else ""
        parts.append(f"[{c.id}] (p.{c.page}{sec}) {c.text}\n")
    parts.append(f"\n问题：{question}\n回答：")
    return "".join(parts)


def build_messages(question: str, chunks: Sequence[Chunk]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": build_user_prompt(question, chunks)},
    ]
