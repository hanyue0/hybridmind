"""置信度兜底：本地答完先自评，自评不达标就升级重答。

思路借自推测执行（speculative execution）——先让便宜的算，再验证，
验证不通过才付昂贵的那份钱。区别在于这里的"验证器"是模型自评而非硬件比对，
所以必须承认它不可靠：评测里会单独做一组「关闭兜底」的消融，让数据说话。
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

from ..providers.base import Provider, ProviderError

_CONFIDENCE_PROMPT = (
    "你刚才给出的回答，请评估其正确概率。\n"
    "只输出一个 0 到 1 之间的小数，不要输出任何其他文字。\n"
    "0 表示完全没把握，1 表示确定正确。"
)

_NUMBER = re.compile(r"(?:^|[^\d])(0(?:\.\d+)?|1(?:\.0+)?|\.\d+)")


def parse_confidence(text: str) -> Optional[float]:
    match = _NUMBER.search(text.strip())
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    return min(max(value, 0.0), 1.0)


def estimate(
    provider: Provider,
    messages: Sequence[dict],
    answer: str,
    max_tokens: int = 32,
    temperature: float = 0.0,
) -> Optional[float]:
    """让本地模型给自己的答案打分。失败一律返回 None，由上层决定降级策略。"""
    verify_messages = list(messages) + [
        {"role": "assistant", "content": answer},
        {"role": "user", "content": _CONFIDENCE_PROMPT},
    ]
    try:
        generation = provider.generate(
            verify_messages, max_tokens=max_tokens, temperature=temperature
        )
    except ProviderError:
        return None
    return parse_confidence(generation.text)
