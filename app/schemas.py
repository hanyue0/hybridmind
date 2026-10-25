"""接口与数据模型：Web 层、评测层共用同一套结构，保证「演示所见」= 「评测所测」。"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant"]
Mode = Literal["auto", "local", "cloud"]


class Message(BaseModel):
    role: Role
    content: str


class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., min_length=1)
    mode: Mode = "auto"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_cny: float = 0.0


class RouteTrace(BaseModel):
    """路由决策的完整凭证——前端展示、日志落盘、评测统计都读这一份。"""

    route: Literal["local", "cloud"]
    difficulty: float
    threshold: float
    reasons: List[str] = Field(default_factory=list)
    escalated: bool = False
    confidence: Optional[float] = None
    latency_ms: float = 0.0
    # 分值语义：difficulty=难度分（越高越难）；p_local_correct=本地答对概率（越低越该上云）
    score_kind: Literal["difficulty", "p_local_correct"] = "difficulty"


class ChatResponse(BaseModel):
    answer: str
    model: str
    trace: RouteTrace
    usage: Usage
