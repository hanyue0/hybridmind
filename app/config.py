"""集中配置：所有可调参数都收在这里，便于写进技术文档与做消融实验。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """极简 .env 解析：不引入额外依赖，也避免 python-dotenv 的覆盖行为。"""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Price:
    """计价单位：人民币元 / 百万 token。"""

    input: float
    output: float


# 云端价格表（元/百万 token）。数值以 2026-10 官方公示价存档，评审若不认可可在此单点替换。
DEFAULT_PRICES: Dict[str, Price] = {
    "deepseek-chat": Price(input=2.0, output=8.0),
    "qwen-plus": Price(input=0.8, output=2.0),
    "glm-4-flash": Price(input=0.0, output=0.0),
}


@dataclass
class Settings:
    # 推理端
    local_model: str = os.getenv("LOCAL_MODEL", "qwen2.5:1.5b")
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    cloud_model: str = os.getenv("CLOUD_MODEL", "deepseek-chat")
    cloud_base_url: str = os.getenv("CLOUD_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
    cloud_api_key: str = os.getenv("CLOUD_API_KEY", "")
    prices: Dict[str, Price] = field(default_factory=lambda: dict(DEFAULT_PRICES))

    # 路由旋钮
    theta_local: float = float(os.getenv("THETA_LOCAL", "0.45"))
    tau_confidence: float = float(os.getenv("TAU_CONFIDENCE", "0.55"))
    enable_escalation: bool = os.getenv("ENABLE_ESCALATION", "1") == "1"
    budget_cny: float = float(os.getenv("BUDGET_CNY", "20.0"))

    # 生成参数
    # 默认 256：本地 CPU 推理约 4 tok/s，上限过高只会让单次请求拖到一分钟
    max_tokens: int = int(os.getenv("MAX_TOKENS", "256"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.2"))
    timeout_s: float = float(os.getenv("TIMEOUT_S", "60"))

    def price_of(self, model: str) -> Price:
        return self.prices.get(model, Price(input=0.0, output=0.0))

    @property
    def cloud_ready(self) -> bool:
        return bool(self.cloud_api_key)


settings = Settings()
