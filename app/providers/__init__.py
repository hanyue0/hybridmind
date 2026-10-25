from .base import Generation, Provider, ProviderError
from .llamacpp_provider import LlamaCppProvider
from .ollama_provider import OllamaProvider
from .openai_provider import OpenAICompatProvider

__all__ = [
    "Generation",
    "Provider",
    "ProviderError",
    "OllamaProvider",
    "LlamaCppProvider",
    "OpenAICompatProvider",
    "build_local_provider",
]


def build_local_provider(settings):
    """按配置选择本地后端。

    开发用 Ollama（换模型方便），部署用 llama.cpp（单容器、无守护进程）。
    上层拿到的都是同一个 Provider 协议，编排与评测代码完全不用改。
    """
    if getattr(settings, "local_backend", "ollama") == "openai":
        # 部署形态：本地模型由 llama-server 提供 OpenAI 兼容服务，
        # 复用云端 Provider 的协议，不需要为本地单独写一套客户端。
        return OpenAICompatProvider(
            settings.local_base_url,
            "local",
            settings.local_served_model,
            settings.price_of("local"),
            timeout_s=max(settings.timeout_s, 120),
        )
    if getattr(settings, "local_backend", "ollama") == "llamacpp":
        return LlamaCppProvider(
            settings.local_model_path,
            n_ctx=int(getattr(settings, "local_n_ctx", 4096)),
        )
    return OllamaProvider(
        settings.ollama_host, settings.local_model, timeout_s=settings.timeout_s
    )
