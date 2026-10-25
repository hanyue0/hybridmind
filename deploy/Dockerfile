# 单容器部署镜像：应用 + 本地模型，不依赖外部 Ollama 守护进程
#
# 设计取舍：
# - 本地推理用 llama.cpp 的 server（OpenAI 兼容），而不是 Ollama 守护进程，
#   因为多数托管平台只给一个容器、不保证能常驻第二个服务；
# - 模型在构建期下载（hf-mirror 国内可达），运行时只加载，不联网；
# - 应用与模型同容器，单进程编排由 start.sh 负责。
#
# 注意：本文件在编写时未能在本机构建验证（Docker 守护进程未启动），
# 首次构建请留意 llama-cpp-python 的 wheel 与模型下载是否成功。

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    LOCAL_BACKEND=openai \
    LOCAL_BASE_URL=http://127.0.0.1:8080/v1 \
    LOCAL_SERVED_MODEL=qwen2.5-1.5b

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# llama-cpp-python 提供 llama_cpp.server，暴露 OpenAI 兼容接口。
# 用官方预编译索引，避免在容器里现编（编一次十几分钟）。
RUN pip install llama-cpp-python \
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    || pip install llama-cpp-python

# 模型：构建期下载，运行时离线可用
RUN mkdir -p /app/models && \
    curl -fL --retry 3 -o /app/models/model.gguf \
    https://hf-mirror.com/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf

COPY app ./app
COPY deploy/start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=6s --start-period=90s \
    CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

CMD ["/start.sh"]
