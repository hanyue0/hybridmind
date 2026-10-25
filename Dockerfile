# 生产镜像：单容器运行「应用 + 本地小模型」。
#
# 与开发形态（docker-compose 里的 Ollama）的区别：
# - 不依赖外部守护进程，托管平台给一个容器就能跑；
# - 本地推理用 llama.cpp 的 OpenAI 兼容服务，应用侧复用同一套 Provider；
# - 模型在构建期从 hf-mirror 下载（国内可达），运行期离线。
#
# 构建：docker build -t hybridmind .
# 运行：docker run -p 8000:8000 -e CLOUD_API_KEY=sk-xxx hybridmind

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    LOCAL_BACKEND=openai \
    LOCAL_BASE_URL=http://127.0.0.1:8080/v1 \
    LOCAL_SERVED_MODEL=qwen2.5-1.5b \
    ROUTER_MODE=learned

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# 本地推理服务：llama-cpp-python 自带 OpenAI 兼容的 server 子模块。
# 优先取预编译轮子（快，不用编译器）；取不到时才装工具链现编。
RUN pip install "llama-cpp-python[server]" \
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    || (apt-get update \
        && apt-get install -y --no-install-recommends build-essential cmake \
        && pip install "llama-cpp-python[server]" \
        && apt-get purge -y build-essential cmake \
        && rm -rf /var/lib/apt/lists/*)

# 模型权重：构建期下载，约 1 GB
RUN mkdir -p /app/models && \
    curl -fL --retry 3 --retry-delay 2 -o /app/models/model.gguf \
    https://hf-mirror.com/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf

COPY app ./app
COPY data/eval_set.jsonl ./data/eval_set.jsonl
COPY data/router_model.json ./data/router_model.json
COPY deploy/start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=6s --start-period=120s \
    CMD curl -fsS http://127.0.0.1:7860/api/health || exit 1

CMD ["/start.sh"]
