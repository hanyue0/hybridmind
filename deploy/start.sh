#!/usr/bin/env bash
# 单容器内编排两个进程：本地推理服务（8080）+ 应用（8000）。
# 托管平台只给一个入口（通常是 $PORT），所以必须由一个脚本看住两个进程。
set -euo pipefail

MODEL_PATH="${LOCAL_MODEL_PATH:-/app/models/model.gguf}"
MODEL_NAME="${LOCAL_SERVED_MODEL:-qwen2.5-1.5b}"
N_THREADS="${N_THREADS:-2}"
APP_PORT="${PORT:-7860}"   # ModelScope Docker 空间端口固定 7860

echo "[start] 本地推理服务启动：${MODEL_PATH}"
python -m llama_cpp.server \
    --model "$MODEL_PATH" \
    --model_alias "$MODEL_NAME" \
    --host 127.0.0.1 --port 8080 \
    --n_ctx 4096 \
    --n_threads "$N_THREADS" \
    > /tmp/llama.log 2>&1 &
LLAMA_PID=$!

echo "[start] 等待模型就绪（最多 240 秒）…"
READY=0
for _ in $(seq 1 120); do
    if curl -fsS http://127.0.0.1:8080/v1/models > /dev/null 2>&1; then
        READY=1
        echo "[start] 模型已加载"
        break
    fi
    if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
        echo "[start] 推理服务异常退出，日志："
        tail -60 /tmp/llama.log
        exit 1
    fi
    sleep 2
done

if [ "$READY" -ne 1 ]; then
    echo "[start] 模型加载超时，但仍继续启动应用（应用会在请求时重试）"
fi

echo "[start] 应用启动：0.0.0.0:${APP_PORT}"
exec python -m uvicorn app.web.main:app --host 0.0.0.0 --port "$APP_PORT"
