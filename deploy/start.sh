#!/usr/bin/env bash
# 单容器内启动两个进程：本地推理服务（8080）+ 应用（8000）。
# 平台只给一个入口，所以必须由一个脚本管住两个进程的生死。
set -euo pipefail

MODEL_PATH="${LOCAL_MODEL_PATH:-/app/models/model.gguf}"
MODEL_NAME="${LOCAL_SERVED_MODEL:-qwen2.5-1.5b}"
N_THREADS="${N_THREADS:-2}"

echo "[start] 启动本地推理服务：${MODEL_NAME}"
python -m llama_cpp.server \
    --model "$MODEL_PATH" \
    --model_alias "$MODEL_NAME" \
    --host 127.0.0.1 --port 8080 \
    --n_ctx 4096 \
    --n_threads "$N_THREADS" \
    > /tmp/llama.log 2>&1 &

LLAMA_PID=$!

echo "[start] 等待推理服务就绪（最多 180 秒）…"
for i in $(seq 1 90); do
    if curl -fsS http://127.0.0.1:8080/v1/models > /dev/null 2>&1; then
        echo "[start] 模型已加载"
        break
    fi
    if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
        echo "[start] 推理服务启动失败，日志如下："
        tail -50 /tmp/llama.log
        exit 1
    fi
    sleep 2
done

echo "[start] 启动应用"
exec python -m uvicorn app.web.main:app --host 0.0.0.0 --port "${PORT:-8000}"
