FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY data/eval_set.jsonl ./data/eval_set.jsonl

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')" || exit 1

# 单进程即可：本地推理是 IO/CPU 密集但请求量低，多 worker 会各自加载一份模型
CMD ["python", "-m", "uvicorn", "app.web.main:app", "--host", "0.0.0.0", "--port", "8000"]
