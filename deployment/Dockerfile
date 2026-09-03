# 制造业Agent — 后端 Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 应用代码
COPY src/ ./src/
COPY data/ ./data/
COPY evaluation/ ./evaluation/

# 创建必要目录
RUN mkdir -p logs data/uploads data/finetune models

# 模型缓存持久化
VOLUME ["/root/.cache"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
