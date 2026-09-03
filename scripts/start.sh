#!/bin/bash
# 一键启动 (开发模式)
# 后端
echo "Starting backend..."
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# 等后端就绪
echo "Waiting for backend..."
for i in $(seq 1 60); do
    curl -s http://localhost:8000/api/v1/health > /dev/null 2>&1 && break
    sleep 3
done

# 前端
echo "Starting frontend..."
cd frontend/chat && npm run dev &
FRONTEND_PID=$!

echo ""
echo "========================================"
echo "  后端: http://localhost:8000"
echo "  前端: http://localhost:5173"
echo "  API文档: http://localhost:8000/docs"
echo "  Metrics: http://localhost:8000/metrics"
echo "  管理后台: cd frontend/admin && npm run dev"
echo "========================================"

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT
wait
