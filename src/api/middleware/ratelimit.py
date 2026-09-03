"""
ratelimit.py — API 限流中间件
=============================
每用户每分钟 N 次请求，超出返回 429。

Phase 3: 内存计数 (开发)
Phase 4: Redis 滑动窗口 (生产, 多实例共享)

用法:
    app.add_middleware(RateLimitMiddleware)
"""

import time
import asyncio
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from src.core.config import get_settings
from src.core.logging import get_logger

logger = get_logger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """每用户每分钟限流中间件。

    计数存储在内存 dict，重启清零。
    生产环境替换为 Redis 滑动窗口。

    示例:
        app.add_middleware(RateLimitMiddleware, max_requests=30, window_seconds=60)
    """

    def __init__(self, app, max_requests: int | None = None, window_seconds: int = 60):
        super().__init__(app)
        settings = get_settings()
        self.max_requests = max_requests or settings.rate_limit_per_minute
        self.window_seconds = window_seconds
        self._counters: dict[str, list[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def dispatch(self, request: Request, call_next):
        # 跳过健康检查和静态文件
        if request.url.path in ("/api/v1/health", "/api/v1/health/readiness", "/api/v1/health/liveness", "/"):
            return await call_next(request)

        # 获取用户标识 (优先从 Header, 降级用 IP)
        user_id = request.headers.get("X-User-Id", "")
        if not user_id:
            user_id = request.client.host if request.client else "unknown"

        # 滑动窗口计数
        async with self._lock:
            now = time.time()
            window_start = now - self.window_seconds
            # 清理过期记录
            self._counters[user_id] = [
                t for t in self._counters[user_id] if t > window_start
            ]
            count = len(self._counters[user_id])

            if count >= self.max_requests:
                logger.warning(f"限流触发: user={user_id}, count={count}/{self.max_requests}")
                return JSONResponse(
                    status_code=429,
                    content={
                        "code": 42900,
                        "message": f"请求过于频繁，请稍后再试 (每{self.window_seconds}秒最多{self.max_requests}次)",
                        "detail": {"user_id": user_id, "current_count": count},
                    },
                )

            self._counters[user_id].append(now)

        return await call_next(request)
