"""
concurrency.py — LLM 并发管控 & 排队
=====================================
防止多用户并发请求同时调 LLM API，触发上游限流。

工作机制:
  1. 全局信号量控制最大并发 LLM 调用数 (默认 10)
  2. 超出并发 → 排队等待，最长等 60s
  3. 排队超时 → 返回降级信息

用法:
    from src.model.concurrency import LLMConcurrencyLimiter
    limiter = LLMConcurrencyLimiter(max_concurrent=10)
    async with limiter.acquire(user_id="worker_zhang"):
        result = await call_llm(...)
"""

import asyncio
import time
from collections import defaultdict

from src.core.logging import get_logger

logger = get_logger(__name__)


class LLMConcurrencyLimiter:
    """LLM 调用并发限制器。

    全局信号量控制最大并发，超出排队等待。
    支持按用户统计排队时间，超时降级。

    示例:
        limiter = LLMConcurrencyLimiter(max_concurrent=10)
        async with limiter.acquire(user_id="worker_zhang"):
            result = await router.chat(messages=...)
    """

    def __init__(self, max_concurrent: int = 10):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._max_concurrent = max_concurrent
        self._wait_times: dict[str, list[float]] = defaultdict(list)  # 用户排队时间统计
        self._total_requests = 0
        self._total_waited = 0  # 累计排队次数
        logger.info(f"LLM 并发限制: max={max_concurrent}")

    def acquire(self, user_id: str = "", timeout: float = 60.0):
        """获取 LLM 调用槽位。

        返回 async context manager。

        排队超时返回 None → 调用方应降级。

        示例:
            async with limiter.acquire("worker_zhang", timeout=30) as slot:
                if slot is None:
                    return "系统繁忙，请稍后重试"
                result = await call_llm()
        """
        return _SlotContext(self, user_id, timeout)

    @property
    def available(self) -> int:
        """可用槽位数。"""
        return self._semaphore._value

    @property
    def waiting_count(self) -> int:
        """排队中的请求数。"""
        return self._total_requests - (self._max_concurrent - self.available)


class _SlotContext:
    def __init__(self, limiter: LLMConcurrencyLimiter, user_id: str, timeout: float):
        self._limiter = limiter
        self._user_id = user_id
        self._timeout = timeout
        self._slot = False

    async def __aenter__(self):
        try:
            t0 = time.time()
            acquired = await asyncio.wait_for(
                self._limiter._semaphore.acquire(),
                timeout=self._timeout,
            )
            wait_time = (time.time() - t0) * 1000
            if acquired and wait_time > 100:
                logger.info(f"LLM 排队: user={self._user_id}, wait={wait_time:.0f}ms")
                self._limiter._total_waited += 1
            self._slot = acquired
            return self if acquired else None
        except asyncio.TimeoutError:
            logger.warning(f"LLM 排队超时: user={self._user_id}, timeout={self._timeout}s")
            self._slot = False
            return None

    async def __aexit__(self, *args):
        if self._slot:
            self._limiter._semaphore.release()

    def __bool__(self):
        return self._slot


# 全局单例
llm_limiter = LLMConcurrencyLimiter(max_concurrent=10)
