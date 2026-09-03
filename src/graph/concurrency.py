"""
concurrency.py — 并行 Agent 资源管控
====================================
角色：限制单次请求中并行执行的 Agent 数量，防止多 Agent 并发打满 LLM 调用和资源。

为什么需要并发管控:
- 一次复杂查询可能触发 diagnosis + knowledge + quality 3个Agent
- 3个Agent 同时调 LLM → 3倍 Token 消耗瞬间爆发
- 没有上限的话，告警风暴时可能同时跑几十个 Agent → API rate limit 被触发

管控策略:
1. 单次请求最多并行 N 个 Agent (配置项 MAX_PARALLEL_AGENTS，默认3)
2. 超出部分串行排队
3. 每个 Agent 的 Token 预算不能超过单次上限

使用方式:
    from src.graph.concurrency import AgentConcurrencyLimiter

    limiter = AgentConcurrencyLimiter(max_parallel=3)
    async with limiter:
        await some_agent_work()
    # 如果超出并发数 → 排队等待

注意事项:
- 这是请求级别的并发控制 (单个 session 内部)
- 全局 A2A 并发控制在 model/fallback.py 的 A2A 信号量中
- max_parallel_agents 通过 .env 可配，不需要改代码
"""

import asyncio

from src.core.config import get_settings
from src.core.logging import get_logger

logger = get_logger(__name__)


class AgentConcurrencyLimiter:
    """单次请求的 Agent 并发限制器。

    每个 session 有自己的 semaphore，跨 session 不互相影响。

    示例:
        limiter = AgentConcurrencyLimiter(max_parallel=3)
        async with limiter.acquire():
            await run_agent("diagnosis")
        # 如果已经有 3 个 Agent 在跑 → 等待直到有槽位
    """

    def __init__(self, max_parallel: int | None = None):
        settings = get_settings()
        self.max_parallel = max_parallel or settings.max_parallel_agents
        self._semaphore = asyncio.Semaphore(self.max_parallel)
        logger.debug(f"Agent 并发限制: max={self.max_parallel}")

    def acquire(self):
        """获取并发槽位。用 async context manager。

        示例:
            async with limiter.acquire():
                await heavy_agent_work()
        """
        return self._semaphore

    async def wait_if_busy(self, timeout: float = 5.0) -> bool:
        """等待直到有空闲槽位。

        返回: True=获取成功, False=超时
        """
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            logger.warning(f"Agent 并发槽位等待超时 ({timeout}s)")
            return False

    def release(self) -> None:
        """释放槽位。"""
        self._semaphore.release()

    @property
    def available(self) -> int:
        """当前可用槽位数。"""
        return self._semaphore._value
