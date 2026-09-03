"""
retry.py — 重试机制 & 熔断器
============================
角色：全系统对外部服务调用的容错层。任何可能失败的外部调用 (LLM API、MES、数据库)
      都应包装在 async_retry 中，避免单次失败导致用户看到错误。

核心组件:

1. async_retry() — 通用异步重试
   - 指数退避: 1s → 2s → 4s → 8s
   - 随机抖动: 避免惊群效应 (Thundering Herd)
   - 可选的熔断器集成

2. CircuitBreaker — 熔断器 (状态机)
   状态流转: CLOSED(正常) → OPEN(熔断) → HALF_OPEN(探测) → CLOSED
   恢复机制: 半开渐进: 10%探测→50%放量→100%恢复
   多实例: K8s 多 Pod 共享 Redis 熔断状态，避免各管各的

使用示例:
    from src.core.retry import async_retry, default_retry
    result = await async_retry(router.chat, messages=msgs, retry_config=default_retry)

注意事项:
- 不要把重试用在不幂等的操作上 (如 MES 工单创建由 MES 层自己处理幂等)
- 熔断器实例按模型名创建，不要共用
- 生产环境务必 configure_redis() 否则多实例各管各
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Awaitable

import redis.asyncio as aioredis

from src.core.config import get_settings
from src.core.logging import get_logger

logger = get_logger(__name__)


class CircuitState(str, Enum):
    CLOSED = "closed"           # 正常
    OPEN = "open"               # 熔断
    HALF_OPEN = "half_open"     # 半开探测


@dataclass
class CircuitBreaker:
    """熔断器: 支持多实例共享 Redis 状态 + 半开渐进恢复。"""

    name: str
    failure_threshold: int = 5
    cooldown_seconds: int = 30
    max_cooldown_seconds: int = 300
    _failure_count: int = 0
    _state: CircuitState = CircuitState.CLOSED
    _last_failure_time: float = 0.0
    _cooldown_multiplier: int = 1
    _half_open_probe_count: int = 0
    _half_open_success_count: int = 0
    _redis: aioredis.Redis | None = None

    def configure_redis(self, redis: aioredis.Redis) -> None:
        self._redis = redis

    async def _sync_to_redis(self) -> None:
        if not self._redis:
            return
        key = f"circuit:{self.name}"
        await self._redis.hset(key, mapping={
            "state": self._state.value,
            "failure_count": str(self._failure_count),
            "cooldown_multiplier": str(self._cooldown_multiplier),
        })

    async def _load_from_redis(self) -> None:
        if not self._redis:
            return
        key = f"circuit:{self.name}"
        data = await self._redis.hgetall(key)
        if data:
            self._state = CircuitState(data.get(b"state", b"closed").decode())
            self._failure_count = int(data.get(b"failure_count", 0))
            self._cooldown_multiplier = int(data.get(b"cooldown_multiplier", 1))

    @property
    def is_open(self) -> bool:
        if self._state == CircuitState.CLOSED:
            return False
        if self._state == CircuitState.OPEN:
            cooldown = self.cooldown_seconds * self._cooldown_multiplier
            if time.time() - self._last_failure_time > cooldown:
                self._state = CircuitState.HALF_OPEN
                self._half_open_probe_count = 0
                self._half_open_success_count = 0
                logger.info(f"熔断器 [{self.name}] 进入半开探测")
                return False
            return True
        return False  # HALF_OPEN allows probes

    def record_success(self) -> None:
        if self._state == CircuitState.HALF_OPEN:
            self._half_open_probe_count += 1
            self._half_open_success_count += 1
            if self._half_open_success_count >= 5:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._cooldown_multiplier = 1
                logger.info(f"熔断器 [{self.name}] 完全恢复")
        else:
            self._failure_count = 0

    def record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            self._cooldown_multiplier = min(self._cooldown_multiplier * 2, self.max_cooldown_seconds // self.cooldown_seconds)
            logger.warning(f"熔断器 [{self.name}] 半开探测失败，回到熔断 (冷却×{self._cooldown_multiplier})")
        elif self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            cooldown = self.cooldown_seconds * self._cooldown_multiplier
            logger.warning(f"熔断器 [{self.name}] 熔断 (连续失败{self._failure_count}次, 冷却{cooldown}s)")


class RetryConfig:
    """重试配置。"""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        exponential: bool = True,
        jitter: bool = True,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential = exponential
        self.jitter = jitter


async def async_retry(
    func: Callable[..., Awaitable],
    *args,
    retry_config: RetryConfig | None = None,
    circuit_breaker: CircuitBreaker | None = None,
    **kwargs,
):
    """异步重试执行，支持熔断器。"""
    config = retry_config or RetryConfig()
    last_exception = None

    for attempt in range(config.max_retries + 1):
        if circuit_breaker and circuit_breaker.is_open:
            logger.warning(f"熔断器 [{circuit_breaker.name}] 开路，拒绝请求")
            from src.core.exceptions import ModelCircuitOpenError
            raise ModelCircuitOpenError()

        try:
            result = await func(*args, **kwargs)
            if circuit_breaker:
                circuit_breaker.record_success()
            return result
        except Exception as e:
            last_exception = e
            if circuit_breaker:
                circuit_breaker.record_failure()

            if attempt == config.max_retries:
                logger.error(f"重试耗尽 ({attempt+1}/{config.max_retries+1}): {e}")
                raise

            delay = config.base_delay
            if config.exponential:
                delay = min(config.base_delay * (2 ** attempt), config.max_delay)
            if config.jitter:
                import random
                delay *= 0.5 + random.random()

            logger.warning(f"重试 {attempt+1}/{config.max_retries}, {delay:.1f}s 后重试: {e}")
            await asyncio.sleep(delay)


# 预配置的 retry 实例
default_retry = RetryConfig(max_retries=3, base_delay=1.0, max_delay=30.0)
fast_retry = RetryConfig(max_retries=2, base_delay=0.5, max_delay=5.0)
