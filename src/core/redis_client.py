"""
redis_client.py — Redis 连接管理
================================
Phase 2: 为缓存/会话/Checkpoint/熔断器提供 Redis 连接。

使用方式:
    from src.core.redis_client import get_redis, is_redis_available
    redis = await get_redis()
    if redis:
        await redis.set("key", "value")

自动降级: Redis 不可用时返回 None，调用方自行降级到内存实现。
"""

import redis.asyncio as aioredis
from redis.asyncio import Redis

from src.core.config import get_settings
from src.core.logging import get_logger

logger = get_logger(__name__)

_redis: Redis | None = None
_checked = False


async def get_redis() -> Redis | None:
    """获取 Redis 连接。不可用返回 None。"""
    global _redis, _checked
    if _checked:
        return _redis

    settings = get_settings()
    try:
        _redis = aioredis.from_url(
            settings.redis_url,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        await _redis.ping()
        logger.info(f"Redis 已连接: {settings.redis_host}:{settings.redis_port}")
        _checked = True
        return _redis
    except Exception as e:
        logger.warning(f"Redis 不可用 ({e})，降级到内存实现")
        _redis = None
        _checked = True
        return None


async def is_redis_available() -> bool:
    r = await get_redis()
    return r is not None


async def close_redis():
    global _redis, _checked
    if _redis:
        await _redis.close()
        _redis = None
    _checked = False
