"""
cache.py — 语义缓存 (权限隔离)
==============================
角色：缓存相似的 LLM 回答，减少重复调用，降低 Token 消耗。

核心安全设计:
缓存 Key = sha256(query) + role + workshop_id + classification
必须 4 个维度全部匹配才命中，防止跨权限缓存泄露。

Phase 1: 内存 dict 实现 (不需要 Redis)
Phase 2: 切 Redis，通过 _use_redis=True 控制

使用示例:
    from src.model.cache import semantic_cache
    cached = await semantic_cache.get(query, role="worker", workshop_id="workshop-a")
    if cached:
        return cached["response"]
"""

import hashlib
import json
import time
from typing import Any

from src.core.logging import get_logger

logger = get_logger(__name__)

DEFAULT_TTL = 3600  # 1 小时


class SemanticCache:
    """权限隔离语义缓存 (Phase 1: 内存实现)。

    Phase 2 切 Redis，接口不变。
    """

    def __init__(self, _use_redis: bool = False):
        self._store: dict[str, tuple[dict, float]] = {}  # {key: (data, expire_at)}

    def _normalize(self, query: str) -> str:
        return " ".join(query.lower().split())

    def _make_key(
        self, query: str, role: str, workshop_id: str,
        classification: str = "internal",
    ) -> str:
        qhash = hashlib.sha256(self._normalize(query).encode()).hexdigest()[:16]
        return f"{qhash}:{role}:{workshop_id}:{classification}"

    async def get(
        self, query: str, role: str, workshop_id: str,
        classification: str = "internal",
    ) -> dict | None:
        key = self._make_key(query, role, workshop_id, classification)
        entry = self._store.get(key)
        if entry is None:
            return None
        data, expire_at = entry
        if time.time() > expire_at:
            del self._store[key]
            return None
        logger.debug(f"语义缓存命中: {key[:40]}")
        return dict(data)  # shallow copy

    async def set(
        self, query: str, response: dict[str, Any], role: str,
        workshop_id: str, classification: str = "internal",
        ttl: int = DEFAULT_TTL,
    ) -> None:
        key = self._make_key(query, role, workshop_id, classification)
        self._store[key] = (response, time.time() + ttl)
        logger.debug(f"语义缓存写入: {key[:40]}")

    async def invalidate_by_doc(self, doc_id: str) -> None:
        self._store.clear()
        logger.info(f"缓存清空 (文档 {doc_id} 变更)")

    async def clear_all(self) -> None:
        self._store.clear()

    async def clean_expired(self) -> int:
        now = time.time()
        expired = [k for k, (_, exp) in self._store.items() if now > exp]
        for k in expired:
            del self._store[k]
        return len(expired)

    @property
    def size(self) -> int:
        return len(self._store)


# 全局单例 (Phase 1: 纯内存)
semantic_cache = SemanticCache()
