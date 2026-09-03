"""
fallback.py — 降级策略
======================
角色：当 LLM API 故障时，自动降级保证服务不中断。
降级链: 主模型 → 备选模型 → 规则兜底

核心机制:
1. 三层降级: 主模型失败 → 自动切备选 → 备选也失败 → 规则库兜底
2. 全局降级锁: 触发规则兜底后保持5分钟不切回，防止"大模型恢复→切回去又挂"的振荡
3. 熔断联动: 与 retry.py 的 CircuitBreaker 配合，熔断后自动触发降级

使用示例:
    from src.model.fallback import fallback_chain
    result = await fallback_chain.chat_with_fallback(
        messages=[...],
        rule_fallback_fn=lambda q: f"系统降级中。问题: {q}",
        query="料筒温度异常"
    )

注意事项:
- 规则库初始化在 seed 数据中，Phase1 包含 10-20 条核心报警码
- 全局降级锁是进程级别的，多实例通过 Redis 共享熔断状态
- 规则降级返回的答案要明确告知用户"当前使用降级模式"
"""

import time
from typing import AsyncIterator

from src.core.config import get_settings
from src.core.exceptions import AllModelsFailedError, ModelCircuitOpenError
from src.core.logging import get_logger
from src.core.retry import CircuitBreaker, async_retry, default_retry
from src.model.router import ModelResult, ModelRouter, TaskComplexity, router

logger = get_logger(__name__)

# 全局降级锁: 触发规则兜底后至少保持降级状态 5 分钟
_degrade_lock_until: float = 0.0
DEGRADE_LOCK_SECONDS = 300


class FallbackChain:
    """模型降级链: 主模型 → 备选模型 → 规则兜底。"""

    def __init__(self):
        settings = get_settings()
        self.primary_model = settings.default_model
        self.fallback_model = settings.fallback_model
        self.primary_cb = CircuitBreaker(
            name=f"model:{self.primary_model}",
            failure_threshold=settings.circuit_failure_threshold,
            cooldown_seconds=settings.circuit_cooldown_seconds,
            max_cooldown_seconds=settings.circuit_max_cooldown_seconds,
        )
        self.fallback_cb = CircuitBreaker(
            name=f"model:{self.fallback_model}",
            failure_threshold=settings.circuit_failure_threshold,
            cooldown_seconds=settings.circuit_cooldown_seconds,
            max_cooldown_seconds=settings.circuit_max_cooldown_seconds,
        )

    def _is_globally_degraded(self) -> bool:
        return time.time() < _degrade_lock_until

    def _set_global_degrade(self) -> None:
        global _degrade_lock_until
        _degrade_lock_until = time.time() + DEGRADE_LOCK_SECONDS
        logger.warning(f"全局降级锁激活: {DEGRADE_LOCK_SECONDS}秒内不切回大模型")

    async def chat_with_fallback(
        self,
        messages: list[dict],
        complexity: TaskComplexity | None = None,
        rule_fallback_fn=None,  # callable(query) -> str | None
        query: str = "",
        stream: bool = False,
    ) -> ModelResult:
        """带降级链的对话调用。"""

        # Step 1: 尝试主模型
        model = self.primary_model if not complexity else router.select_model(complexity)
        fallback_used = False

        try:
            if self._is_globally_degraded():
                raise ModelCircuitOpenError("全局降级锁生效")

            result = await async_retry(
                router.chat,
                messages=messages,
                model=model,
                retry_config=default_retry,
                circuit_breaker=self.primary_cb,
            )
            return result
        except (ModelCircuitOpenError, Exception) as e:
            logger.warning(f"主模型 [{model}] 不可用: {e}")
            fallback_used = True

        # Step 2: 尝试备选模型
        try:
            result = await async_retry(
                router.chat,
                messages=messages,
                model=self.fallback_model,
                retry_config=default_retry,
                circuit_breaker=self.fallback_cb,
            )
            result.fallback_used = True
            return result
        except (ModelCircuitOpenError, Exception) as e:
            logger.warning(f"备选模型 [{self.fallback_model}] 不可用: {e}")

        # Step 3: 规则库兜底
        self._set_global_degrade()
        from src.model.rule_engine import match_rule
        rule_result = match_rule(query)
        if rule_result:
            logger.info(f"规则库命中: {query[:50]}")
            return ModelResult(content=rule_result, model="rule_fallback", fallback_used=True)
        if rule_fallback_fn and query:
            fallback_text = await rule_fallback_fn(query) if callable(rule_fallback_fn) else None
            if fallback_text:
                logger.info("使用自定义降级兜底")
                return ModelResult(content=fallback_text, model="rule_fallback", fallback_used=True)

        raise AllModelsFailedError("所有模型 + 规则兜底均不可用")

    async def chat_stream_with_fallback(
        self,
        messages: list[dict],
        complexity: TaskComplexity | None = None,
        query: str = "",
    ) -> AsyncIterator[str]:
        """流式降级调用 (简化版: 仅尝试主模型, 失败直接规则兜底)。"""
        model = self.primary_model if not complexity else router.select_model(complexity)

        try:
            if self._is_globally_degraded():
                raise ModelCircuitOpenError("全局降级锁生效")

            async for chunk in router.chat_stream(messages=messages, model=model):
                yield chunk
            return
        except Exception as e:
            logger.warning(f"流式主模型失败: {e}")

        # Fallback: 规则兜底输出
        fallback_text = f"系统降级中，请拨打维修热线或查阅设备手册。\n问题: {query}"
        yield fallback_text


# 全局单例
fallback_chain = FallbackChain()
