"""
events.py — 事件总线
====================
角色：解耦模块间通信。发布者发事件不需要知道谁在听，订阅者关注事件不需要知道谁发的。

核心设计:
- publish(event, **kwargs): 并发执行所有订阅者，返回结果列表
- publish_one(event, **kwargs): 只发给第一个订阅者，用于请求-响应模式
- 预定义事件名在 Events 类中，避免散落字符串

使用示例:
    from src.core.events import event_bus, Events

    # 订阅
    await event_bus.subscribe(Events.ALARM_RECEIVED, my_handler)

    # 发布
    await event_bus.publish(Events.ALARM_RECEIVED, equipment_id="HA-003")

注意事项:
- 事件处理函数必须是非阻塞的 async 函数
- 订阅者异常不会影响其他订阅者和发布者
- 不要在这里做重业务逻辑，事件总线只做路由不做计算
- A2A 跨 Agent 调用也走这里 (不是 HTTP)
"""

import asyncio
from collections import defaultdict
from typing import Any, Callable, Awaitable

from src.core.logging import get_logger

logger = get_logger(__name__)

Handler = Callable[..., Awaitable[Any]]


class EventBus:
    """基于发布-订阅的异步事件总线。"""

    def __init__(self):
        self._handlers: dict[str, list[Handler]] = defaultdict(list)

    def subscribe(self, event: str, handler: Handler) -> None:
        self._handlers[event].append(handler)
        logger.debug(f"订阅事件: {event} -> {handler.__name__}")

    def unsubscribe(self, event: str, handler: Handler) -> None:
        if handler in self._handlers[event]:
            self._handlers[event].remove(handler)

    async def publish(self, event: str, **kwargs) -> list[Any]:
        """发布事件，并发执行所有订阅者，返回结果列表。"""
        handlers = self._handlers.get(event, [])
        if not handlers:
            return []

        tasks = [handler(**kwargs) for handler in handlers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.error(f"事件 [{event}] handler {handlers[i].__name__} 异常: {r}")
        return results

    async def publish_one(self, event: str, **kwargs) -> Any:
        """发布事件给第一个订阅者，返回单个结果。"""
        handlers = self._handlers.get(event, [])
        if not handlers:
            return None
        return await handlers[0](**kwargs)


# 全局单例
event_bus = EventBus()


# ── 预定义事件 ──────────────────────────────────
class Events:
    """预定义事件名，统一管理避免字符串散落。"""

    # 告警
    ALARM_RECEIVED = "alarm:received"
    ALARM_ESCALATED = "alarm:escalated"

    # 工单
    WORK_ORDER_CREATED = "work_order:created"
    WORK_ORDER_STATUS_CHANGED = "work_order:status_changed"
    WORK_ORDER_COMPLETED = "work_order:completed"

    # 知识沉淀
    KNOWLEDGE_CASE_DRAFTED = "knowledge:case_drafted"
    KNOWLEDGE_CASE_APPROVED = "knowledge:case_approved"
    KNOWLEDGE_CASE_ARCHIVED = "knowledge:case_archived"

    # 文档
    DOCUMENT_INGESTED = "document:ingested"
    DOCUMENT_DELETED = "document:deleted"

    # 系统
    MODE_SWITCHED_TO_OFFLINE = "system:offline"
    MODE_SWITCHED_TO_ONLINE = "system:online"
    CHECKPOINT_RECOVERED = "system:checkpoint_recovered"

    # Token 预算
    TOKEN_BUDGET_WARNING = "token:budget_warning"
    TOKEN_BUDGET_CRITICAL = "token:budget_critical"
