"""
cost.py — Token 成本追踪
========================
角色：统计每次 LLM 调用的 Token 消耗和费用，按车间/Agent/模型三维度聚合。

预算告警机制:
- daily_budget = 50 元/车间/天 (默认，可调)
- 80% 消耗 → 触发 WARNING 事件
- 90% 消耗 → 触发 DEGRADE 事件 (自动降级到更便宜的模型)
- 100% 消耗 → 触发 LIMIT 事件 (拒绝该车间当天新的复杂 LLM 调用)

定价参考 (元/百万 token, 2026年):
- DeepSeek-V3:  prompt ¥1.0, completion ¥2.0
- Qwen-Turbo:   prompt ¥0.3, completion ¥0.6
- 本地模型:      免费

使用示例:
    from src.model.cost import budget_tracker
    total = budget_tracker.record("workshop-a", "knowledge", "deepseek-chat", usage)

注意事项:
- 当前是内存计数器 (进程重启清零)，生产需持久化到 PG 的 token_usage_daily 表
- pricing 需定期更新，模型价格变动频繁
- 本地模型 (离线模式) 成本为 0，不计入预算
"""

import time
from dataclasses import dataclass, field
from datetime import date

from src.core.config import get_settings
from src.core.logging import get_logger
from src.core.events import event_bus, Events

logger = get_logger(__name__)

# 价格: 元/百万 token (2026年参考)
PRICING = {
    "deepseek-chat":    {"prompt": 1.0, "completion": 2.0},
    "deepseek-v3":      {"prompt": 1.0, "completion": 2.0},
    "qwen-turbo":       {"prompt": 0.3, "completion": 0.6},
    "qwen-plus":        {"prompt": 0.8, "completion": 2.0},
    "qwen-2.5-14b-int4":{"prompt": 0.0, "completion": 0.0},  # 本地模型
    "rule_fallback":    {"prompt": 0.0, "completion": 0.0},
}


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    cost: float = 0.0

    def calculate_cost(self) -> float:
        pricing = PRICING.get(self.model, {"prompt": 1.0, "completion": 2.0})
        self.cost = (
            self.prompt_tokens / 1_000_000 * pricing["prompt"]
            + self.completion_tokens / 1_000_000 * pricing["completion"]
        )
        return self.cost


@dataclass
class BudgetTracker:
    """按车间/日的 Token 预算追踪。"""

    _daily_usage: dict[str, float] = field(default_factory=dict)  # {workshop_agent_model_date: cost}
    _daily_budget: float = 50.0  # 默认每车间每天 50 元

    def _key(self, workshop_id: str, agent: str, model: str, dt: date | None = None) -> str:
        dt = dt or date.today()
        return f"{workshop_id}:{agent}:{model}:{dt.isoformat()}"

    def record(self, workshop_id: str, agent: str, model: str, usage: TokenUsage) -> float:
        """记录消耗，返回当日该车间累计成本。"""
        import asyncio
        cost = usage.calculate_cost()
        key = self._key(workshop_id, agent, model)
        self._daily_usage[key] = self._daily_usage.get(key, 0) + cost

        # Phase 3: 持久化到 token_usage_daily 表
        try:
            from datetime import date as dt_date
            from src.core.database import get_session_factory
            from src.data.models import TokenUsageDaily
            factory = get_session_factory()
            async def _save():
                async with factory() as session:
                    from sqlalchemy import select
                    today = dt_date.today()
                    result = await session.execute(
                        select(TokenUsageDaily).where(
                            TokenUsageDaily.date == today,
                            TokenUsageDaily.workshop_id == workshop_id,
                            TokenUsageDaily.agent == agent,
                            TokenUsageDaily.model == model,
                        )
                    )
                    row = result.scalar_one_or_none()
                    if row:
                        row.prompt_tokens += usage.prompt_tokens
                        row.completion_tokens += usage.completion_tokens
                        row.cost += cost
                        row.request_count += 1
                    else:
                        session.add(TokenUsageDaily(
                            date=today, workshop_id=workshop_id, agent=agent, model=model,
                            prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens,
                            cost=cost, request_count=1,
                        ))
                    await session.commit()
            import asyncio as _asyncio
            try: _asyncio.get_running_loop(); _asyncio.create_task(_save())
            except RuntimeError: pass  # 非异步上下文跳过
        except Exception: pass

        # 检查预算告警
        total = self.get_workshop_daily_cost(workshop_id)
        settings = get_settings()
        if total >= self._daily_budget * settings.token_budget_limit_ratio:
            logger.warning(f"车间 [{workshop_id}] Token预算已耗尽: ￥{total:.2f}/{self._daily_budget}")

        return total

    def get_workshop_daily_cost(self, workshop_id: str, dt: date | None = None) -> float:
        dt = dt or date.today()
        prefix = f"{workshop_id}:"
        suffix = f":{dt.isoformat()}"
        return sum(v for k, v in self._daily_usage.items() if k.startswith(prefix) and k.endswith(suffix))

    def is_over_budget(self, workshop_id: str) -> tuple[bool, str]:
        """返回 (是否预警, 级别: ok/warn/degrade/limit)。"""
        settings = get_settings()
        total = self.get_workshop_daily_cost(workshop_id)
        ratio = total / self._daily_budget

        if ratio >= settings.token_budget_limit_ratio:
            return True, "limit"
        elif ratio >= settings.token_budget_degrade_ratio:
            return True, "degrade"
        elif ratio >= settings.token_budget_warn_ratio:
            return True, "warn"
        return False, "ok"


# 全局单例
budget_tracker = BudgetTracker()
