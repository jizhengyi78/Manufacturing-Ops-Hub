"""
edges.py — LangGraph 条件边/路由逻辑
=====================================
角色：定义 LangGraph workfow 中节点之间的跳转逻辑。

Phase 1 工作流:
  guard → (blocked? → END) → router → knowledge → aggregate → memory → END

条件路由:
1. route_after_guard(state) → "router" (正常) / END (注入拦截)
2. route_after_router(state) → "knowledge" (Phase1) / 多Agent分发 (Phase2)

注意事项:
- 条件函数返回的是 next node 的 name 字符串
- LangGraph 根据返回值自动路由
"""

from src.graph.state import AgentState
from src.core.logging import get_logger

logger = get_logger(__name__)


def route_after_guard(state: AgentState) -> str:
    """Guard Node 之后的路由。

    返回:
      "router" — 安全通过，进入下一步
      "__end__" — 注入拦截，直接结束 (用户看到拦截信息)

    示例:
        state.injection_blocked = True
        result = route_after_guard(state)
        # result = "__end__"
    """
    if state.injection_blocked:
        logger.warning("Guard 拦截 → 跳过后续节点, 直接返回拦截信息")
        return "__end__"
    return "router"


def route_after_router(state: AgentState) -> list[str]:
    """Router Node 之后的路由。

    Phase 1: 全部路由到 knowledge

    Phase 2: 根据 state.routed_agents 分发:
      - 单 Agent → 直接到那个 Agent 的 node
      - 多 Agent → LangGraph 的 Send API 并行分发

    返回: 目标 node 名列表

    示例:
        state.routed_agents = ["knowledge", "diagnosis"]
        result = route_after_router(state)
        # Phase 2: [Send("diagnosis", state), Send("knowledge", state)]
        # Phase 1: ["knowledge"]
    """
    agents = state.routed_agents

    if not agents:
        logger.warning("Router: 无匹配 Agent, 到 END")
        return ["__end__"]

    # Phase 1: 单 Agent
    return agents
