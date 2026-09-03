"""
builder.py — LangGraph 工作流构建器
====================================
角色：将所有 Node 和 Edge 组装成完整的 LangGraph StateGraph。

Phase 2 工作流:
  START → guard → router → knowledge → [specialist agents] → aggregate → memory → END

Agent 分发逻辑:
  - knowledge 始终执行（提供检索上下文）
  - router 决定调用哪些 specialist agents
  - knowledge 完成后 → agent_router → 条件分发到各 agent → aggregate

图结构:
  START
    │
    ▼
  guard ──(blocked)──► END
    │(pass)
    ▼
  router
    │
    ▼
  knowledge ─────────────┐
    │                     │
    ▼                     │
  agent_router            │
    │                     │
    ├── diagnosis ────────┤
    ├── inspection ───────┤
    ├── scheduling ───────┤
    ├── quality ──────────┤
    ├── report ───────────┤
    │                     │
    ▼                     ▼
  aggregate ◄─────────────┘
    │
    ▼
  memory
    │
    ▼
   END
"""

from langgraph.graph import StateGraph, END, START

from src.graph.state import AgentState
from src.graph.nodes.guard import guard_node
from src.graph.nodes.router import router_node
from src.graph.nodes.knowledge import create_knowledge_node, KnowledgeNodeContext
from src.graph.nodes.diagnosis import diagnosis_node
from src.graph.nodes.inspection import inspection_node
from src.graph.nodes.scheduling import scheduling_node
from src.graph.nodes.quality import quality_node
from src.graph.nodes.report import report_node
from src.graph.nodes.aggregate import aggregate_node
from src.graph.nodes.memory_node import memory_node
from src.graph.edges import route_after_guard
from src.retrieval.hybrid import HybridRetriever
from src.memory.session import SessionMemory
from src.memory.compressor import ContextCompressor
from src.core.logging import get_logger

logger = get_logger(__name__)


def route_after_knowledge(state: AgentState) -> str:
    """knowledge 完成后, 根据 routed_agents 决定下一个节点。

    逻辑:
    - routed_agents 只有 ["knowledge"] → 直接到 aggregate
    - routed_agents 包含其他 agent → 依次分发（Phase 2 顺序执行）
    """
    agents = state.routed_agents
    # 排除 knowledge 本身
    specialist = [a for a in agents if a != "knowledge"]
    if not specialist:
        return "aggregate"
    # 返回第一个 specialist agent
    return specialist[0]


def route_after_specialist(state: AgentState) -> str:
    """specialist agent 完成后, 决定下一个节点。

    逻辑: 依次执行 specialist agents, 全部完成后到 aggregate。
    """
    agents = [a for a in state.routed_agents if a != "knowledge"]
    executed = [a for a in agents if a in state.agent_outputs]
    remaining = [a for a in agents if a not in executed]
    if not remaining:
        return "aggregate"
    return remaining[0]


def build_graph(
    hybrid_retriever: HybridRetriever,
    session_memory: SessionMemory | None = None,
    compressor: ContextCompressor | None = None,
) -> StateGraph:
    """构建 Phase 2 LangGraph 工作流。"""
    ctx = KnowledgeNodeContext(
        hybrid_retriever=hybrid_retriever,
        session_memory=session_memory,
        compressor=compressor,
    )
    knowledge_fn = create_knowledge_node(ctx)

    graph = StateGraph(AgentState)

    # ── 注册所有节点 ──
    graph.add_node("guard", guard_node)
    graph.add_node("router", router_node)
    graph.add_node("knowledge", knowledge_fn)
    graph.add_node("diagnosis", diagnosis_node)
    graph.add_node("inspection", inspection_node)
    graph.add_node("scheduling", scheduling_node)
    graph.add_node("quality", quality_node)
    graph.add_node("report", report_node)
    graph.add_node("aggregate", aggregate_node)
    graph.add_node("memory", memory_node)

    # ── 接线 ──
    graph.add_edge(START, "guard")
    graph.add_conditional_edges("guard", route_after_guard, {
        "router": "router",
        "__end__": END,
    })
    graph.add_edge("router", "knowledge")

    # knowledge → aggregate (只有 knowledge) 或 → first specialist
    graph.add_conditional_edges("knowledge", route_after_knowledge, {
        "aggregate": "aggregate",
        "diagnosis": "diagnosis",
        "inspection": "inspection",
        "scheduling": "scheduling",
        "quality": "quality",
        "report": "report",
    })

    # 每个 specialist → aggregate 或 → next specialist
    for agent_id in ["diagnosis", "inspection", "scheduling", "quality", "report"]:
        graph.add_conditional_edges(agent_id, route_after_specialist, {
            "aggregate": "aggregate",
            "diagnosis": "diagnosis",
            "inspection": "inspection",
            "scheduling": "scheduling",
            "quality": "quality",
            "report": "report",
        })

    graph.add_edge("aggregate", "memory")
    graph.add_edge("memory", END)

    logger.info("LangGraph Phase 2: guard → router → knowledge → [specialists] → aggregate → memory")
    return graph
