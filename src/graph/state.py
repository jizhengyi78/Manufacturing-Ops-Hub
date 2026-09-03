"""
state.py — LangGraph AgentState 定义
=====================================
角色：定义在 LangGraph 工作流中流转的状态对象。
      每个 Node 读取 state 做决策，返回 state 的增量更新。

AgentState 是整个工作流的数据总线:
┌──────────────────────────────────────────────────────┐
│                    AgentState                        │
│  ┌─────────────┐ ┌─────────────┐ ┌───────────────┐  │
│  │ 用户输入      │ │ 中间结果     │ │ 最终输出      │  │
│  │ user_query   │ │ agent_outputs│ │ final_answer  │  │
│  │ user_context │ │ citations    │ │ errors        │  │
│  │ session_id   │ │ routed_agents│ │               │  │
│  └─────────────┘ └─────────────┘ └───────────────┘  │
└──────────────────────────────────────────────────────┘

字段生命周期:
- user_query: Guard Node 写入，全程只读
- user_context: Guard Node 校验后注入
- routed_agents: Router Node 决定调用哪些 Agent
- agent_outputs: 各 Agent 执行后写入结果
- final_answer: Aggregate Node 写入最终答案
- errors: 任何 Node 异常都写这里，最终反映到响应

使用方式:
    # Node 内部
    async def my_node(state: AgentState) -> dict:
        query = state.user_query
        result = await do_something(query)
        return {"agent_outputs": {"my_agent": result}}

    # LangGraph 自动合并返回的 dict 到 state

注意事项:
- AgentState 是 dataclass，用 TypedDict 或 Pydantic 也可以 (LangGraph 都支持)
- 每个字段都可选 (Optional)，因为 LangGraph 逐步构建 state
- 不要在这里放大量数据 (如完整检索结果)，只放引用和摘要
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from src.security.rbac import UserContext, UserRole


@dataclass
class AgentState:
    """LangGraph 工作流状态。

    从 Guard Node 开始逐步填充，在 Aggregate Node 完成。

    示例 (简化版):
        state = AgentState(
            user_query="注塑机料筒温度异常怎么处理",
            user_context=UserContext("worker_zhang", UserRole.WORKER, "workshop-a"),
            session_id="session_abc",
        )
        # → Graph 执行
        # → state.final_answer = "根据SOP-HA-12, 处理步骤如下..."
    """

    # ── 输入 (Guard Node 填充) ──────────────────
    user_query: str = ""
    """用户输入的原始查询文本。示例: "注塑机料筒温度异常怎么处理" """

    user_context: Optional[UserContext] = None
    """用户身份上下文 (角色/车间/权限)。Guard Node 完成 RBAC 校验后注入。"""

    session_id: str = ""
    """会话 ID，用于关联短期记忆和多轮对话。"""

    # ── 路由 (Router Node 填充) ──────────────────
    routed_agents: list[str] = field(default_factory=list)
    """Router 决定调用的 Agent 列表。示例: ["knowledge"] 或 ["diagnosis", "knowledge"]"""

    query_type: str = ""
    """查询类型分类: "exact" / "mixed" / "semantic"。用于调整 RRF 权重。"""

    # ── 中间结果 (各 Agent Node 填充) ────────────
    agent_outputs: dict[str, str] = field(default_factory=dict)
    """各 Agent 的输出结果。
    示例: {"knowledge": "根据SOP-HA-12, 换模步骤为...", "diagnosis": "根因: 料筒温度传感器故障"}
    """

    citations: list[dict] = field(default_factory=list)
    """引用来源列表。
    示例: [{"doc_id": "sop-HA-12", "title": "换模指导书", "chunk_index": 3}]
    """

    retrieved_context: str = ""
    """检索到的原始文档上下文 (已清洗 + 压缩)。送给 LLM 作为 RAG 的参考材料。"""

    # ── 输出 (Aggregate Node 填充) ───────────────
    final_answer: str = ""
    """最终回答文本。用户看到的内容。"""

    # ── 错误 & 降级 ─────────────────────────────
    errors: list[str] = field(default_factory=list)
    """执行过程中收集的错误信息。
    示例: ["diagnosis_agent: LLM调用超时, 已降级使用规则兜底"]
    """

    fallback_used: bool = False
    """是否使用了降级策略 (备选模型或规则兜底)。"""

    model_used: str = ""
    """实际使用的 LLM 模型。示例: "deepseek-chat" / "qwen-turbo" / "rule_fallback" """

    token_usage: dict[str, int] = field(default_factory=dict)
    """Token 消耗统计。示例: {"prompt": 1200, "completion": 350}"""

    # ── 安全 ────────────────────────────────────
    injection_blocked: bool = False
    """是否在 Guard Node 拦截了注入攻击。"""

    blocked_reason: str = ""
    """拦截原因 (匹配到的注入模式)。"""

    # ── 元信息 ──────────────────────────────────
    latency_ms: float = 0.0
    """总延迟 (从 Guard 到 Aggregate 的耗时)。"""

    checkpoint_step: str = ""
    """当前 Checkpoint 步骤名。用于服务重启恢复。"""
