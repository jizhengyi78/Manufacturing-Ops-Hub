"""
base.py — Agent 基类
=====================
角色：定义所有 Agent 的统一接口，强制每个 Agent 实现标准方法。
      配合 LangGraph 使用，每个 Agent 对应图中的一个 Node。

Agent 生命周期:
  1. __init__(): 加载 System Prompt、注册 Tools
  2. invoke(state): 被 LangGraph Node 调用，接收 AgentState，返回更新后的 state

标准 Agent 结构:
  class KnowledgeAgent(BaseAgent):
      agent_id = "knowledge"           # 唯一标识
      agent_name = "知识检索 Agent"     # 显示名称

      async def invoke(self, state: AgentState) -> dict:
          # 1. 从 state 获取用户查询
          # 2. 调用检索 → 获取相关文档
          # 3. 调用 LLM → 生成回答
          # 4. 返回 state 更新 dict

Tools 注册 (MCP 风格):
  每个 Agent 通过 register_tool() 注册可调用的工具。
  工具执行前会经过沙箱校验 (权限、速率限制)。
  Phase 1: 工具直接在本地执行，Phase 2: 走 MCP 协议。

使用示例:
    class MyAgent(BaseAgent):
        agent_id = "my_agent"
        agent_name = "我的 Agent"

        async def invoke(self, state: AgentState) -> dict:
            # 业务逻辑
            return {"agent_outputs": {self.agent_id: "处理完成"}}

注意事项:
- 不要在 invoke() 里直接抛出异常，应该把错误写入 state 的 errors 字段
- 每个 Agent 的 invoke 返回 dict 是 LangGraph 的 state update，不是完整 state
- 所有 Agent 共享同一套 retry/fallback 机制 (在 graph edges 中处理)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from src.core.logging import get_logger


@dataclass
class ToolDefinition:
    """MCP 工具定义。

    示例:
        tool = ToolDefinition(
            name="search_sop",
            description="搜索标准操作程序文档",
            parameters={"query": "str", "top_k": "int=5"},
            max_calls_per_minute=30,
            timeout_seconds=10,
            allowed_tables=[],  # 不涉及数据库
        )
    """
    name: str
    description: str
    parameters: dict[str, str]
    handler: Callable | None = None
    max_calls_per_minute: int = 30
    timeout_seconds: int = 10
    max_db_rows: int = 1000
    allowed_tables: list[str] = field(default_factory=list)


class BaseAgent(ABC):
    """所有 Agent 的抽象基类。

    子类必须实现:
    - agent_id: 唯一标识字符串
    - agent_name: 显示名称
    - invoke(): 核心执行逻辑
    - _get_system_prompt(): 返回 System Prompt 模板
    """

    agent_id: str = "base"
    agent_name: str = "基础 Agent"

    def __init__(self):
        self.logger = get_logger(f"agent.{self.agent_id}")
        self._tools: dict[str, ToolDefinition] = {}
        self._system_prompt: str = ""

    def register_tool(self, tool: ToolDefinition) -> None:
        """注册一个工具。

        示例:
            agent.register_tool(ToolDefinition(
                name="search_knowledge",
                description="检索知识库",
                parameters={"query": "str", "top_k": "int=10"},
                handler=self._search_knowledge,
            ))
        """
        self._tools[tool.name] = tool
        self.logger.debug(f"注册工具: {tool.name}")

    async def call_tool(self, tool_name: str, **kwargs) -> Any:
        """调用已注册的工具 (带沙箱校验)。

        Phase 1: 直接调用 handler
        Phase 2: 走 MCP 协议

        示例:
            result = await agent.call_tool("search_knowledge", query="料筒温度异常", top_k=5)
        """
        tool = self._tools.get(tool_name)
        if not tool:
            raise ValueError(f"工具 {tool_name} 未注册到 Agent {self.agent_id}")

        if tool.handler is None:
            raise ValueError(f"工具 {tool_name} 没有绑定的 handler")

        # TODO Phase 2: 加沙箱校验 (速率限制、参数校验、资源限制)
        return await tool.handler(**kwargs)

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """返回 System Prompt 模板字符串 (Jinja2 格式)。

        示例:
            def _get_system_prompt(self) -> str:
                return '''你是一个制造业知识助手。
                当前车间: {{ workshop_id }}
                请基于以下文档回答用户问题: {{ retrieved_context }}
                '''
        """
        ...

    @abstractmethod
    async def invoke(self, state: "AgentState") -> dict:
        """执行 Agent 核心逻辑。

        参数:
            state: 当前 AgentState (LangGraph 传入)

        返回:
            state 的增量更新 dict (不是完整 state)

        示例:
            async def invoke(self, state: AgentState) -> dict:
                query = state.user_query
                results = await self.search(query)
                answer = await self.llm(results, query)
                return {
                    "agent_outputs": {self.agent_id: answer},
                    "citations": results,
                }
        """
        ...
