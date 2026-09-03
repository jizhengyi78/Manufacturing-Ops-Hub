"""report.py — 报表 Agent 节点。自然语言查询 OEE/良率/产量等生产数据。"""

from src.graph.state import AgentState
from src.model.fallback import fallback_chain
from src.model.router import TaskComplexity
from src.core.logging import get_logger

logger = get_logger(__name__)

REPORT_PROMPT = """你是工厂生产数据报表专家。根据查询要求提供关键指标摘要。

## 约束
- 只汇报已有数据，不编造数字
- 成本数据仅对授权角色开放
"""


async def report_node(state: AgentState) -> dict:
    query = state.user_query
    context = state.retrieved_context
    logger.info(f"Report Node: query='{query[:60]}...'")

    messages = [
        {"role": "system", "content": REPORT_PROMPT},
        {"role": "user", "content": f"## 参考数据\n{context}\n\n## 查询\n{query}"},
    ]

    try:
        result = await fallback_chain.chat_with_fallback(
            messages=messages, complexity=TaskComplexity.SIMPLE, query=query,
        )
    except Exception as e:
        return {"agent_outputs": {**state.agent_outputs, "report": ""}, "errors": state.errors + [f"report: {e}"]}

    return {"agent_outputs": {**state.agent_outputs, "report": result.content}}
