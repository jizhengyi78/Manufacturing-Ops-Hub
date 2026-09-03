"""quality.py — 质量分析 Agent 节点。分析产品缺陷与工艺参数的关联。"""

from src.graph.state import AgentState
from src.model.fallback import fallback_chain
from src.model.router import TaskComplexity
from src.core.logging import get_logger

logger = get_logger(__name__)

QUALITY_PROMPT = """你是制造质量分析专家。根据缺陷描述和设备状况，追溯可能的工艺原因并给出调整建议。

## 回答格式
### 缺陷分析
- 缺陷类型: ...
- 可能原因（按概率排序）
### 建议措施
1. ...
### 质检建议
- 是否增加频次: ...
"""


async def quality_node(state: AgentState) -> dict:
    query = state.user_query
    context = state.retrieved_context
    diagnosis = state.agent_outputs.get("diagnosis", "")
    logger.info(f"Quality Node: query='{query[:60]}...'")

    messages = [
        {"role": "system", "content": QUALITY_PROMPT},
        {"role": "user", "content": f"## 故障诊断\n{diagnosis}\n\n## 文档参考\n{context}\n\n## 用户查询\n{query}"},
    ]

    try:
        result = await fallback_chain.chat_with_fallback(
            messages=messages, complexity=TaskComplexity.COMPLEX, query=query,
        )
    except Exception as e:
        return {"agent_outputs": {**state.agent_outputs, "quality": ""}, "errors": state.errors + [f"quality: {e}"]}

    return {"agent_outputs": {**state.agent_outputs, "quality": result.content}}
