"""inspection.py — 巡检预警 Agent 节点。分析设备状态和报警趋势。"""

from src.graph.state import AgentState
from src.model.fallback import fallback_chain
from src.model.router import TaskComplexity
from src.core.logging import get_logger

logger = get_logger(__name__)

INSPECTION_PROMPT = """你是工厂设备巡检预警专家。分析设备运行状况，给出预警等级和巡检建议。

## 回答格式
### 当前状态
- 整体评估: ...
### 预警清单
| 优先级 | 设备 | 异常指标 | 建议 |
### 巡检建议
- 重点区域: ...
- 建议频次: ...
"""


async def inspection_node(state: AgentState) -> dict:
    query = state.user_query
    context = state.retrieved_context
    logger.info(f"Inspection Node: query='{query[:60]}...'")

    messages = [
        {"role": "system", "content": INSPECTION_PROMPT},
        {"role": "user", "content": f"## 设备数据\n{context}\n\n## 查询\n{query}"},
    ]

    try:
        result = await fallback_chain.chat_with_fallback(
            messages=messages, complexity=TaskComplexity.COMPLEX, query=query,
        )
    except Exception as e:
        return {"agent_outputs": {**state.agent_outputs, "inspection": ""}, "errors": state.errors + [f"inspection: {e}"]}

    return {"agent_outputs": {**state.agent_outputs, "inspection": result.content}}
