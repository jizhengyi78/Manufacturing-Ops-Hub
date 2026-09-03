"""
scheduling.py — 排程优化 Agent 节点
===================================
Phase 2 新增。评估设备故障对生产排程的影响，给出维修窗口建议。
只有维修工及以上角色可触发。
"""

import time
from src.graph.state import AgentState
from src.model.fallback import fallback_chain
from src.model.router import TaskComplexity
from src.core.logging import get_logger

logger = get_logger(__name__)

SCHEDULING_PROMPT = """你是生产排程优化专家。根据设备故障情况和当前排程信息，给出维修时间窗口建议和产能影响评估。

## 回答格式
### 影响评估
- 受影响产线/设备: ...
- 预估产能影响: ...
### 维修窗口建议
建议维修时间: ...（理由: ...）
### 注意事项
需车间主任确认后执行调整。
"""


async def scheduling_node(state: AgentState) -> dict:
    query = state.user_query
    workshop_id = state.user_context.workshop_id if state.user_context else "workshop-a"
    logger.info(f"Scheduling Node: query='{query[:60]}...'")

    # 读取诊断结论作为输入
    diagnosis_result = state.agent_outputs.get("diagnosis", "")
    context = state.retrieved_context

    messages = [
        {"role": "system", "content": SCHEDULING_PROMPT},
        {"role": "user", "content": f"## 故障诊断结果\n{diagnosis_result}\n\n## 文档参考\n{context}\n\n## 用户查询\n{query}"},
    ]

    try:
        result = await fallback_chain.chat_with_fallback(
            messages=messages, complexity=TaskComplexity.COMPLEX, query=query,
        )
    except Exception as e:
        return {"agent_outputs": {**state.agent_outputs, "scheduling": ""}, "errors": state.errors + [f"scheduling: {e}"]}

    return {"agent_outputs": {**state.agent_outputs, "scheduling": result.content}}
