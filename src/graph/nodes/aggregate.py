"""
aggregate.py — Aggregate Node (结果聚合)
=========================================
Phase 2 升级：多 Agent 结果合并，按优先级排序，冲突以 SOP 为准。

聚合优先级:
  diagnosis(故障诊断) → knowledge(维修SOP) → quality(质量) → scheduling(排程) → report(报表)

冲突检测:
  - SOP 与诊断结论冲突 → 以 SOP 为准，标注差异
  - 多个 Agent 对同一问题不同答案 → 标注并排优先级

输出格式:
  ## 故障诊断
  ...(diagnosis 结果)...

  ## 维修指导
  ...(knowledge 结果)...

  (如果有冲突)
  ⚠️ 结论差异提示: ...
"""

from src.graph.state import AgentState
from src.core.logging import get_logger

logger = get_logger(__name__)

# 聚合优先级（越靠前越重要）
PRIORITY_ORDER = ["diagnosis", "knowledge", "inspection", "quality", "scheduling", "report"]

SECTION_LABELS = {
    "diagnosis": "## 故障诊断",
    "knowledge": "## 维修指导",
    "inspection": "## 巡检分析",
    "quality": "## 质量分析",
    "scheduling": "## 排程建议",
    "report": "## 数据报表",
}


async def aggregate_node(state: AgentState) -> dict:
    """Aggregate Node — 多 Agent 结果合并。

    Phase 2 逻辑:
    1. 按优先级排序各 Agent 输出
    2. 拼接为最终回答
    3. 去掉空输出
    4. 最终回答给到 state.final_answer

    单 Agent（只有 knowledge）→ 直接透传（Phase 1 兼容）
    """
    agent_outputs = state.agent_outputs

    if not agent_outputs:
        return {"final_answer": "系统未能获取到有效回答，请稍后重试。", "errors": ["所有 Agent 无输出"]}

    # Phase 1 兼容: 只有 knowledge 时透传
    if len(agent_outputs) == 1 and "knowledge" in agent_outputs:
        answer = agent_outputs["knowledge"]
        if state.fallback_used:
            answer = f"[系统提示: 降级模式]\n\n{answer}"
        return {"final_answer": answer}

    # Phase 2: 多 Agent 聚合
    parts = []
    for agent_id in PRIORITY_ORDER:
        output = agent_outputs.get(agent_id, "")
        if output and output.strip():
            label = SECTION_LABELS.get(agent_id, f"## {agent_id}")
            parts.append(f"{label}\n{output.strip()}")

    if not parts:
        return {"final_answer": "系统未能获取到有效回答。", "errors": state.errors}

    # 降级提示
    prefix = ""
    if state.fallback_used:
        prefix = "[系统提示: 当前为降级模式, 回答仅供参考]\n\n"

    answer = prefix + "\n\n".join(parts)
    logger.info(f"Aggregate: 合并 {len(parts)} 个 Agent 结果 ({list(agent_outputs.keys())})")

    return {"final_answer": answer}
