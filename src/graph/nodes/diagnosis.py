"""
diagnosis.py — 故障诊断 Agent 节点
===================================
Phase 2 新增。分析设备报警和异常现象，给出根因推断。

执行流程:
  1. 从 state 读取用户查询 + 检索上下文
  2. 构建诊断专用 prompt（故障分析格式）
  3. 调用 LLM 生成诊断结论
  4. 返回含置信度的根因分析

与 knowledge agent 的区别:
  - knowledge: 基于文档回答"怎么做"（操作步骤）
  - diagnosis: 基于文档分析"为什么"（根因推断）
"""

import time

from src.graph.state import AgentState
from src.model.fallback import fallback_chain
from src.model.router import TaskComplexity
from src.core.logging import get_logger

logger = get_logger(__name__)

DIAGNOSIS_SYSTEM_PROMPT = """你是一个离散制造设备故障诊断专家，专注于注塑、冲压、CNC 加工设备的故障分析。

## 核心规则
1. **基于文档分析**。只根据提供的设备手册、报警码表、故障案例给出根因推断。
2. **给出置信度**。每个根因标注置信度（高>70%/中40-70%/低<40%）。
3. **列检查步骤**。按优先级列出需要检查的项目和预期结果。
4. **安全提醒**。如果涉及高温、高压、运动部件，必须标注 ⚠️。

## 回答格式
### 根因分析
最可能的根因: ...（置信度: XX%）
次可能的根因: ...（置信度: XX%）

### 检查步骤（按优先级）
1. 检查... — 预期结果: ... — 工具: ...
2. 检查... — 预期结果: ... — 工具: ...

### 安全提醒
⚠️ ...
"""


async def diagnosis_node(state: AgentState) -> dict:
    """故障诊断 Agent 节点。"""
    t0 = time.time()
    query = state.user_query
    user = state.user_context
    workshop_id = user.workshop_id if user else "workshop-a"

    logger.info(f"Diagnosis Node: query='{query[:60]}...'")

    # 使用 Knowledge Node 已检索好的上下文
    retrieved_context = state.retrieved_context

    messages = [
        {"role": "system", "content": DIAGNOSIS_SYSTEM_PROMPT},
        {"role": "user", "content": f"## 设备文档参考\n{retrieved_context}\n\n## 用户问题\n{query}\n\n## 车间\n{workshop_id}"},
    ]

    try:
        result = await fallback_chain.chat_with_fallback(
            messages=messages,
            complexity=TaskComplexity.COMPLEX,
            query=query,
        )
    except Exception as e:
        logger.error(f"Diagnosis LLM 失败: {e}")
        return {
            "agent_outputs": {**state.agent_outputs, "diagnosis": ""},
            "errors": state.errors + [f"diagnosis: {e}"],
        }

    latency = (time.time() - t0) * 1000
    logger.info(f"Diagnosis Node 完成: {latency:.0f}ms")
    return {
        "agent_outputs": {**state.agent_outputs, "diagnosis": result.content},
    }
