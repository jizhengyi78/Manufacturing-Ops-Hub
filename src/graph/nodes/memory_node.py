"""
memory_node.py — Memory Node (记忆持久化 + 知识沉淀)
=====================================================
Phase 3 升级: 对话完成后自动评估是否存入长期记忆。

判定逻辑 (should_archive):
  1. 是否为故障/诊断类查询? (含报警码/故障关键词)
  2. 回答是否包含具体方案? (步骤+更换+调整等)
  3. 是高频复现故障? (同设备+同故障30天内≥3次)
  4. 是新故障类型? (向量库无相似案例)
  → 满足 3+2 或 4+2 → 自动生成案例草稿 → 待班组长审核
"""

from src.graph.state import AgentState
from src.graph.checkpoint import get_checkpoint_manager
from src.core.logging import get_logger

logger = get_logger(__name__)


async def memory_node(state: AgentState) -> dict:
    """Memory Node — Checkpoint + 长期记忆评估。

    Phase 1: 仅 Checkpoint 保存
    Phase 3: + 知识自动沉淀评估
    """
    session_id = state.session_id
    if not session_id:
        return {}

    # 1. Checkpoint 持久化
    ckpt = get_checkpoint_manager()
    await ckpt.save(session_id, "memory", state)
    await ckpt.renew(session_id)

    # 2. Phase 3: 知识沉淀评估
    try:
        await _evaluate_knowledge_archival(state)
    except Exception as e:
        logger.info(f"知识沉淀评估异常: {e}")

    return {}


async def _evaluate_knowledge_archival(state: AgentState):
    """评估当前对话是否值得存入长期记忆。"""
    query = state.user_query
    answer = state.final_answer or ""
    if not answer or len(answer) < 50:
        logger.debug(f"知识沉淀: 回答太短 ({len(answer)} chars)")
        return

    # 提取设备型号和故障码
    import re
    alarm_match = re.search(r'[A-Z]{2,}-[A-Z0-9]+-\d+', query)
    fault_code = alarm_match.group(0) if alarm_match else ""

    model_match = re.search(r'(海天MA\d+|FANUC|扬力\w+)', query)
    equipment_model = model_match.group(0) if model_match else ""
    if not equipment_model and answer:
        model_match = re.search(r'(海天MA\d+|FANUC|扬力\w+)', answer)
        equipment_model = model_match.group(0) if model_match else ""

    workshop_id = state.user_context.workshop_id if state.user_context else "workshop-a"

    logger.info(
        f"知识沉淀评估: fc={fault_code}, eq={equipment_model}, "
        f"ws={workshop_id}, ans_len={len(answer)}"
    )

    from src.memory.persistent import get_long_term_memory
    ltm = get_long_term_memory()

    should, reason, meta = await ltm.should_archive(
        query=query, answer=answer,
        equipment_model=equipment_model, fault_code=fault_code,
        workshop_id=workshop_id,
    )

    if should:
        case_id = await ltm.save_case_draft(meta)
        logger.info(
            f"知识沉淀: {reason} -> case_id={case_id} "
            f"(设备={equipment_model}, 故障码={fault_code})"
        )
    else:
        logger.info(f"知识沉淀跳过: {reason}")
