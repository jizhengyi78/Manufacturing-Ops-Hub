"""
router.py — Router Node (意图路由)
===================================
Phase 2 升级：LLM 意图分类 + 多 Agent 分发。

路由策略:
  1. 关键词快速匹配（正则，<1ms）
  2. 命中 → 直接路由到对应 Agent
  3. 未命中 → 用轻量 LLM 做意图分类
  4. 所有路由默认包含 knowledge agent（提供文档支撑）

Agent 覆盖范围:
  - diagnosis:  故障分析、根因推断
  - inspection: 巡检预警、设备状态分析
  - scheduling: 排程优化、产能评估
  - quality:    质量分析、缺陷追溯
  - report:     数据报表、OEE查询
  - knowledge:  文档检索（所有查询都会走的基底Agent）
"""

import re
import time

from src.graph.state import AgentState
from src.model.router import router as model_router, TaskComplexity
from src.core.logging import get_logger

logger = get_logger(__name__)

# 意图 → Agent 映射（关键词匹配）
INTENT_KEYWORDS: dict[str, list[str]] = {
    "diagnosis":  ["故障", "报警", "异常", "不工作", "坏了", "停机", "怎么修", "排查原因", "根因",
                   "怎么回事", "出问题", "报错", "error", "alarm", "fault"],
    "inspection": ["巡检", "点检", "运行状态", "监控", "预警", "趋势", "检查项目", "每天检查"],
    "scheduling": ["排程", "排产", "产能", "加班", "调班", "交货期", "能不能做", "产能够"],
    "quality":    ["缺陷", "不良", "废品", "质量", "良率", "缩水", "毛边", "飞边", "尺寸超差",
                   "表面", "外观", "颜色", "变形", "裂开"],
    "report":     ["OEE", "报表", "统计", "产量", "本月", "昨天", "趋势", "对比", "多少",
                   "合格率", "良品率", "良率", "产出", "效率", "下降", "增长", "上升",
                   "环比", "同比", "每日", "每周", "月度"],
}

# 关键词匹配的意图优先级（某些词可能匹配多个意图，按优先级选）
INTENT_PRIORITY = ["diagnosis", "inspection", "quality", "scheduling", "report"]


def _keyword_classify(query: str) -> list[str]:
    """关键词快速分类，返回匹配的 Agent 列表。

    示例:
        _keyword_classify("料筒温度异常怎么处理")
        → ["diagnosis", "knowledge"]   （"异常"命中 diagnosis）

        _keyword_classify("注塑机怎么换模具")
        → ["knowledge"]                 （无关键词命中，兜底）

        _keyword_classify("最近良率怎么样")
        → ["report", "quality", "knowledge"]  （"良率"同时命中 report 和 quality）
    """
    agents = set()
    for agent, keywords in INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in query:
                agents.add(agent)
                break  # 一个关键词命中就够
    agents.add("knowledge")  # knowledge 始终参与
    return list(agents)


async def _llm_classify(query: str) -> list[str]:
    """LLM 意图分类（关键词未命中时的兜底）。

    用轻量模型做一次快速分类，返回 Agent 列表。
    延迟 ~300-500ms，仅兜底时使用。
    """
    prompt = f"""分析以下用户查询，判断属于哪类任务。可多选。

查询: "{query}"

任务类型:
- diagnosis: 设备故障诊断、报警处理、根因分析
- inspection: 巡检、点检、设备状态检查
- scheduling: 排程、产能、交货期
- quality: 产品质量、缺陷分析、良率
- report: 数据报表、统计查询、OEE

回答格式: 只输出任务类型，多个用逗号分隔。
示例: "diagnosis,knowledge"
"""

    try:
        result = await model_router.chat(
            messages=[{"role": "user", "content": prompt}],
            model=model_router.simple_task_model,
            max_tokens=50,
            temperature=0,
        )
        raw = result.content.strip().lower()
        agents = [a.strip() for a in raw.split(",") if a.strip() in INTENT_KEYWORDS]
        if not agents:
            agents = ["knowledge"]
        agents.append("knowledge")
        return list(set(agents))
    except Exception:
        return ["knowledge"]


async def router_node(state: AgentState) -> dict:
    """Router Node — 意图分类 + Agent 分发。

    Phase 2: 关键词优先 → LLM 兜底 → 多 Agent 分发。

    返回: routed_agents + query_type
    """
    query = state.user_query
    t0 = time.time()

    # 1. 关键词快速分类
    agents = _keyword_classify(query)

    # 2. 只有 knowledge → 尝试 LLM 分类（可能是长尾意图）
    if agents == ["knowledge"] and len(query) > 10:
        agents = await _llm_classify(query)

    # 3. 查询类型（用于 RRF 权重）
    alarm_pattern = re.compile(r'\b[A-Z]{2,}-\w{3,}\b')
    model_pattern = re.compile(r'([A-Z]{2,}\d{2,})')
    has_alarm = bool(alarm_pattern.search(query))
    has_model = bool(model_pattern.search(query))

    if has_alarm:
        query_type = "exact"
    elif has_model and any(kw in query for kw in ["异常", "故障", "报警", "处理", "维修"]):
        query_type = "mixed"
    elif has_model:
        query_type = "exact"
    else:
        query_type = "semantic"

    latency = (time.time() - t0) * 1000
    logger.info(f"Router: agents={agents}, query_type={query_type}, latency={latency:.0f}ms")

    return {
        "routed_agents": agents,
        "query_type": query_type,
    }
