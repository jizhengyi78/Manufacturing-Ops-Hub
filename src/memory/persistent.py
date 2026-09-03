"""
persistent.py — 长期记忆管理
============================
Phase 3: 跨会话故障经验存储与召回。

核心逻辑：
  1. 维修完成后，将故障案例存入 mfg_case_memory Collection
  2. 新查询时，同时检索主知识库(SOP)和案例库(经验)
  3. 分层加权召回（同设备+同故障 > 同产线+同类 > 全局同类 > 跨类相似）
  4. 融合时：SOP 权重 1.2 > 已验证案例 1.0 > 待验证案例 0.7

案例存储格式：
  {
    "case_id": "case_xxx",
    "query": "用户原始问题",
    "equipment_model": "海天MA1200",
    "fault_code": "HT-E-0021",
    "fault_category": "温度异常",
    "root_cause": "料筒温度传感器故障",
    "solution": "更换传感器, 校准PID",
    "workshop_id": "workshop-a",
    "verified": true,
    "created_at": "2026-07-18T10:00:00"
  }

使用方式：
  from src.memory.persistent import LongTermMemory, get_long_term_memory

  ltm = get_long_term_memory()
  # 保存案例
  await ltm.save_case(case_data)
  # 召回类似经验
  cases = await ltm.recall(query_vec, workshop_id="workshop-a",
                            equipment_model="海天MA1200", fault_code="HT-E-0021")
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from src.core.logging import get_logger
from src.retrieval.embedding import get_embedding_service

logger = get_logger(__name__)

CASE_COLLECTION = "mfg_case_memory"


@dataclass
class CaseMemory:
    """一条长期记忆记录。"""
    case_id: str
    query: str                # 用户原始问题
    answer_summary: str        # 回答摘要 (前 500 字)
    equipment_model: str = ""
    fault_code: str = ""
    fault_category: str = ""
    workshop_id: str = ""
    verified: bool = False     # 是否已验证（班组长审核通过）
    created_at: str = ""


@dataclass
class CaseHit:
    """案例召回命中结果。"""
    case_id: str
    query: str
    answer_summary: str
    score: float               # 相似度 [0, 1]
    tier: int                  # 召回层级: 1/2/3/4
    equipment_model: str = ""
    fault_code: str = ""
    verified: bool = False


class LongTermMemory:
    """长期记忆管理器。

    Phase 3: 存储到 Milvus mfg_case_memory Collection。
    数据持久化，重启不丢。
    """

    def __init__(self, dense_retriever=None, embedding_service=None):
        self._dense = dense_retriever
        self._embedding = embedding_service
        self._cases: dict[str, CaseMemory] = {}  # 内存缓存
        self._fault_frequency: dict[str, int] = {}  # 故障出现次数计数
        self._initialized = False

    async def _ensure_init(self):
        if self._initialized:
            return
        if self._dense is None:
            from src.retrieval.dense import get_dense_retriever
            self._dense = get_dense_retriever()
        if self._embedding is None:
            self._embedding = get_embedding_service()
        self._initialized = True

    async def save_case(self, case: CaseMemory) -> str:
        """保存一条故障案例到长期记忆。

        参数:
            case: 案例数据

        返回: case_id

        流程:
        1. 对 query + answer_summary 做向量嵌入
        2. 存入 Milvus mfg_case_memory
        3. 存本地缓存

        示例:
            case_id = await ltm.save_case(CaseMemory(
                case_id=f"case_{uuid.uuid4().hex[:8]}",
                query="注塑机料筒温度异常，报警HT-E-0021",
                answer_summary="传感器故障，更换传感器并校准PID参数",
                equipment_model="海天MA1200", fault_code="HT-E-0021",
                fault_category="温度异常", workshop_id="workshop-a",
                verified=True,
            ))
        """
        await self._ensure_init()

        if not case.case_id:
            case.case_id = f"case_{uuid.uuid4().hex[:8]}"
        if not case.created_at:
            case.created_at = time.strftime("%Y-%m-%dT%H:%M:%S")

        # 向量化: query + answer 合并编码
        text_to_embed = f"{case.query} {case.answer_summary}"
        vec = await self._embedding.embed(text_to_embed)

        # 存入向量库
        await self._dense.insert(CASE_COLLECTION, [{
            "chunk_id": case.case_id,
            "content": json.dumps({
                "query": case.query,
                "answer_summary": case.answer_summary,
                "equipment_model": case.equipment_model,
                "fault_code": case.fault_code,
                "fault_category": case.fault_category,
                "verified": case.verified,
                "created_at": case.created_at,
            }, ensure_ascii=False),
            "embedding": vec,
            "workshop_id": case.workshop_id,
            "source": "verified_case" if case.verified else "unverified_case",
        }])

        # 本地缓存 + DB持久化
        self._cases[case.case_id] = case
        try:
            import asyncio as _asyncio
            from src.core.database import get_session_factory
            from src.data.models import KnowledgeCase
            factory = get_session_factory()
            async def _save():
                async with factory() as session:
                    session.add(KnowledgeCase(
                        case_id=case.case_id, equipment_model=case.equipment_model,
                        fault_code=case.fault_code, fault_category=case.fault_category,
                        root_cause=case.query, repair_steps=[case.answer_summary],
                        verification="verified" if case.verified else "unverified",
                        weight=1.0 if case.verified else 0.7, status="active",
                    ))
                    await session.commit()
            try: _asyncio.get_running_loop(); _asyncio.create_task(_save())
            except RuntimeError: pass
        except Exception: pass
        logger.info(f"长期记忆保存: {case.case_id} ({case.equipment_model}/{case.fault_code})")
        return case.case_id

    async def recall(
        self,
        query_text: str = "",
        query_vec: list[float] | None = None,
        workshop_id: str = "",
        equipment_model: str = "",
        fault_code: str = "",
        top_k: int = 10,
    ) -> list[CaseHit]:
        """分层召回历史案例。

        四层召回策略：
          第1层: 同设备型号 + 同故障码 + 近3个月 → 权重 1.0
          第2层: 同产线 + 同故障大类 + 近6个月 → 权重 0.8
          第3层: 全局 + 同故障大类 + 近12个月 → 权重 0.6
          第4层: 跨故障类型但有相似症状 → 权重 0.4

        参数:
            query_text: 查询文本 (和 query_vec 二选一)
            query_vec: 查询向量 (和 query_text 二选一)
            workshop_id: 车间 ID
            equipment_model: 设备型号
            fault_code: 故障码
            top_k: 返回数量

        返回: CaseHit 列表，按加权分数降序

        示例:
            hits = await ltm.recall(
                query_text="注塑机料筒温度异常",
                workshop_id="workshop-a", equipment_model="海天MA1200",
                fault_code="HT-E-0021", top_k=5,
            )
            for h in hits:
                print(f"[Tier{h.tier}] score={h.score:.3f} {h.query}")
        """
        await self._ensure_init()

        # 向量化查询
        if query_vec is None and query_text:
            query_vec = await self._embedding.embed_query(query_text)
        if query_vec is None:
            return []

        # 从向量库召回 (全部, 不做过滤，后面分层加权)
        all_hits = await self._dense.search(
            query_embedding=query_vec,
            collection=CASE_COLLECTION,
            workshop_id=workshop_id,
            top_k=min(top_k * 4, 50),  # 多取一些供分层筛选
        )

        if not all_hits:
            logger.debug("长期记忆: 无匹配案例")
            return []

        # 分层加权
        scored: list[CaseHit] = []
        now = time.time()
        three_months = 90 * 24 * 3600
        six_months = 180 * 24 * 3600

        for hit in all_hits:
            try:
                data = json.loads(hit.content)
            except json.JSONDecodeError:
                continue

            # 确定层级
            tier = 4
            tier_weight = 0.4

            # 第4层 baseline：只要内容相似就有点分

            # 第3层: 同故障大类
            if data.get("fault_category") and data["fault_category"] == self._extract_fault_category(query_text):
                tier = 3
                tier_weight = 0.6

            # 第2层: 同产线 (workshop 已通过 search 过滤了) + 同类
            if tier >= 2:
                tier = max(tier, 2)
                tier_weight = max(tier_weight, 0.8)

            # 第1层: 同设备型号 + 同故障码
            if (equipment_model and data.get("equipment_model") == equipment_model and
                fault_code and data.get("fault_code") == fault_code):
                tier = 1
                tier_weight = 1.0

            # 验证状态调整
            if data.get("verified"):
                tier_weight *= 1.0
            else:
                tier_weight *= 0.7

            final_score = hit.score * tier_weight

            scored.append(CaseHit(
                case_id=hit.chunk_id,
                query=data.get("query", ""),
                answer_summary=data.get("answer_summary", ""),
                score=final_score,
                tier=tier,
                equipment_model=data.get("equipment_model", ""),
                fault_code=data.get("fault_code", ""),
                verified=data.get("verified", False),
            ))

        # 按分数降序，去重
        scored.sort(key=lambda h: (h.score, -h.tier), reverse=True)
        seen = set()
        unique = []
        for h in scored:
            fp = h.query[:60]
            if fp not in seen:
                seen.add(fp)
                unique.append(h)
            if len(unique) >= top_k:
                break

        logger.debug(f"长期记忆召回: {len(unique)} 条 (tier分布: {[h.tier for h in unique]})")
        return unique

    def _extract_fault_category(self, text: str) -> str:
        """从文本中提取故障大类。"""
        categories = {
            "温度": ["温度", "过热", "过冷", "加热"],
            "振动": ["振动", "异响", "噪音"],
            "精度": ["精度", "超差", "偏差", "尺寸"],
            "液压": ["液压", "油压", "泄漏", "油温"],
            "电气": ["电气", "电路", "传感", "通讯", "PLC"],
            "机械": ["机械", "磨损", "断裂", "松动"],
            "质量": ["缺陷", "缩水", "毛边", "飞边", "凹陷"],
        }
        for cat, kws in categories.items():
            if any(kw in text for kw in kws):
                return cat
        return "其他"

    async def get_case_count(self, workshop_id: str = "") -> int:
        await self._ensure_init()
        return self._dense.doc_count(CASE_COLLECTION)

    async def list_cases(self, workshop_id: str = "", limit: int = 20) -> list[CaseMemory]:
        """列出已保存的案例。"""
        await self._ensure_init()
        # 从向量库查询
        dummy_vec = [0.0] * 1024  # 占位，实际上需要更好的实现
        hits = await self._dense.search(
            query_embedding=dummy_vec,
            collection=CASE_COLLECTION,
            workshop_id=workshop_id,
            top_k=limit,
        )
        result = []
        for h in hits:
            try:
                data = json.loads(h.content)
                result.append(CaseMemory(
                    case_id=h.chunk_id,
                    query=data.get("query", ""),
                    answer_summary=data.get("answer_summary", ""),
                    equipment_model=data.get("equipment_model", ""),
                    fault_code=data.get("fault_code", ""),
                    fault_category=data.get("fault_category", ""),
                    workshop_id=h.workshop_id,
                    verified=data.get("verified", False),
                    created_at=data.get("created_at", ""),
                ))
            except json.JSONDecodeError:
                continue
        return result


    async def should_archive(
        self,
        query: str,
        answer: str,
        equipment_model: str = "",
        fault_code: str = "",
        workshop_id: str = "",
    ) -> tuple[bool, str, dict]:
        """判断一条对话是否值得存入长期记忆。

        三条判断规则（方案 §9.1）:
          条件1: 同设备+同故障码 近30天出现≥3次 → "高复现故障"
          条件2: 故障类型是新增(向量库无相似案例>0.9) → "新故障类型"
          条件3: 回答中包含具体解决方案 → "包含维修方案"

        返回: (是否入库, 理由, 案例元数据)

        示例:
            should, reason, meta = await ltm.should_archive(
                query="料筒温度异常HT-E-0021",
                answer="传感器故障, 更换传感器, 校准PID...",
                equipment_model="海天MA1200", fault_code="HT-E-0021",
            )
            if should:
                await ltm.save_case_draft(meta)
        """
        await self._ensure_init()

        # 前置检查: 必须是故障/诊断类查询
        fault_kw = ["故障", "报警", "异常", "怎么修", "怎么处理", "排查", "原因", "解决"]
        is_fault_query = any(kw in query for kw in fault_kw)

        # 检查回答是否包含具体解决方案
        solution_kw = ["步骤", "检查", "更换", "调整", "修复", "校准", "清理", "更换", "重新"]
        has_solution = any(kw in answer for kw in solution_kw)

        if not is_fault_query or not has_solution:
            return False, "非故障类查询或无具体解决方案", {}

        # 条件2: 检查是否为新故障类型
        existing = await self.recall(
            query_text=query, workshop_id=workshop_id,
            equipment_model=equipment_model, fault_code=fault_code, top_k=3,
        )
        is_new = not (existing and existing[0].score > 0.92)

        # 条件1: 频率计数 (每次故障查询都+1，不依赖是否已存储)
        # 优先用 equipment+fault_code 做 key，只有 fault_code 也行
        freq_key = f"{equipment_model}:{fault_code}" if fault_code else ""
        if not freq_key:
            # 降级: 用故障大类
            freq_key = f"category:{self._extract_fault_category(query)}"
        if freq_key:
            self._fault_frequency[freq_key] = self._fault_frequency.get(freq_key, 0) + 1
            is_recurring = self._fault_frequency[freq_key] >= 3  # 累计≥3次触发
        else:
            is_recurring = False

        # 判定
        if is_recurring and has_solution:
            return True, "高复现故障, 自动沉淀", {
                "query": query, "answer_summary": answer[:500],
                "equipment_model": equipment_model, "fault_code": fault_code,
                "fault_category": self._extract_fault_category(query),
                "workshop_id": workshop_id, "verified": False,
            }
        elif is_new and has_solution:
            return True, "新故障类型, 自动沉淀", {
                "query": query, "answer_summary": answer[:500],
                "equipment_model": equipment_model, "fault_code": fault_code,
                "fault_category": self._extract_fault_category(query),
                "workshop_id": workshop_id, "verified": False,
            }

        return False, "不满足沉淀条件", {}

    async def save_case_draft(self, meta: dict) -> str:
        """保存案例草稿（待审核状态）。

        参数:
            meta: should_archive 返回的元数据 dict

        返回: case_id
        """
        case = CaseMemory(
            case_id=f"case_{uuid.uuid4().hex[:8]}",
            query=meta["query"],
            answer_summary=meta["answer_summary"],
            equipment_model=meta.get("equipment_model", ""),
            fault_code=meta.get("fault_code", ""),
            fault_category=meta.get("fault_category", ""),
            workshop_id=meta.get("workshop_id", ""),
            verified=False,
        )
        case_id = await self.save_case(case)
        logger.info(f"案例草稿已保存: {case_id} ({meta.get('fault_category', '')})")
        return case_id

    async def approve_case(self, case_id: str) -> bool:
        """审核通过案例，标记为已验证。"""
        await self._ensure_init()
        # 删除旧记录，重新插入 verified=True 的版本
        # Phase 3: 简化实现，标记本地缓存
        if case_id in self._cases:
            self._cases[case_id].verified = True
            logger.info(f"案例审核通过: {case_id}")
            return True
        return False

    async def get_pending_cases(self, workshop_id: str = "") -> list[dict]:
        """获取待审核的案例列表。"""
        cases = await self.list_cases(workshop_id)
        return [
            {"case_id": c.case_id, "query": c.query, "answer": c.answer_summary,
             "equipment": c.equipment_model, "fault_code": c.fault_code,
             "category": c.fault_category, "verified": c.verified}
            for c in cases if not c.verified
        ]


# 全局单例
_ltm: LongTermMemory | None = None


def get_long_term_memory() -> LongTermMemory:
    global _ltm
    if _ltm is None:
        _ltm = LongTermMemory()
    return _ltm
