"""
sync.py — ES/Milvus 双写一致性
==============================
角色：保证 Elasticsearch 和 Milvus 的数据一致性。
      文档写入流程中，ES 和 Milvus 是两个独立的存储系统，必须保证两者数据一致。

为什么需要双写一致性:
- 混合检索同时依赖 ES (BM25) 和 Milvus (向量)
- 如果 ES 写入成功但 Milvus 失败 → 只有关键词结果，无语义结果
- 如果 Milvus 写入成功但 ES 失败 → 只有语义结果，精确型号查询失败
- 两种都是数据分裂，检索质量严重下降

分布式事务方案 (Phase 1 轻量实现):

写入流程:
  1. 生成 chunk_id
  2. 写 PG transaction 表 (status=pending)
  3. 写 Milvus → 更新 milvus_ok=true
  4. 写 ES → 更新 es_ok=true → status=done
  5. 若步骤3/4失败 → compensation worker 重试

补偿机制 (background worker):
  每分钟扫描 PG 表中 status=pending 的记录
  - milvus_ok 但 es_not_ok → 重试 ES 写入 (指数退避, 最多5次)
  - es_ok 但 milvus_not_ok → 重试 Milvus 写入
  - 5次仍失败 → status=anomaly → 推送运维告警

对账机制 (background worker):
  每10分钟对比 ES/Milvus 文档数
  - 差异 > 0.1% → 触发告警 → 人工介入

使用示例:
    from src.retrieval.sync import dual_write, compensate

    # 写入
    await dual_write(chunk_id, content, embedding, workshop_id, es_client, milvus_client)

    # 补偿
    await compensate()

注意事项:
- Phase 1 使用内存模拟 (PG 事务表为内存 dict, ES/Milvus 为当前进程内存)
- Phase 2 切到真正的 PostgreSQL + Elasticsearch + Milvus
- 补偿 Worker 在 FastAPI lifespan 中启动
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum

from src.core.logging import get_logger

logger = get_logger(__name__)


class TxStatus(str, Enum):
    PENDING = "pending"
    DONE = "done"
    ANOMALY = "anomaly"


@dataclass
class TxRecord:
    chunk_id: str
    milvus_ok: bool = False
    es_ok: bool = False
    status: str = TxStatus.PENDING.value
    retry_count: int = 0
    last_error: str = ""


class DualWriteSync:
    """双写协调器 (Phase 1: 内存实现)。

    Phase 2 替换为 PostgreSQL + 真正的补偿机制。

    示例:
        sync = DualWriteSync()
        await sync.start_tx("doc_chunk_001")
        # ... 写 Milvus ...
        await sync.mark_milvus_ok("doc_chunk_001")
        # ... 写 ES ...
        await sync.mark_es_ok("doc_chunk_001")
    """

    def __init__(self):
        self._transactions: dict[str, TxRecord] = {}
        self._compensating = False

    async def start_tx(self, chunk_id: str) -> None:
        """开始一次双写事务。"""
        self._transactions[chunk_id] = TxRecord(chunk_id=chunk_id)
        logger.debug(f"双写事务开始: {chunk_id}")

    async def mark_milvus_ok(self, chunk_id: str) -> None:
        """标记 Milvus 写入成功。"""
        if chunk_id in self._transactions:
            self._transactions[chunk_id].milvus_ok = True
            await self._check_done(chunk_id)

    async def mark_es_ok(self, chunk_id: str) -> None:
        """标记 ES 写入成功。"""
        if chunk_id in self._transactions:
            self._transactions[chunk_id].es_ok = True
            await self._check_done(chunk_id)

    async def _check_done(self, chunk_id: str) -> None:
        tx = self._transactions.get(chunk_id)
        if tx and tx.milvus_ok and tx.es_ok:
            tx.status = TxStatus.DONE.value
            logger.debug(f"双写事务完成: {chunk_id}")

    async def mark_error(self, chunk_id: str, error: str) -> None:
        """标记写入错误。"""
        if chunk_id in self._transactions:
            tx = self._transactions[chunk_id]
            tx.retry_count += 1
            tx.last_error = error
            if tx.retry_count >= 5:
                tx.status = TxStatus.ANOMALY.value
                logger.error(f"双写异常: {chunk_id}, 重试{tx.retry_count}次: {error}")
            else:
                logger.warning(f"双写重试 {tx.retry_count}/5: {chunk_id}: {error}")

    async def compensate(self, retry_fn) -> int:
        """补偿: 扫描 pending 记录，重试失败的写入。

        参数:
            retry_fn: async callable(chunk_id, target) → bool
                      target = "milvus" or "es"

        返回: 本次修复的记录数

        执行逻辑:
        - pending + milvus_ok but es_not_ok → 重试 ES
        - pending + es_ok but milvus_not_ok → 重试 Milvus
        - 5次重试后标记 anomaly
        """
        fixed = 0
        for chunk_id, tx in list(self._transactions.items()):
            if tx.status != TxStatus.PENDING.value:
                continue
            if tx.retry_count >= 5:
                tx.status = TxStatus.ANOMALY.value
                continue

            if tx.milvus_ok and not tx.es_ok:
                try:
                    success = await retry_fn(chunk_id, "es")
                    if success:
                        tx.es_ok = True
                        fixed += 1
                        await self._check_done(chunk_id)
                    else:
                        await self.mark_error(chunk_id, "ES compensation failed")
                except Exception as e:
                    await self.mark_error(chunk_id, str(e))

            elif tx.es_ok and not tx.milvus_ok:
                try:
                    success = await retry_fn(chunk_id, "milvus")
                    if success:
                        tx.milvus_ok = True
                        fixed += 1
                        await self._check_done(chunk_id)
                    else:
                        await self.mark_error(chunk_id, "Milvus compensation failed")
                except Exception as e:
                    await self.mark_error(chunk_id, str(e))

        if fixed:
            logger.info(f"补偿修复: {fixed} 条")
        return fixed

    def get_anomalies(self) -> list[TxRecord]:
        """获取所有异常记录 (需要人工处理)。"""
        return [tx for tx in self._transactions.values() if tx.status == TxStatus.ANOMALY.value]

    def reconcile(self, es_count: int, milvus_count: int) -> bool:
        """对账: 比较 ES 和 Milvus 的文档数。

        返回: True=一致, False=不一致
        """
        if max(es_count, milvus_count) == 0:
            return True
        diff_pct = abs(es_count - milvus_count) / max(es_count, milvus_count)
        if diff_pct > 0.001:  # 0.1% 差异
            logger.error(f"双写对账异常: ES={es_count}, Milvus={milvus_count}, diff={diff_pct:.3%}")
            return False
        return True


async def dual_write_chunk(
    chunk_id: str,
    content: str,
    embedding: list[float],
    workshop_id: str,
    doc_type: str,
    metadata: dict,
    bm25_retriever,
    dense_retriever,
    sync: DualWriteSync,
) -> bool:
    """写入单个 chunk 到双索引。

    完整流程:
    1. 开启事务
    2. 写 Milvus → 标记成功
    3. 写 ES/BM25 → 标记成功
    4. 两步都成功 → 事务完成

    任何一步失败都会触发补偿机制。

    示例:
        success = await dual_write_chunk(
            chunk_id="doc001_chunk_3",
            content="料筒温度异常处理步骤...",
            embedding=[0.1, -0.2, ...],
            workshop_id="workshop-a",
            doc_type="sop",
            metadata={"equipment_model": "海天MA1200"},
            bm25_retriever=bm25,
            dense_retriever=dense,
            sync=sync,
        )
    """
    await sync.start_tx(chunk_id)

    try:
        # 写 Milvus
        await dense_retriever.insert("mfg_general_knowledge", [{
            "chunk_id": chunk_id,
            "content": content,
            "embedding": embedding,
            "workshop_id": workshop_id,
            "doc_type": doc_type,
            "source": "sop",
            **{k: v for k, v in metadata.items() if k in ("equipment_model", "doc_title", "classification")},
        }])
        await sync.mark_milvus_ok(chunk_id)
    except Exception as e:
        await sync.mark_error(chunk_id, f"Milvus: {e}")
        return False

    try:
        # 写 ES/BM25
        await bm25_retriever.index([{
            "id": chunk_id,
            "content": content,
            "title": metadata.get("doc_title", ""),
            "equipment_model": metadata.get("equipment_model", ""),
        }])
        await sync.mark_es_ok(chunk_id)
    except Exception as e:
        await sync.mark_error(chunk_id, f"ES/BM25: {e}")
        return False

    return True
