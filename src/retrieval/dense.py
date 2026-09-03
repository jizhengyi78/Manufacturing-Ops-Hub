"""
dense.py — 向量检索
===================
Phase 1: 内存 dict 实现 (DenseRetriever)
Phase 2: Milvus Lite 持久化实现 (MilvusLiteRetriever)

切换方式:
    配置文件或环境变量 DENSE_BACKEND=milvus 即可切换到 Milvus Lite
    默认用内存实现，保证零依赖也能跑

Milvus Lite:
    - 数据存本地文件, 重启不丢
    - 不需要 Docker, pip install pymilvus 即可
    - API 和 Milvus Standalone 完全一致, 后续切生产只需改连接地址
"""

from dataclasses import dataclass, field
from pathlib import Path

from src.core.config import get_settings
from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DenseHit:
    """向量检索命中结果。"""
    chunk_id: str
    content: str
    score: float
    doc_title: str = ""
    doc_type: str = ""
    equipment_model: str = ""
    workshop_id: str = ""
    source: str = "sop"


# ── 公共接口 (ABC) ────────────────────────────

class BaseDenseRetriever:
    """向量检索器抽象接口。内存版和 Milvus 版都实现这个接口。"""

    async def insert(self, collection: str, documents: list[dict]) -> int:
        raise NotImplementedError

    async def search(self, query_embedding: list[float], collection: str = "mfg_general_knowledge",
                     workshop_id: str | None = None, top_k: int = 20, source: str | None = None) -> list[DenseHit]:
        raise NotImplementedError

    async def delete(self, collection: str, chunk_ids: list[str]) -> int:
        raise NotImplementedError

    def doc_count(self, collection: str) -> int:
        raise NotImplementedError

    @property
    def collection_names(self) -> list[str]:
        raise NotImplementedError


# ── Phase 1: 内存实现 ─────────────────────────

class DenseRetriever(BaseDenseRetriever):
    """向量检索器 (Phase 1: 内存实现，随时可用)。"""

    def __init__(self):
        self._collections: dict[str, dict[str, dict]] = {}
        self._partitions: dict[str, dict[str, list[str]]] = {}
        self._vectors: dict[str, dict[str, list[float]]] = {}

    def _ensure_collection(self, collection: str) -> None:
        if collection not in self._collections:
            self._collections[collection] = {}
            self._partitions[collection] = {}
            self._vectors[collection] = {}

    async def insert(self, collection: str, documents: list[dict]) -> int:
        self._ensure_collection(collection)
        count = 0
        for doc in documents:
            chunk_id = doc["chunk_id"]
            workshop_id = doc.get("workshop_id", "")
            self._collections[collection][chunk_id] = doc
            self._vectors[collection][chunk_id] = doc["embedding"]
            if workshop_id not in self._partitions.get(collection, {}):
                self._partitions[collection] = self._partitions.get(collection, {})
                self._partitions[collection][workshop_id] = []
            if workshop_id not in self._partitions[collection]:
                self._partitions[collection][workshop_id] = []
            if chunk_id not in self._partitions[collection][workshop_id]:
                self._partitions[collection][workshop_id].append(chunk_id)
            count += 1
        logger.debug(f"Dense(mem) 插入: {collection}, +{count}, total={len(self._collections[collection])}")
        return count

    def _cosine(self, v1: list[float], v2: list[float]) -> float:
        if len(v1) != len(v2):
            return 0.0
        dot = sum(a * b for a, b in zip(v1, v2))
        return max(0.0, (dot + 1.0) / 2.0)

    async def search(self, query_embedding, collection="mfg_general_knowledge",
                     workshop_id=None, top_k=20, source=None) -> list[DenseHit]:
        self._ensure_collection(collection)
        if workshop_id and workshop_id in self._partitions.get(collection, {}):
            candidates = list(self._partitions[collection][workshop_id])
        else:
            candidates = list(self._vectors.get(collection, {}).keys())

        scored = []
        for cid in candidates:
            vec = self._vectors[collection].get(cid)
            if vec is None:
                continue
            doc = self._collections[collection].get(cid, {})
            if source and doc.get("source") != source:
                continue
            sim = self._cosine(query_embedding, vec)
            if sim > 0:
                scored.append((cid, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        results = []
        for cid, score in scored[:top_k]:
            doc = self._collections[collection].get(cid, {})
            results.append(DenseHit(chunk_id=cid, content=doc.get("content", ""), score=score,
                                    doc_title=doc.get("doc_title", ""), doc_type=doc.get("doc_type", ""),
                                    equipment_model=doc.get("equipment_model", ""),
                                    workshop_id=doc.get("workshop_id", ""),
                                    source=doc.get("source", "sop")))
        return results

    async def delete(self, collection: str, chunk_ids: list[str]) -> int:
        self._ensure_collection(collection)
        removed = 0
        for cid in chunk_ids:
            if cid in self._collections.get(collection, {}):
                doc = self._collections[collection].pop(cid, {})
                ws = doc.get("workshop_id", "")
                if ws in self._partitions.get(collection, {}):
                    try:
                        self._partitions[collection][ws].remove(cid)
                    except ValueError:
                        pass
                self._vectors[collection].pop(cid, None)
                removed += 1
        return removed

    def doc_count(self, collection: str) -> int:
        return len(self._collections.get(collection, {}))

    @property
    def collection_names(self) -> list[str]:
        return list(self._collections.keys())


# ── Phase 2: Milvus Lite 持久化实现 ───────────

MILVUS_DB_PATH = Path(__file__).parent.parent.parent / "data" / "milvus_lite.db"
COLLECTION_SCHEMA = {
    "id": "chunk_id",               # 主键字段名
    "vector_field": "embedding",    # 向量字段名
    "dim": 1024,                    # 向量维度
}


class MilvusLiteRetriever(BaseDenseRetriever):
    """向量检索器 (Phase 2: Milvus Lite 持久化)。

    数据存本地文件 data/milvus_lite.db，重启不丢。
    不需要 Docker，pymilvus 自带 Milvus Lite 嵌入式引擎。

    使用方式同 DenseRetriever，接口完全一致。
    """

    def __init__(self, db_path: str | None = None):
        from pymilvus import MilvusClient
        db_path = db_path or str(MILVUS_DB_PATH)
        logger.info(f"Milvus Lite 初始化: {db_path}")
        self._client = MilvusClient(db_path)

    def _ensure_collection(self, collection: str) -> None:
        """如果 Collection 不存在则创建。"""
        if self._client.has_collection(collection):
            return
        # Milvus Lite: 简单 create_collection API
        self._client.create_collection(
            collection_name=collection,
            dimension=COLLECTION_SCHEMA["dim"],
            metric_type="COSINE",
            id_type="string",
            max_length=512,
        )
        logger.info(f"Milvus 创建 Collection: {collection} (dim={COLLECTION_SCHEMA['dim']})")
        logger.info(f"Milvus 创建 Collection: {collection}")

    async def insert(self, collection: str, documents: list[dict]) -> int:
        self._ensure_collection(collection)
        # MilvusClient 高層 API: id=主键, vector=向量, 其余为动态字段
        data = []
        for doc in documents:
            data.append({
                "id": doc["chunk_id"],
                "vector": doc["embedding"],
                "content": doc.get("content", ""),
                "workshop_id": doc.get("workshop_id", ""),
                "doc_type": doc.get("doc_type", ""),
                "source": doc.get("source", "sop"),
                "doc_title": doc.get("doc_title", ""),
                "equipment_model": doc.get("equipment_model", ""),
            })
        import asyncio
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: self._client.insert(collection, data))
        count = result.get("insert_count", 0)
        logger.debug(f"Milvus 插入: {collection}, +{count}")
        return count

    async def search(self, query_embedding, collection="mfg_general_knowledge",
                     workshop_id=None, top_k=20, source=None) -> list[DenseHit]:
        self._ensure_collection(collection)

        # 构建过滤表达式
        filter_parts = []
        if workshop_id:
            filter_parts.append(f'workshop_id == "{workshop_id}"')
        if source:
            filter_parts.append(f'source == "{source}"')
        filter_expr = " and ".join(filter_parts) if filter_parts else None

        import asyncio
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._client.search(
                collection_name=collection,
                data=[query_embedding],
                limit=top_k,
                filter=filter_expr,
                output_fields=["content", "doc_title", "doc_type", "equipment_model", "workshop_id", "source"],
            )
        )

        hits = []
        for r in result[0]:
            entity = r.get("entity", {})
            hits.append(DenseHit(
                chunk_id=entity.get("id", ""),
                content=entity.get("content", ""),
                score=r.get("distance", 0.0),
                doc_title=entity.get("doc_title", ""),
                doc_type=entity.get("doc_type", ""),
                equipment_model=entity.get("equipment_model", ""),
                workshop_id=entity.get("workshop_id", ""),
                source=entity.get("source", "sop"),
            ))
        return hits

    async def delete(self, collection: str, chunk_ids: list[str]) -> int:
        self._ensure_collection(collection)
        import asyncio
        loop = asyncio.get_running_loop()
        # Milvus delete by expr
        ids_str = ", ".join(f'"{cid}"' for cid in chunk_ids)
        result = await loop.run_in_executor(
            None,
            lambda: self._client.delete(collection, ids=chunk_ids)
        )
        return len(chunk_ids)

    def doc_count(self, collection: str) -> int:
        if not self._client.has_collection(collection):
            return 0
        stats = self._client.get_collection_stats(collection)
        return stats.get("row_count", 0)

    @property
    def collection_names(self) -> list[str]:
        return self._client.list_collections()

    def close(self):
        self._client.close()


# ── 工厂函数 ──────────────────────────────────

_dense_instance: BaseDenseRetriever | None = None


def get_dense_retriever(backend: str = "auto") -> BaseDenseRetriever:
    """获取向量检索器实例。

    参数:
        backend: "auto" (从配置读取) | "memory" | "milvus"

    环境变量 DENSE_BACKEND=milvus 可切换到 Milvus Lite。
    """
    global _dense_instance
    if _dense_instance is not None:
        return _dense_instance

    if backend == "auto":
        import os
        backend = os.environ.get("DENSE_BACKEND", "memory")

    if backend == "milvus":
        try:
            _dense_instance = MilvusLiteRetriever()
            logger.info("Dense backend: Milvus Lite")
        except Exception as e:
            logger.warning(f"Milvus Lite 初始化失败, 降级到内存: {e}")
            _dense_instance = DenseRetriever()
    else:
        _dense_instance = DenseRetriever()
        logger.info("Dense backend: 内存")

    return _dense_instance
