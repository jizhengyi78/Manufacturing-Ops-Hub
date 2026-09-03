"""
hybrid.py — 混合检索编排
=========================
角色：混合检索的总入口，编排 BM25 + Dense 两路召回 → RRF 融合 → Re-ranker 精排。

检索流程 (6步):
┌─────────────────┐
│ 1. 查询分类      │ ← 精确型号? 模糊语义? 混合?
├─────────────────┤
│ 2. 并行召回      │
│   BM25 (ES)      │ ← 关键词精确匹配
│   Dense (Milvus) │ ← 语义向量相似
├─────────────────┤
│ 3. RRF 融合      │ ← 动态权重 (按查询类型调整 BM25:dense 比例)
├─────────────────┤
│ 4. 去重合并      │ ← 相同内容保留分数最高的
├─────────────────┤
│ 5. Re-ranker     │ ← Cross-encoder 精排
├─────────────────┤
│ 6. 返回 Top-K    │ ← 标注来源 (sop/case)
└─────────────────┘

动态 RRF 权重 — 根据查询类型自动调整:

查询类型 "精确型号":
  判断条件: LLM 分类为 exact 或正则命中设备型号/报警码
  权重: BM25 0.7 : Dense 0.3
  场景: "海天MA1200 润滑点位置"、"HT-E-0021 报警码含义"

查询类型 "混合组合":
  判断条件: LLM 分类为 mixed 或同时含型号+故障描述
  权重: BM25 0.5 : Dense 0.5
  场景: "海天MA1200 料筒温度异常怎么处理"

查询类型 "模糊语义":
  判断条件: LLM 分类为 semantic 或无语义型号
  权重: BM25 0.3 : Dense 0.7
  场景: "打出来的件有毛边怎么回事"

RRF (Reciprocal Rank Fusion) 公式:
  score = sum( 1 / (k + rank_i) ), 其中 rank_i 是第 i 个检索器的排名, k=60

使用示例:
    from src.retrieval.hybrid import HybridRetriever

    retriever = HybridRetriever(bm25_retriever, dense_retriever, reranker)
    results = await retriever.search(
        query="海天MA1200 料筒温度异常怎么处理",
        workshop_id="workshop-a",
        top_k=10,
    )
    for r in results:
        print(f"[{r['source']}] {r['relevance_score']:.3f} | {r['content'][:60]}...")

注意事项:
- 查询类型分类优先用正则 (不调 LLM)，正则没命中才用 LLM 分类
- 主知识库 (source=sop) 和长期记忆 (source=case) 分两个 Collection 分别召回
- 融合时 SOP 权重 1.2，已审核案例 1.0，待验证案例 0.7
- 检索失败不抛异常，返回空列表 + 日志记录
"""

import re
from typing import Optional

from src.core.config import get_settings
from src.core.logging import get_logger
from src.retrieval.bm25 import BM25Retriever, BM25Hit
from src.retrieval.dense import DenseRetriever, DenseHit
from src.retrieval.reranker import Reranker, RerankResult
from src.retrieval.embedding import get_embedding_service

logger = get_logger(__name__)


class QueryType:
    """查询类型分类。"""
    EXACT = "exact"        # 精确型号/报警码
    MIXED = "mixed"        # 型号 + 故障组合
    SEMANTIC = "semantic"  # 模糊语义


class HybridRetriever:
    """混合检索编排器。

    将所有检索组件组合成一个完整的混合检索管线。

    示例:
        retriever = HybridRetriever(bm25, dense, reranker)
        results = await retriever.search("料筒温度异常怎么处理", "workshop-a")
    """

    # 报警码正则: 如 HT-E-0021, CN-E-0015
    ALARM_CODE_PATTERN = re.compile(r'\b[A-Z]{2,}-\w{3,}\b')
    # 设备型号: 中文品牌+字母数字
    MODEL_PATTERN = re.compile(r'([A-Z]{2,}\d{2,}|[A-Z]+-[A-Z]+\d+)|\b[A-Z]{2,}\d{3,}\b')

    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        reranker: Reranker,
    ):
        self.bm25 = bm25
        self.dense = dense
        self.reranker = reranker
        self.embedding = get_embedding_service()

    def classify_query(self, query: str) -> str:
        """查询类型分类 — 正则先行，正则没命中再走 LLM。

        分类逻辑:
        1. 正则命中报警码 → EXACT
        2. 正则命中设备型号 → 检查是否同时有故障描述词
           - 有 → MIXED
           - 无 → EXACT
        3. 都没命中 → SEMANTIC

        示例:
            >>> retriever.classify_query("HT-E-0021")
            "exact"
            >>> retriever.classify_query("海天MA1200 料筒温度异常")
            "mixed"
            >>> retriever.classify_query("打出来的件有毛边")
            "semantic"
        """
        has_alarm = bool(self.ALARM_CODE_PATTERN.search(query))
        has_model = bool(self.MODEL_PATTERN.search(query))

        if has_alarm:
            return QueryType.EXACT

        if has_model:
            fault_kw = ["异常", "故障", "报警", "怎么", "处理", "维修", "排查", "原因"]
            if any(kw in query for kw in fault_kw):
                return QueryType.MIXED
            return QueryType.EXACT

        return QueryType.SEMANTIC

    def _rrf_fusion(
        self,
        bm25_hits: list[BM25Hit],
        dense_hits: list[DenseHit],
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
        k: int = 60,
    ) -> list[dict]:
        """RRF (Reciprocal Rank Fusion) 融合。

        公式: score = weight * sum(1 / (k + rank))

        为什么用 RRF 而不是简单平均:
        - RRF 不依赖原始分数的归一化 (BM25 和余弦相似度的 scale 不同)
        - RRF 对排名稳定: 某个检索器返回的极端分数不会主导结果

        参数:
            bm25_hits: BM25 结果
            dense_hits: 向量检索结果
            bm25_weight: BM25 权重
            dense_weight: Dense 权重
            k: 平滑参数, 默认60

        返回: 融合后的候选列表 [{id, content, score, source}]
        """
        fusion_scores: dict[str, dict] = {}  # chunk_id → {score, content, source}

        # BM25 贡献
        for rank, hit in enumerate(bm25_hits):
            chunk_id = hit.chunk_id or hit.content[:80]
            rrf_score = bm25_weight / (k + rank + 1)
            if chunk_id not in fusion_scores:
                fusion_scores[chunk_id] = {"content": hit.content, "score": 0.0, "source": "bm25"}
            fusion_scores[chunk_id]["score"] += rrf_score

        # Dense 贡献
        for rank, hit in enumerate(dense_hits):
            chunk_id = hit.chunk_id or hit.content[:80]
            rrf_score = dense_weight / (k + rank + 1)
            if chunk_id not in fusion_scores:
                fusion_scores[chunk_id] = {"content": hit.content, "score": 0.0, "source": "dense"}
            fusion_scores[chunk_id]["score"] += rrf_score
            # 如果 dense 来源是 sop/case，覆盖 source
            if hit.source:
                fusion_scores[chunk_id]["source"] = hit.source

        # 转为列表
        fused = [
            {"id": cid, "content": info["content"], "score": info["score"], "source": info["source"]}
            for cid, info in fusion_scores.items()
        ]
        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused

    async def search(
        self,
        query: str,
        workshop_id: str,
        top_k: int = 10,
        include_sop: bool = True,
        include_cases: bool = False,
        equipment_model: str | None = None,
    ) -> list[dict]:
        """混合检索主入口。

        参数:
            query: 用户查询
            workshop_id: 车间 ID (必传，用于分区过滤)
            top_k: 最终返回数量
            include_sop: 是否检索主知识库
            include_cases: 是否检索长期记忆 (Phase 3)
            equipment_model: 设备型号 (可选，用于精确过滤)

        返回:
            [{chunk_id, content, relevance_score, source, doc_title, ...}, ...]

        完整检索流程示例:
            retriever = HybridRetriever(bm25, dense, reranker)
            results = await retriever.search(
                query="料筒温度异常怎么处理",
                workshop_id="workshop-a",
                top_k=5,
            )
            for r in results:
                source_label = "SOP" if r['source'] == 'sop' else "案例"
                print(f"[{source_label}] score={r['relevance_score']:.3f}")
                print(f"  {r['content'][:80]}...")
        """
        settings = get_settings()

        # 1. 查询分类 → 确定 RRF 权重
        query_type = self.classify_query(query)

        if query_type == QueryType.EXACT:
            bm25_w, dense_w = settings.hybrid_bm25_weight_exact, 1 - settings.hybrid_bm25_weight_exact
        elif query_type == QueryType.MIXED:
            bm25_w, dense_w = settings.hybrid_bm25_weight_mixed, 1 - settings.hybrid_bm25_weight_mixed
        else:
            bm25_w, dense_w = settings.hybrid_bm25_weight_semantic, 1 - settings.hybrid_bm25_weight_semantic

        logger.info(f"混合检索: type={query_type}, bm25_w={bm25_w:.1f}, dense_w={dense_w:.1f}, q='{query[:50]}'")

        # 2. 并行召回
        bm25_future = self.bm25.search(query, top_k=30)
        query_vec = await self.embedding.embed_query(query)

        dense_futures = []
        if include_sop:
            dense_futures.append(
                self.dense.search(query_vec, "mfg_general_knowledge", workshop_id, top_k=30, source="sop")
            )
        if include_cases:
            dense_futures.append(
                self.dense.search(query_vec, "mfg_case_memory", workshop_id, top_k=30, source="case")
            )

        # 等两路结果
        import asyncio
        bm25_hits = await bm25_future
        dense_results = await asyncio.gather(*dense_futures) if dense_futures else []

        # 合并所有 Dense 结果
        all_dense_hits: list[DenseHit] = []
        for hits in dense_results:
            all_dense_hits.extend(hits)

        # 3. RRF 融合 → 30 条候选
        fused = self._rrf_fusion(bm25_hits, all_dense_hits, bm25_w, dense_w)
        candidates_30 = fused[:30]

        if not candidates_30:
            logger.warning(f"混合检索无结果: q='{query[:50]}'")
            return []

        # 4. Re-ranker 精排 → Top-K
        reranked = await self.reranker.rerank(
            query=query,
            candidates=candidates_30,
            top_k=top_k,
            equipment_model=equipment_model,
        )

        # 5. 构建返回结果
        results = []
        for r in reranked:
            results.append({
                "chunk_id": r.chunk_id,
                "content": r.content,
                "relevance_score": r.relevance_score,
                "source": r.source,
                "doc_title": "",
                "equipment_model": equipment_model or "",
            })

        logger.info(f"混合检索完成: {len(results)} 条 (BM25:{len(bm25_hits)}, Dense:{len(all_dense_hits)})")
        return results
