"""
reranker.py — Cross-encoder 重排序
===================================
Phase 1: 轻量模拟 (Reranker) — 按来源权重 + 型号boost + 报警码boost
Phase 2: 真实模型 (BGEReranker) — BGE-Reranker-v2-m3 Cross-encoder

切换方式: RERANKER_BACKEND=bge 环境变量
默认 Phase 1 轻量模拟，无需下载模型。
"""

import re
from dataclasses import dataclass
from pathlib import Path

from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RerankResult:
    chunk_id: str
    content: str
    relevance_score: float
    original_rank: int
    source: str = ""


# ── Phase 1: 轻量模拟 ─────────────────────────

class Reranker:
    """Phase 1 轻量重排序: 按权重 + boost 模拟 Cross-encoder 行为。

    不需要下载模型，首次调用即可用。
    """

    def __init__(self):
        self._model_loaded = True

    async def rerank(self, query: str, candidates: list[dict], top_k: int = 10,
                     equipment_model: str | None = None) -> list[RerankResult]:
        if not candidates:
            return []

        query_upper = query.upper()
        source_weights = {"sop": 1.2, "verified_case": 1.0, "unverified_case": 0.7, "dense": 1.0, "bm25": 1.0}
        results: list[RerankResult] = []
        seen = set()

        for rank, cand in enumerate(candidates):
            content = cand.get("content", "")
            source = cand.get("source", "bm25")
            score = cand.get("score", 0.5)
            weight = source_weights.get(source, 1.0)

            if equipment_model and equipment_model.lower() in content.lower():
                weight *= 1.3

            alarm_codes = re.findall(r'[A-Z]{2,}-\w{3,}', query_upper)
            for code in alarm_codes:
                if code in content.upper():
                    weight *= 1.5
                    break

            final_score = min(score * weight, 1.0)
            fp = content[:100].strip()
            if fp in seen:
                continue
            seen.add(fp)
            results.append(RerankResult(chunk_id=cand.get("id", ""), content=content,
                                        relevance_score=final_score, original_rank=rank, source=source))

        results.sort(key=lambda r: (r.relevance_score, -r.original_rank), reverse=True)
        return results[:top_k]


# ── Phase 2: BGE-Reranker-v2-m3 ───────────────

class BGEReranker:
    """Phase 2 真实 Cross-encoder 重排序。

    加载 BGE-Reranker-v2-m3 (~1.1GB)，对 query-doc pair 做深度语义匹配。
    首次调用自动下载模型 (优先 ModelScope 缓存)。

    性能: 30 个候选对约 300-500ms (CPU)，GPU < 50ms。
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load(self):
        if self._model is not None:
            return

        # 优先 ModelScope 缓存
        import os
        model_path = str(Path.home() / ".cache" / "modelscope" / "models" / self.model_name.replace("/", "--") / "snapshots" / "master")
        if not Path(model_path).exists():
            model_path = self.model_name
            if "HF_ENDPOINT" not in os.environ:
                os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

        logger.info(f"加载 Re-ranker 模型: {model_path}")
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(model_path, device=self.device)
        logger.info("Re-ranker 模型加载完成")

    async def rerank(self, query: str, candidates: list[dict], top_k: int = 10,
                     equipment_model: str | None = None) -> list[RerankResult]:
        """Cross-encoder 精排。

        对每个 (query, candidate_content) pair 计算相关性分数，
        按分数降序返回 Top-K。

        示例:
            reranker = BGEReranker()
            results = await reranker.rerank(
                query="料筒温度异常",
                candidates=[{"id":"c1","content":"...","score":0.85,"source":"dense"}, ...],
                top_k=5,
            )
        """
        if not candidates:
            return []

        self._load()

        # 构建 query-doc pairs
        pairs = [(query, cand.get("content", "")) for cand in candidates]

        # 批量推理 (同步, 用 run_in_executor 异步化)
        import asyncio
        loop = asyncio.get_running_loop()
        scores = await loop.run_in_executor(None, lambda: self._model.predict(pairs).tolist())

        # 构建结果
        results = []
        for i, cand in enumerate(candidates):
            score = float(scores[i])
            # Cross-encoder 输出是原始分数，做 sigmoid 映射到 [0, 1]
            import math
            normalized = 1.0 / (1.0 + math.exp(-score))
            results.append(RerankResult(
                chunk_id=cand.get("id", ""),
                content=cand.get("content", ""),
                relevance_score=normalized,
                original_rank=i,
                source=cand.get("source", ""),
            ))

        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results[:top_k]


# ── 工厂函数 ──────────────────────────────────

_reranker: Reranker | BGEReranker | None = None


def get_reranker(backend: str = "auto") -> Reranker | BGEReranker:
    """获取重排序器实例。

    参数:
        backend: "auto" (读环境变量) | "light" | "bge"

    环境变量 RERANKER_BACKEND=bge 切换到真实模型。
    """
    global _reranker
    if _reranker is not None:
        return _reranker

    if backend == "auto":
        import os
        backend = os.environ.get("RERANKER_BACKEND", "light")

    if backend == "bge":
        try:
            _reranker = BGEReranker()
            logger.info("Reranker backend: BGE-Reranker-v2-m3")
        except Exception as e:
            logger.warning(f"BGE Reranker 加载失败，降级到轻量版: {e}")
            _reranker = Reranker()
    else:
        _reranker = Reranker()
        logger.info("Reranker backend: 轻量模拟")

    return _reranker
