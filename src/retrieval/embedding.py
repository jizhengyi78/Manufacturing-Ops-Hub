"""
embedding.py — 文本嵌入服务
===========================
角色：将文本转换成向量，是混合检索和长期记忆的基础。
      所有需要做向量化的模块都通过这个文件调用，不直接调用 sentence-transformers。

为什么用 BGE-large-zh-v1.5:
- 中文语义理解 SOTA 级别
- 1024 维向量，权衡了精度和存储成本
- 制造业专业术语 (如 "注塑缩水"、"刀纹异常") 的语义表示效果好

使用方式:
    from src.retrieval.embedding import EmbeddingService, get_embedding_service

    service = get_embedding_service()  # 单例，避免重复加载模型

    # 单条文本
    vec = await service.embed("注塑机料筒温度异常")
    # vec 是 list[float], 长度 1024

    # 批量 (性能更好)
    vecs = await service.embed_batch([
        "注塑机怎么换模具",
        "冲压线安全操作规程",
        "CNC 加工精度异常怎么排查"
    ])
    # vecs 是 list[list[float]], 每个长度 1024

    # 查询专用 (BGE 模型对查询有特殊前缀)
    query_vec = await service.embed_query("料筒温度异常怎么处理")

查询 vs 文档的区别:
- 查询: 前面加 "为这个句子生成表示以用于检索相关文章：" prefix (BGE 官方推荐)
- 文档: 不加 prefix
- 这样做能让查询向量和文档向量在同一向量空间中更精准匹配

注意事项:
- 模型首次加载需要下载 (~1.3GB)，建议在应用启动时预热
- GPU 推理 (embedding_device="cuda") 比 CPU 快 10-50 倍
- 同一文本两次调用结果完全一致 (已归一化)
- 不要在循环里单条调用 embed()，用 embed_batch() 批量处理
- 向量维度 1024 和 Milvus Collection 定义必须一致
"""

import asyncio
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from src.core.config import get_settings
from src.core.logging import get_logger

logger = get_logger(__name__)

# BGE 查询前缀: 官方推荐用于查询编码的 prefix
BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


class EmbeddingService:
    """文本嵌入服务 — 封装 sentence-transformers，提供同步/异步接口。"""

    def __init__(self, model_name: str, device: str = "cpu"):
        """
        Args:
            model_name: HuggingFace 模型名，如 "BAAI/bge-large-zh-v1.5"
            device: "cpu" 或 "cuda" 或 "cuda:0"

        示例:
            service = EmbeddingService("BAAI/bge-large-zh-v1.5", "cpu")
        """
        logger.info(f"加载嵌入模型: {model_name} on {device}")

        # 优先从 ModelScope 缓存加载 (国内快), 找不到再从 HuggingFace 下载
        import os
        from pathlib import Path
        modelscope_path = Path.home() / ".cache" / "modelscope" / "models" / model_name.replace("/", "--")
        if modelscope_path.exists():
            model_path = str(modelscope_path / "snapshots" / "master")
            logger.info(f"从 ModelScope 缓存加载: {model_path}")
            self.model = SentenceTransformer(model_path, device=device)
        else:
            if "HF_ENDPOINT" not in os.environ:
                os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            self.model = SentenceTransformer(model_name, device=device)
        self._dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"嵌入模型加载完成, 维度: {self._dim}")

    @property
    def dim(self) -> int:
        """向量维度。"""
        return self._dim

    def _encode(self, texts: list[str]) -> list[list[float]]:
        """同步编码，返回归一化向量列表。"""
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,  # L2 归一化, 配合 Milvus COSINE 度量
            show_progress_bar=False,
        )
        return embeddings.tolist()

    async def embed(self, text: str) -> list[float]:
        """单条文本编码 (异步)。

        示例:
            vec = await service.embed("注塑机料筒温度异常")
            # vec = [0.012, -0.034, 0.056, ...]  # 1024 维
        """
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, self._encode, [text])
        return results[0]

    async def embed_query(self, query: str) -> list[float]:
        """查询文本编码 — 自动加 BGE 查询 prefix。

        为什么查询要加 prefix:
        BGE 模型训练时，查询样本都加了 "为这个句子生成表示以用于检索相关文章：" prefix，
        推理时保持一致性能最好。文档不需要加。

        示例:
            vec = await service.embed_query("料筒温度异常怎么处理")
        """
        return await self.embed(BGE_QUERY_PREFIX + query)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量编码 — 比逐条调用快 5-20 倍。

        示例:
            vecs = await service.embed_batch(["文本1", "文本2", "文本3"])
            # vecs = [[0.1, 0.2, ...], [0.3, 0.4, ...], [0.5, 0.6, ...]]
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._encode, texts)

    async def embed_batch_queries(self, queries: list[str]) -> list[list[float]]:
        """批量查询编码 — 自动加 prefix。"""
        return await self.embed_batch([BGE_QUERY_PREFIX + q for q in queries])


@lru_cache
def get_embedding_service() -> EmbeddingService:
    """获取嵌入服务单例。整个应用生命周期内只加载一次模型 (避免重复加载 ~1.3GB)。

    示例:
        service = get_embedding_service()
    """
    settings = get_settings()
    return EmbeddingService(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
    )
