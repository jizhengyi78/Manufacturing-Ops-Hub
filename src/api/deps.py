"""
deps.py — FastAPI 依赖注入
==========================
管理全局单例的生命周期，通过 FastAPI Depends() 注入到路由处理器。

包含:
- get_graph(): 获取编译后的 LangGraph 工作流
- get_hybrid_retriever(): 获取混合检索器
- get_session_memory(): 获取会话记忆

为什么用依赖注入:
- 全局单例在 app lifespan 中初始化，路由中通过 Depends 获取
- 方便测试时替换为 mock 对象
- 避免在路由中直接 import 全局变量
"""

from functools import lru_cache

from src.graph.builder import build_graph
from src.graph.checkpoint import get_checkpoint_manager
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever, get_dense_retriever
from src.retrieval.reranker import Reranker, get_reranker
from src.memory.session import get_session_memory
from src.memory.compressor import get_compressor


@lru_cache
def get_hybrid_retriever() -> HybridRetriever:
    """获取混合检索器 (全局单例)。"""
    bm25 = BM25Retriever()
    dense = get_dense_retriever()  # 自动选择 memory/milvus
    reranker = get_reranker()  # auto: 环境变量控制 light/bge
    return HybridRetriever(bm25, dense, reranker)


@lru_cache
def get_compiled_graph():
    """获取编译后的 LangGraph 工作流 (全局单例)。"""
    hybrid = get_hybrid_retriever()
    session_mem = get_session_memory()
    compressor = get_compressor()

    graph = build_graph(
        hybrid_retriever=hybrid,
        session_memory=session_mem,
        compressor=compressor,
    )
    return graph.compile()
