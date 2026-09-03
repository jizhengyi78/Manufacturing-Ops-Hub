"""
bm25.py — Elasticsearch BM25 关键词检索
=======================================
角色：提供基于 BM25 算法的精确关键词匹配能力。

为什么制造场景需要 BM25:
- "海天MA1200 料筒温度" → BM25 对型号、参数名做精确匹配，比向量检索更准
- "HT-E-0021 报警码" → 精确匹配报警码，向量检索可能找错相近含义但不相关的文档
- 中文分词器 (ik_smart) 能正确切分 "注塑机"/"模具"/"料筒" 等制造术语

BM25 vs Dense 检索:
- BM25: 适合精确型号、编号、报警码、专业术语
- Dense: 适合模糊语义如"打出来的件有毛边"→ 匹配到"飞边缺陷SOP"
- 混合: 两者融合 (RRF) 取长补短

ES 索引设计 (manufacturing_knowledge):
- content: text (ik_smart 分词) → BM25 主检索字段
- equipment_model: text + keyword 子字段 → 精确 + 模糊匹配
- fault_code: keyword → 报警码精确匹配
- field weights: fault_code(3.0) > equipment_model(2.0) > title(1.5) > content(1.0)

使用示例:
    from src.retrieval.bm25 import BM25Retriever
    retriever = BM25Retriever(es_client)
    results = await retriever.search("海天MA1200 料筒温度异常", top_k=10)
    for r in results:
        print(f"Score: {r['score']}, Content: {r['content'][:50]}")

注意事项:
- Phase 1: 使用内存 rank_bm25 做轻量实现 (不需要 ES 服务)
- Phase 2: 切到 Elasticsearch，复用同样的接口
- 中文分词: ES 需要安装 ik_smart 插件，内存版无分词直接字符匹配
- 报警码必须用 exact match (keyword)，不能用 like '%xxx%' 模糊查
"""

from dataclasses import dataclass
from typing import Optional

from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BM25Hit:
    """BM25 检索命中结果。"""
    chunk_id: str
    content: str
    score: float
    doc_title: str = ""
    equipment_model: str = ""
    fault_code: str = ""


class BM25Retriever:
    """BM25 关键词检索器 (Phase 1: 内存 rank_bm25 实现)。

    设计: 接口抽象，Phase 2 替换为 ES 实现不改调用方代码。

    示例:
        retriever = BM25Retriever()
        await retriever.index([
            {"id": "c1", "content": "注塑机料筒温度异常处理", "title": "注塑机常见故障"},
            {"id": "c2", "content": "冲压模具保养步骤", "title": "模具保养SOP"},
        ])
        results = await retriever.search("料筒温度异常", top_k=5)
        # results[0].chunk_id == "c1"
    """

    def __init__(self):
        self._corpus: list[dict] = []          # 文档语料库
        self._id_to_idx: dict[str, int] = {}   # chunk_id → 索引位置
        self._bm25 = None                      # rank_bm25 实例 (惰性构建)
        self._tokenized_corpus: list[list[str]] = []
        self._dirty = True                     # 语料变更标记

    async def index(self, documents: list[dict]) -> None:
        """批量索引文档。

        参数:
            documents: 每个 dict 需要:
                - id: 唯一 chunk_id
                - content: 检索文本
                - title (可选): 文档标题
                - equipment_model (可选): 设备型号
                - fault_code (可选): 报警码
        """
        for doc in documents:
            chunk_id = doc["id"]
            if chunk_id in self._id_to_idx:
                # 更新已有文档
                idx = self._id_to_idx[chunk_id]
                self._corpus[idx] = doc
            else:
                self._id_to_idx[chunk_id] = len(self._corpus)
                self._corpus.append(doc)
        self._dirty = True
        logger.info(f"BM25 索引: {len(documents)} 条, 总计 {len(self._corpus)} 条")

    def _simple_tokenize(self, text: str) -> list[str]:
        """轻量中文分词: 按字/词切分 (Phase 1 简化版, 不需要 jieba)。

        示例:
            >>> retriever._simple_tokenize("注塑机料筒温度异常")
            # 返回按二元组分词的结果，适配中文
        """
        # 简单二元组分词 (bigram) + 单个字符
        tokens = []
        cleaned = text.strip().lower()
        # 按二元组分
        for i in range(len(cleaned) - 1):
            pair = cleaned[i:i+2]
            # 跳过纯空白/标点组合
            if not pair.isspace() and not all(c in ' \t\n\r,.;;:：，。；：！？""''（）【】《》' for c in pair):
                tokens.append(pair)
        # 加上单字符
        for ch in cleaned:
            if not ch.isspace() and ch not in ' \t\n\r,.;;:：，。；：！？""''（）【】《》':
                tokens.append(ch)
        return tokens

    def _build_index(self) -> None:
        """构建 BM25 索引。"""
        if not self._dirty:
            return

        from rank_bm25 import BM25Okapi
        self._tokenized_corpus = [
            self._simple_tokenize(doc.get("content", ""))
            for doc in self._corpus
        ]
        # 过滤空文档，避免 BM25Okapi 内部除零错误
        non_empty = [(i, tokens) for i, tokens in enumerate(self._tokenized_corpus) if tokens]
        if non_empty:
            indices, filtered = zip(*non_empty)
            self._corpus = [self._corpus[i] for i in indices]
            self._id_to_idx = {d["id"]: i for i, d in enumerate(self._corpus)}
            self._tokenized_corpus = list(filtered)
            self._bm25 = BM25Okapi(self._tokenized_corpus)
        else:
            self._bm25 = None
        self._dirty = False
        logger.info(f"BM25 索引构建完成: {len(self._corpus)} 条")

    async def search(
        self,
        query: str,
        top_k: int = 20,
        equipment_model: str | None = None,
        fault_code: str | None = None,
    ) -> list[BM25Hit]:
        """BM25 检索。

        参数:
            query: 查询文本
            top_k: 返回结果数
            equipment_model: 可选，设备型号过滤 (精确匹配)
            fault_code: 可选，报警码过滤 (精确匹配)

        返回: 按 BM25 分数降序排列的结果

        示例:
            results = await retriever.search("料筒温度异常", top_k=5)
            for r in results:
                print(f"[{r.score:.3f}] {r.chunk_id}: {r.content[:60]}")
        """
        self._build_index()
        if not self._corpus or self._bm25 is None:
            return []

        tokens = self._simple_tokenize(query)
        scores = self._bm25.get_scores(tokens)

        hits = []
        for i, score in enumerate(scores):
            if score <= 0:
                continue
            doc = self._corpus[i]

            # 型号/报警码过滤 (精确匹配)
            if equipment_model and doc.get("equipment_model"):
                if equipment_model.lower() not in doc["equipment_model"].lower():
                    continue
            if fault_code and doc.get("fault_code"):
                if fault_code.lower() != doc["fault_code"].lower():
                    continue

            hits.append(BM25Hit(
                chunk_id=doc["id"],
                content=doc.get("content", ""),
                score=float(score),
                doc_title=doc.get("title", ""),
                equipment_model=doc.get("equipment_model", ""),
                fault_code=doc.get("fault_code", ""),
            ))

        # 按分数降序
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    async def delete(self, chunk_ids: list[str]) -> int:
        """删除指定 chunk。"""
        removed = 0
        for cid in chunk_ids:
            if cid in self._id_to_idx:
                idx = self._id_to_idx.pop(cid)
                self._corpus[idx] = None  # 标记删除
                removed += 1
        self._corpus = [d for d in self._corpus if d is not None]
        # 重建 id 映射
        self._id_to_idx = {d["id"]: i for i, d in enumerate(self._corpus)}
        self._dirty = True
        return removed

    @property
    def doc_count(self) -> int:
        return len(self._corpus)
