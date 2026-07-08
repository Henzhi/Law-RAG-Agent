"""
Reranker 二次精排模块。

流程: 粗排(top_k*N) → Reranker 精排 → Top_K

使用 BAAI/bge-reranker-v2-m3, 多语言 Cross-Encoder,
对中文法律文本重排序效果显著。
"""
from __future__ import annotations

import logging
from typing import Optional

from FlagEmbedding import FlagReranker

from .retriever import RetrievedDoc, BaseRetriever

logger = logging.getLogger(__name__)

# 默认 reranker 模型
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


class Reranker:
    """Cross-Encoder 精排器

    用法:
        reranker = Reranker()
        top5 = reranker.rerank(query, top20_docs, top_k=5)
    """

    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL, use_fp16: bool = False):
        """
        Args:
            model_name: HuggingFace 模型名
            use_fp16: 是否使用半精度（CPU 下建议关闭）
        """
        self.model_name = model_name
        logger.info(f"加载 Reranker 模型: {model_name} ...")
        self._model = FlagReranker(model_name, use_fp16=use_fp16)
        logger.info("Reranker 加载完成")

    def rerank(
        self,
        query: str,
        docs: list[RetrievedDoc],
        top_k: int = 5,
    ) -> list[RetrievedDoc]:
        """对候选文档精排，返回 top_k

        Args:
            query: 用户查询
            docs: 候选文档列表
            top_k: 返回条数

        Returns:
            重排序后的 top_k 文档（score 更新为 reranker 分数）
        """
        if len(docs) <= top_k:
            return docs

        # 构建 (query, doc_content) 对
        pairs = [[query, doc.content] for doc in docs]

        # 批量打分
        scores = self._model.compute_score(pairs, normalize=True)

        # 按分数降序排序
        scored = list(zip(docs, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        # 取 top_k，更新分数
        result = []
        for doc, score in scored[:top_k]:
            doc.score = round(float(score), 4)
            result.append(doc)
        return result

    def compute_scores(self, query: str, docs: list[RetrievedDoc]) -> list[float]:
        """对所有文档打分（不排序）"""
        pairs = [[query, doc.content] for doc in docs]
        return [float(s) for s in self._model.compute_score(pairs, normalize=True)]


class RerankRetriever(BaseRetriever):
    """带 Reranker 的检索器装饰器

    用法:
        base = FAISSRetriever(store)
        retriever = RerankRetriever(base, reranker, recall_k=20, top_k=5)
        results = retriever.search(query)
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        reranker: Reranker,
        recall_k: int = 20,
        top_k: int = 5,
    ):
        """
        Args:
            base_retriever: 基础检索器（粗排）
            reranker: 精排器
            recall_k: 粗排召回数
            top_k: 最终返回数
        """
        self._base = base_retriever
        self._reranker = reranker
        self._recall_k = recall_k
        self._top_k = top_k

    def search(self, query: str, top_k: int = 5) -> list[RetrievedDoc]:
        """粗排 → 精排 → 返回"""
        # 粗排：多召回一些
        candidates = self._base.search(query, top_k=self._recall_k)
        # 精排
        return self._reranker.rerank(query, candidates, top_k=top_k or self._top_k)

    def is_ready(self) -> bool:
        return self._base.is_ready()
