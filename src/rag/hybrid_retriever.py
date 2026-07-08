"""
混合检索器：向量语义检索 + BM25 关键词检索。

设计原理：
  - 向量检索擅长语义相似，但法律条文编号（如"第十条"）难以精确命中
  - BM25 擅长精确关键词匹配，能补上条文编号、法律名称等特征
  - 两者加权融合，通过 .env 中的 RETRIEVAL_BM25_WEIGHT 调权重
"""
from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Optional

import jieba
from rank_bm25 import BM25Okapi

from src.config import RETRIEVAL_BM25_WEIGHT, RETRIEVAL_HYBRID_ENABLED
from .retriever import BaseRetriever, RetrievedDoc, FAISSRetriever


# ---------------------------------------------------------------------------
# 混合检索器
# ---------------------------------------------------------------------------

class HybridRetriever(BaseRetriever):
    """向量 + BM25 混合检索器

    用法:
        hr = HybridRetriever(faiss_retriever, documents)
        results = hr.search("治安管理处罚法第十条", top_k=5)

    通过 .env 控制:
        RETRIEVAL_HYBRID_ENABLED=true  → 混合检索
        RETRIEVAL_BM25_WEIGHT=0.3      → BM25 权重 30%, 向量 70%
    """

    def __init__(
        self,
        vector_retriever: FAISSRetriever,
        corpus_texts: list[str],
        corpus_docs: list[RetrievedDoc] | None = None,
        bm25_weight: float | None = None,
    ):
        """
        Args:
            vector_retriever: FAISS 向量检索器
            corpus_texts: BM25 语料（与 FAISS 索引中的文档一一对应）
            corpus_docs: 对应的 RetrievedDoc 列表（用于结果映射）
            bm25_weight: BM25 权重，0~1；None 则从 config 读取
        """
        self._vector = vector_retriever
        self._corpus_texts = corpus_texts
        self._corpus_docs = corpus_docs or []
        self._bm25_weight = bm25_weight if bm25_weight is not None else RETRIEVAL_BM25_WEIGHT

        # 分词 + 构建 BM25 索引
        self._tokenized = [list(jieba.cut(t)) for t in corpus_texts]
        self._bm25 = BM25Okapi(self._tokenized)

    def search(self, query: str, top_k: int = 5) -> list[RetrievedDoc]:
        """混合检索，按加权分数排序返回"""
        if not RETRIEVAL_HYBRID_ENABLED:
            return self._vector.search(query, top_k)

        # 1. 向量检索（取 top_k * 3 作为候选池给融合用）
        vec_results = self._vector.search(query, top_k * 3)
        # 2. BM25 检索
        bm25_results = self._bm25_search(query, top_k * 3)
        # 3. 加权融合
        merged = self._merge(vec_results, bm25_results, top_k)
        return merged

    def is_ready(self) -> bool:
        return self._vector.is_ready()

    # ------------------------------------------------------------------
    # BM25 检索
    # ------------------------------------------------------------------

    def _bm25_search(self, query: str, top_k: int) -> list[RetrievedDoc]:
        """BM25 关键词检索"""
        tokenized_query = list(jieba.cut(query))
        scores = self._bm25.get_scores(tokenized_query)

        # 取 top_k 个最高分
        indexed = [(i, score) for i, score in enumerate(scores) if score > 0]
        indexed.sort(key=lambda x: x[1], reverse=True)
        top = indexed[:top_k]

        results = []
        max_score = top[0][1] if top else 1.0
        for idx, score in top:
            if idx < len(self._corpus_docs):
                doc = self._corpus_docs[idx]
                # 归一化到 0~1
                norm_score = score / max_score
                results.append(RetrievedDoc(
                    content=doc.content,
                    score=round(norm_score, 4),
                    law_name=doc.law_name,
                    chapter=doc.chapter,
                    section=doc.section,
                    article_range=doc.article_range,
                    chunk_type=doc.chunk_type,
                ))
            elif idx < len(self._corpus_texts):
                # fallback：无预存 doc 时直接用文本
                results.append(RetrievedDoc(
                    content=self._corpus_texts[idx],
                    score=round(score / max_score, 4),
                ))
        return results

    # ------------------------------------------------------------------
    # 加权融合
    # ------------------------------------------------------------------

    def _merge(
        self,
        vec_docs: list[RetrievedDoc],
        bm25_docs: list[RetrievedDoc],
        top_k: int,
    ) -> list[RetrievedDoc]:
        """加权融合两路结果，去重后排序"""
        bm25_w = self._bm25_weight
        vec_w = 1.0 - bm25_w

        # 归一化向量分数到 0~1
        vec_max = max((d.score for d in vec_docs), default=1.0)
        bm25_max = max((d.score for d in bm25_docs), default=1.0)

        # 合并：key = (law_name, article_range)，取最高分
        merged: dict[tuple, tuple[RetrievedDoc, float]] = {}

        for doc in vec_docs:
            key = (doc.law_name, doc.article_range)
            norm_score = doc.score / vec_max if vec_max > 0 else 0
            merged[key] = (doc, norm_score * vec_w)

        for doc in bm25_docs:
            key = (doc.law_name, doc.article_range)
            norm_score = doc.score / bm25_max if bm25_max > 0 else 0
            bm25_contrib = norm_score * bm25_w
            if key in merged:
                existing_doc, existing_score = merged[key]
                merged[key] = (existing_doc, existing_score + bm25_contrib)
            else:
                merged[key] = (doc, bm25_contrib)

        # 按融合分数降序
        sorted_items = sorted(merged.values(), key=lambda x: x[1], reverse=True)
        results = []
        for doc, fused_score in sorted_items[:top_k]:
            doc.score = round(fused_score, 4)
            results.append(doc)

        return results

    # ------------------------------------------------------------------
    # 持久化 BM25 语料
    # ------------------------------------------------------------------

    def save_corpus(self, path: Path) -> None:
        """保存 BM25 语料到磁盘（加速下次加载）"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "texts": self._corpus_texts,
                "tokenized": self._tokenized,
            }, f)

    @classmethod
    def from_corpus_file(
        cls,
        vector_retriever: FAISSRetriever,
        corpus_docs: list[RetrievedDoc],
        corpus_path: Path,
        bm25_weight: float | None = None,
    ) -> "HybridRetriever":
        """从磁盘加载 BM25 语料"""
        with open(corpus_path, "rb") as f:
            data = pickle.load(f)
        instance = cls.__new__(cls)
        instance._vector = vector_retriever
        instance._corpus_texts = data["texts"]
        instance._corpus_docs = corpus_docs
        instance._bm25_weight = bm25_weight if bm25_weight is not None else RETRIEVAL_BM25_WEIGHT
        instance._tokenized = data["tokenized"]
        instance._bm25 = BM25Okapi(instance._tokenized)
        return instance


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------

def create_retriever(
    vector_store,
    documents: list | None = None,
) -> BaseRetriever:
    """根据 RETRIEVAL_HYBRID_ENABLED 配置创建对应的检索器

    Args:
        vector_store: VectorStore 实例
        documents: LangChain Document 列表（混合检索时需要，用于构建 BM25 语料）

    Returns:
        FAISSRetriever 或 HybridRetriever
    """
    faiss = FAISSRetriever(vector_store)

    if not RETRIEVAL_HYBRID_ENABLED:
        return faiss

    # 混合检索：需要文档文本构建 BM25
    if documents is None:
        # 从 FAISS 中提取文本
        raise ValueError(
            "混合检索需要 documents 参数来构建 BM25 语料。"
            "请在 build_index 时传入文档列表。"
        )

    # 构建 BM25 语料和预存 doc 列表
    texts = [doc.page_content for doc in documents]
    ret_docs = [FAISSRetriever._to_retrieved(doc, 0.0) for doc in documents]

    # 尝试从缓存加载
    corpus_path = Path(vector_store.store_dir) / "bm25_corpus.pkl"
    if corpus_path.exists():
        print(f"从缓存加载 BM25 语料: {corpus_path}")
        return HybridRetriever.from_corpus_file(
            vector_retriever=faiss,
            corpus_docs=ret_docs,
            corpus_path=corpus_path,
        )

    print("构建 BM25 语料 ...")
    hr = HybridRetriever(
        vector_retriever=faiss,
        corpus_texts=texts,
        corpus_docs=ret_docs,
    )
    hr.save_corpus(corpus_path)
    print(f"BM25 语料已缓存: {corpus_path}")
    return hr
