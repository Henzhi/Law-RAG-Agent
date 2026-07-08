"""
检索器抽象接口与 FAISS 实现。

设计目标：将检索层抽成接口，后续切换 pgvector 只需新建一个实现类。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# 检索结果模型
# ---------------------------------------------------------------------------

@dataclass
class RetrievedDoc:
    """单条检索结果"""
    content: str
    score: float
    law_name: str = ""
    chapter: str = ""
    section: str = ""
    article_range: str = ""
    chunk_type: str = ""

    @property
    def citation(self) -> str:
        """生成引用标注，如 '治安管理处罚法 第十条'"""
        parts = [self.law_name]
        if self.article_range:
            parts.append(self.article_range)
        elif self.chapter:
            parts.append(self.chapter)
        return " · ".join(parts)


# ---------------------------------------------------------------------------
# 抽象检索器
# ---------------------------------------------------------------------------

class BaseRetriever(ABC):
    """检索器抽象基类"""

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[RetrievedDoc]:
        """语义检索，返回 top_k 个最相关文档"""
        ...

    @abstractmethod
    def is_ready(self) -> bool:
        """检索器是否已就绪（索引已加载）"""
        ...


# ---------------------------------------------------------------------------
# FAISS 检索器实现
# ---------------------------------------------------------------------------

class FAISSRetriever(BaseRetriever):
    """基于 FAISS 向量库的检索器"""

    def __init__(self, vector_store):
        """
        Args:
            vector_store: VectorStore 实例（已 build 或 load）
        """
        self._store = vector_store

    def search(self, query: str, top_k: int = 5) -> list[RetrievedDoc]:
        results = self._store.search_with_score(query, k=top_k)
        return [self._to_retrieved(doc, score) for doc, score in results]

    def search_by_law(self, query: str, law_name: str, top_k: int = 5) -> list[RetrievedDoc]:
        """在指定法律内检索"""
        results = self._store.search_with_score(query, k=top_k, filter_dict={"law_name": law_name})
        return [self._to_retrieved(doc, score) for doc, score in results]

    def is_ready(self) -> bool:
        return self._store.store is not None and self._store.doc_count > 0

    @staticmethod
    def _to_retrieved(doc: Document, score: float) -> RetrievedDoc:
        meta = doc.metadata
        return RetrievedDoc(
            content=doc.page_content,
            score=round(score, 4),
            law_name=meta.get("law_name", ""),
            chapter=meta.get("chapter", ""),
            section=meta.get("section", ""),
            article_range=meta.get("article_range", ""),
            chunk_type=meta.get("chunk_type", ""),
        )


# ---------------------------------------------------------------------------
# 占位：pgvector 检索器（后续实现）
# ---------------------------------------------------------------------------

class PgvectorRetriever(BaseRetriever):
    """pgvector 检索器（占位，后续迁移时实现）"""

    def search(self, query: str, top_k: int = 5) -> list[RetrievedDoc]:
        raise NotImplementedError("pgvector 检索器尚未实现")

    def is_ready(self) -> bool:
        return False
