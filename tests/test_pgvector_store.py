"""
pgvector 存储层单元测试。

验证:
  1. PgvectorStore 模块可导入
  2. PgvectorStoreRetriever 可导入
  3. 纯 PG 架构（FAISS 已移除）
"""
from __future__ import annotations


class TestImports:
    def test_import_pgvector_store(self):
        from src.knowledge.pgvector_store import PgvectorStore
        assert PgvectorStore is not None

    def test_import_retriever(self):
        from src.rag.retriever import PgvectorStoreRetriever
        assert PgvectorStoreRetriever is not None

    def test_no_faiss_left(self):
        """v0.6 纯 PG：FAISSRetriever / vector_store 模块应已移除"""
        import pytest
        with pytest.raises(ImportError):
            from src.rag.retriever import FAISSRetriever  # noqa: F401
        with pytest.raises(ImportError):
            import src.embedding.vector_store  # noqa: F401


class TestRetriever:
    def test_row_to_doc(self):
        from src.rag.retriever import PgvectorStoreRetriever
        row = {
            "content": "第一条 为了惩罚犯罪...",
            "score": 0.9521,
            "law_name": "中华人民共和国刑法",
            "chapter": "第一编 总则",
            "section": "",
            "article_range": "第一条",
            "chunk_type": "article",
        }
        doc = PgvectorStoreRetriever._row_to_doc(row)
        assert doc.content == "第一条 为了惩罚犯罪..."
        assert doc.score == 0.9521
        assert doc.law_name == "中华人民共和国刑法"
        assert doc.chapter == "第一编 总则"
        assert doc.article_range == "第一条"
        assert doc.citation == "中华人民共和国刑法 · 第一条"
