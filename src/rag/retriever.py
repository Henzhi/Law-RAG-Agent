"""
检索器抽象接口与 FAISS 实现。

设计目标：将检索层抽成接口，后续切换 pgvector 只需新建一个实现类。
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from langchain_core.documents import Document

from src.config import RETRIEVAL_DROP_SUMMARY_CHUNKS, RETRIEVAL_SIM_THRESHOLD

logger = logging.getLogger(__name__)


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
        docs = [self._to_retrieved(doc, score) for doc, score in results]
        if RETRIEVAL_DROP_SUMMARY_CHUNKS:
            docs = [d for d in docs if d.chunk_type != "chapter_summary"]
        return self._apply_sim_threshold(docs, top_k)

    def search_by_law(self, query: str, law_name: str, top_k: int = 5) -> list[RetrievedDoc]:
        """在指定法律内检索"""
        results = self._store.search_with_score(query, k=top_k, filter_dict={"law_name": law_name})
        docs = [self._to_retrieved(doc, score) for doc, score in results]
        if RETRIEVAL_DROP_SUMMARY_CHUNKS:
            docs = [d for d in docs if d.chunk_type != "chapter_summary"]
        return self._apply_sim_threshold(docs, top_k)

    def _apply_sim_threshold(self, docs: list[RetrievedDoc], top_k: int) -> list[RetrievedDoc]:
        """按向量相似度阈值过滤召回结果。

        RETRIEVAL_SIM_THRESHOLD <= 0 时关闭（返回原结果）；
        过滤后为空则回退保留原结果，避免线上完全哑火。
        """
        if RETRIEVAL_SIM_THRESHOLD <= 0 or not docs:
            return docs
        filtered = [d for d in docs if d.score >= RETRIEVAL_SIM_THRESHOLD]
        if not filtered:
            logger.warning(
                f"[retriever] 向量相似度阈值 {RETRIEVAL_SIM_THRESHOLD} 过滤后无候选，"
                f"回退保留原 {len(docs)} 条结果以避免哑火"
            )
            return docs
        return filtered[:top_k]

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
# pgvector 检索器
# ---------------------------------------------------------------------------

class PgvectorRetriever(BaseRetriever):
    """基于 PostgreSQL + pgvector 的检索器

    用法:
        retriever = PgvectorRetriever(embedder, connection_string)
        retriever.build_from_documents(docs)  # 首次构建
        results = retriever.search(query, top_k=5)
    """

    def __init__(
        self,
        embedder,       # LawEmbedder 实例
        conn_string: str = "",
        table_name: str = "law_chunks",
    ):
        import psycopg2
        from pgvector.psycopg2 import register_vector

        self._embedder = embedder
        self._table = table_name
        self._conn_string = conn_string
        self._conn = psycopg2.connect(conn_string)
        register_vector(self._conn)
        self._create_table()

    def _ensure_connection(self):
        """检查连接是否存活，断开则自动重连"""
        try:
            with self._conn.cursor() as cur:
                cur.execute("SELECT 1")
        except Exception:
            import psycopg2
            from pgvector.psycopg2 import register_vector
            logger.warning("PG 连接已断开，尝试重连...")
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = psycopg2.connect(self._conn_string)
            register_vector(self._conn)
            logger.info("PG 重连成功")

    def _create_table(self):
        dim = self._embedder.get_embedding_dim()
        with self._conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding vector({dim}),
                    law_name TEXT DEFAULT '',
                    chapter TEXT DEFAULT '',
                    section TEXT DEFAULT '',
                    article_range TEXT DEFAULT '',
                    chunk_type TEXT DEFAULT ''
                )
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self._table}_embedding
                ON {self._table} USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)
        self._conn.commit()

    def build_from_documents(self, documents: list, batch_size: int = 32):
        """从 LangChain Document 列表构建 pgvector 索引"""
        total = len(documents)
        for i in range(0, total, batch_size):
            batch = documents[i:i + batch_size]
            texts = [d.page_content for d in batch]
            embeddings = self._embedder.embed_documents(texts)

            with self._conn.cursor() as cur:
                for doc, emb in zip(batch, embeddings):
                    meta = doc.metadata
                    cur.execute(
                        f"INSERT INTO {self._table} (content,embedding,law_name,chapter,section,article_range,chunk_type) "
                        f"VALUES (%s,%s,%s,%s,%s,%s,%s)",
                        (
                            doc.page_content,
                            emb,
                            meta.get("law_name", ""),
                            meta.get("chapter", ""),
                            meta.get("section", ""),
                            meta.get("article_range", ""),
                            meta.get("chunk_type", ""),
                        ),
                    )
            self._conn.commit()
            logger.info(f"pgvector 写入进度: {min(i + batch_size, total)}/{total}")

    def search(self, query: str, top_k: int = 5) -> list[RetrievedDoc]:
        self._ensure_connection()
        vec = self._embedder.embed_query(query)
        where = "WHERE chunk_type <> 'chapter_summary' " if RETRIEVAL_DROP_SUMMARY_CHUNKS else ""
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT content,law_name,chapter,section,article_range,chunk_type,"
                f"1 - (embedding <=> %s::vector) AS score "
                f"FROM {self._table} {where}"
                f"ORDER BY embedding <=> %s::vector LIMIT %s",
                (vec, vec, top_k),
            )
            rows = cur.fetchall()

        results = []
        for row in rows:
            content, law, ch, sec, article, ctype, score = row
            results.append(RetrievedDoc(
                content=content,
                score=round(float(score), 4),
                law_name=law or "",
                chapter=ch or "",
                section=sec or "",
                article_range=article or "",
                chunk_type=ctype or "",
            ))
        return results

    def search_by_law(self, query: str, law_name: str, top_k: int = 5) -> list[RetrievedDoc]:
        self._ensure_connection()
        vec = self._embedder.embed_query(query)
        where = "AND chunk_type <> 'chapter_summary' " if RETRIEVAL_DROP_SUMMARY_CHUNKS else ""
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT content,law_name,chapter,section,article_range,chunk_type,"
                f"1 - (embedding <=> %s::vector) AS score "
                f"FROM {self._table} WHERE law_name = %s {where}"
                f"ORDER BY embedding <=> %s::vector LIMIT %s",
                (vec, law_name, vec, top_k),
            )
            rows = cur.fetchall()

        results = []
        for row in rows:
            content, law, ch, sec, article, ctype, score = row
            results.append(RetrievedDoc(
                content=content,
                score=round(float(score), 4),
                law_name=law or "",
                chapter=ch or "",
                section=sec or "",
                article_range=article or "",
                chunk_type=ctype or "",
            ))
        return results

    def is_ready(self) -> bool:
        try:
            with self._conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self._table}")
                return cur.fetchone()[0] > 0
        except Exception:
            return False

    def close(self):
        self._conn.close()
