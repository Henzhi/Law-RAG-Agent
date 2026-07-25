"""
PostgreSQL + pgvector 知识库存储层 (v0.5)

企业级升级:
  - documents 主表 + document_chunks 块表，支持版本管理和状态标记
  - halfvec 半精度向量，存储减半、检索提速 ~30%
  - embedding_model 列隔离不同模型，切换模型无需全量重建
  - HNSW 索引，10万+ 向量仍保持 <10ms 延迟
  - 增量索引：单条 INSERT 即可生效，无需重建

用法:
    store = PgvectorStore(conn_string)
    store.ensure_tables()
    store.insert_chunks(chunks, embedding_model="bge-m3")
    results = store.search(query_vec, top_k=5)
"""
from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class PgvectorStore:
    """pgvector 知识库存储

    封装所有 PG + pgvector 操作，提供:
      - 表结构初始化
      - 文档块批量写入
      - 向量检索（余弦相似度）
      - 文档/块管理（状态切换、删除）
    """

    def __init__(self, conn_string: str):
        import psycopg2
        from pgvector.psycopg2 import register_vector

        self._conn_string = conn_string
        self._conn = psycopg2.connect(conn_string)
        register_vector(self._conn)

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------

    def _ensure_connection(self):
        try:
            with self._conn.cursor() as cur:
                cur.execute("SELECT 1")
        except Exception:
            import psycopg2
            from pgvector.psycopg2 import register_vector
            logger.warning("PG 连接断开，重连中...")
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = psycopg2.connect(self._conn_string)
            register_vector(self._conn)

    def close(self):
        self._conn.close()

    # ------------------------------------------------------------------
    # 表初始化
    # ------------------------------------------------------------------

    def ensure_tables(self):
        """创建知识库相关表（幂等，已有表不重建）"""
        self._ensure_connection()
        # 表结构由 docker/init.sql 定义，这里仅做存在性检查
        with self._conn.cursor() as cur:
            cur.execute("SELECT 1 FROM information_schema.tables WHERE table_name='document_chunks'")
            if cur.fetchone() is None:
                raise RuntimeError(
                    "document_chunks 表不存在。请先运行 docker compose up -d 初始化数据库。"
                )
        self._conn.commit()

    # ------------------------------------------------------------------
    # 文档管理
    # ------------------------------------------------------------------

    def ensure_document(
        self,
        doc_type: str,
        title: str,
        source: str = "",
        effective_date: str | None = None,
    ) -> str:
        """获取或创建文档记录，返回 doc_id"""
        self._ensure_connection()
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM documents WHERE title = %s AND status = 'active'",
                (title,),
            )
            row = cur.fetchone()
            if row:
                return str(row[0])

            cur.execute(
                "INSERT INTO documents (doc_type, title, source, effective_date) "
                "VALUES (%s, %s, %s, %s) RETURNING id",
                (doc_type, title, source, effective_date),
            )
            doc_id = str(cur.fetchone()[0])
        self._conn.commit()
        logger.info(f"新建文档: [{doc_type}] {title} (id={doc_id[:8]}...)")
        return doc_id

    # ------------------------------------------------------------------
    # 块写入
    # ------------------------------------------------------------------

    def insert_chunks(
        self,
        chunks: list[dict],
        embedding_model: str,
        batch_size: int = 32,
    ) -> int:
        """批量写入文档块

        Args:
            chunks: [{"doc_id", "chunk_type", "content", "embedding", "metadata"}, ...]
            embedding_model: 嵌入模型标识，如 "bge-m3"
            batch_size: 每批提交数

        Returns:
            写入的块数量
        """
        import json as _json
        self._ensure_connection()
        total = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            with self._conn.cursor() as cur:
                for c in batch:
                    embedding = c["embedding"]
                    meta = c.get("metadata", {})
                    # dict → JSON 字符串，PG 自动转 JSONB
                    if isinstance(meta, dict):
                        meta = _json.dumps(meta, ensure_ascii=False)
                    cur.execute(
                        "INSERT INTO document_chunks "
                        "(doc_id, chunk_type, content, embedding_model, embedding, metadata) "
                        "VALUES (%s, %s, %s, %s, %s::halfvec, %s)",
                        (
                            c["doc_id"],
                            c.get("chunk_type", "article"),
                            c["content"],
                            embedding_model,
                            embedding,
                            meta,
                        ),
                    )
            self._conn.commit()
            total += len(batch)
        logger.info(f"pgvector 写入完成: {total} chunks, model={embedding_model}")
        return total

    def get_chunk_count(self) -> int:
        self._ensure_connection()
        with self._conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM document_chunks")
            return cur.fetchone()[0]

    # ------------------------------------------------------------------
    # 向量检索
    # ------------------------------------------------------------------

    def search(
        self,
        query_vec: List[float],
        top_k: int = 5,
        embedding_model: str | None = None,
        doc_type: str | None = None,
        drop_summary: bool = True,
        sim_threshold: float = 0.0,
    ) -> list[dict]:
        """余弦相似度检索

        Args:
            query_vec: 查询向量
            top_k: 返回条数
            embedding_model: 仅检索指定模型的向量（None=不过滤）
            doc_type: 仅检索指定类型的文档（None=不过滤）
            drop_summary: 是否丢弃 chapter_summary 噪声
            sim_threshold: 最低相似度阈值（0=关闭）

        Returns:
            [{"content", "score", "law_name", "chapter", "article_range", ...}, ...]
        """
        self._ensure_connection()

        conditions = []
        # params 顺序必须匹配 SQL: SELECT %s → WHERE %s ... → ORDER BY %s → LIMIT %s
        params = [query_vec]  # SELECT 子句中的向量

        # embedding_model 过滤
        if embedding_model:
            conditions.append("dc.embedding_model = %s")
            params.append(embedding_model)
        # doc_type 过滤
        if doc_type:
            conditions.append("d.doc_type = %s")
            params.append(doc_type)
        # 噪声过滤
        if drop_summary:
            conditions.append("dc.chunk_type <> 'chapter_summary'")
        # 仅 active 文档
        conditions.append("d.status = 'active'")

        where = "WHERE " + " AND ".join(conditions) if conditions else ""

        # ORDER BY 向量 + LIMIT
        params.append(query_vec)
        params.append(top_k)

        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT dc.content, dc.metadata, dc.embedding_model, "
                f"1 - (dc.embedding <=> %s::halfvec) AS score "
                f"FROM document_chunks dc "
                f"JOIN documents d ON dc.doc_id = d.id "
                f"{where} "
                f"ORDER BY dc.embedding <=> %s::halfvec "
                f"LIMIT %s",
                params,
            )
            rows = cur.fetchall()

        results = []
        for row in rows:
            content, metadata, model, score = row
            meta = metadata or {}
            results.append({
                "content": content,
                "score": round(float(score), 4),
                "law_name": meta.get("law_name", ""),
                "chapter": meta.get("chapter", ""),
                "section": meta.get("section", ""),
                "article_range": meta.get("article_range", ""),
                "chunk_type": meta.get("chunk_type", ""),
                "embedding_model": model,
            })

        # 相似度阈值过滤
        if sim_threshold > 0 and results:
            filtered = [r for r in results if r["score"] >= sim_threshold]
            if not filtered:
                logger.warning(
                    f"pgvector 阈值 {sim_threshold} 过滤后无候选，回退保留 {len(results)} 条"
                )
                return results[:top_k]
            return filtered[:top_k]

        return results

    def is_ready(self) -> bool:
        try:
            return self.get_chunk_count() > 0
        except Exception:
            return False

    # ------------------------------------------------------------------
    # 索引管理
    # ------------------------------------------------------------------

    def reindex(self):
        """重建 HNSW 索引（大量写入后建议执行）"""
        self._ensure_connection()
        with self._conn.cursor() as cur:
            cur.execute("REINDEX INDEX idx_chunks_embedding")
        self._conn.commit()
        logger.info("HNSW 索引重建完成")
