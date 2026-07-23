"""
FAISS → pgvector v2 数据迁移脚本。

将现有 FAISS 索引中的法律文档迁移到新的 document_chunks 表。
用法:
    # 先确保 PostgreSQL + Docker 运行中
    docker compose up -d postgres

    # 然后运行迁移
    uv run python scripts/migrate_faiss_to_pgvector.py

    # 迁移完成后，在 .env 中设置 PG_ENABLED=true 即可切换
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("migrate")

from src.config import INDEX_DIR, INDEX_NAME, PG_CONN
from src.embedding.factory import create_embedding_backend
from src.knowledge.pgvector_store import PgvectorStore


def main():
    # 1. 加载 FAISS 数据
    store_dir = INDEX_DIR / INDEX_NAME
    article_map_path = store_dir / "article_map.json"

    if not article_map_path.exists():
        logger.error(f"article_map.json 不存在: {article_map_path}")
        logger.error("请先运行 uv run python scripts/build_index.py build 构建 FAISS 索引")
        sys.exit(1)

    with open(article_map_path, "r", encoding="utf-8") as f:
        article_map = json.load(f)

    logger.info(f"从 article_map.json 加载 {len(article_map)} 条记录")

    # 2. 创建 embedder（用当前配置）
    embedder = create_embedding_backend()
    logger.info(f"Embedding 后端: {type(embedder).__name__} / {embedder.model}")

    # 3. 创建 pgvector store
    store = PgvectorStore(PG_CONN)
    store.ensure_tables()
    existing = store.get_chunk_count()
    if existing > 0:
        logger.warning(f"document_chunks 已有 {existing} 条数据，将清空后重新导入")
        import psycopg2
        conn = psycopg2.connect(PG_CONN)
        try:
            # 先删块（有外键依赖），再删文档
            with conn.cursor() as cur:
                cur.execute("DELETE FROM document_chunks")
                cur.execute("DELETE FROM documents")
            conn.commit()
            logger.info("已清空旧数据")
        except Exception as e:
            conn.rollback()
            logger.error(f"清空旧数据失败: {e}")
            raise
        finally:
            conn.close()

    # 4. 逐条迁移
    BATCH_SIZE = 32
    chunks = []
    total = len(article_map)

    for i, (key, info) in enumerate(article_map.items()):
        law_name = info.get("law_name", "")
        article = info.get("article", "")
        article_range = info.get("article_range", article)

        # 跳过章级摘要
        chunk_type = info.get("chunk_type", "article")
        if chunk_type == "chapter_summary":
            continue

        # 获取文档 ID
        doc_id = store.ensure_document(
            doc_type="law",
            title=law_name,
            source=info.get("source", ""),
        )

        content = info.get("content", "")
        if not content:
            continue

        chunks.append({
            "doc_id": doc_id,
            "chunk_type": chunk_type,
            "content": content,
            "metadata": {
                "law_name": law_name,
                "chapter": info.get("chapter", ""),
                "section": info.get("section", ""),
                "article_range": article_range,
                "chunk_type": chunk_type,
            },
        })

        # 达到批次大小，向量化 + 写入
        if len(chunks) >= BATCH_SIZE:
            texts = [c["content"] for c in chunks]
            embeddings = embedder.embed(texts)
            for c, emb in zip(chunks, embeddings):
                c["embedding"] = emb
            store.insert_chunks(chunks, embedding_model=embedder.model)
            logger.info(f"迁移进度: {min(i + 1, total)}/{total}")
            chunks = []

    # 处理剩余
    if chunks:
        texts = [c["content"] for c in chunks]
        embeddings = embedder.embed(texts)
        for c, emb in zip(chunks, embeddings):
            c["embedding"] = emb
        store.insert_chunks(chunks, embedding_model=embedder.model)

    # 5. 重建索引
    store.reindex()
    logger.info(f"✅ 迁移完成！共导入 {store.get_chunk_count()} 条")
    logger.info("现在在 .env 中设置 PG_ENABLED=true 即可使用 pgvector")


if __name__ == "__main__":
    main()
