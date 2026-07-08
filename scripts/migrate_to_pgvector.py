"""FAISS → pgvector 数据迁移脚本

用法:
    uv run python scripts/migrate_to_pgvector.py

前提: docker-compose up db 已启动 postgres+pgvector
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import EMBED_MODEL, EMBED_BASE_URL, EMBED_BATCH_SIZE
from src.embedding.embedder import LawEmbedder
from src.embedding.vector_store import VectorStore
from src.rag.retriever import FAISSRetriever, PgvectorRetriever

# FAISS 索引路径
INDEX_DIR = Path("data/vector_store")
INDEX_NAME = "law_index_bge"

# pg 连接
PG_CONN = "postgresql://lawrag:lawrag123@localhost:5432/lawrag"

print("加载 FAISS 索引 ...")
embedder = LawEmbedder(model=EMBED_MODEL, base_url=EMBED_BASE_URL, batch_size=EMBED_BATCH_SIZE)
store = VectorStore(embedder=embedder, persist_dir=INDEX_DIR, index_name=INDEX_NAME)
store.load()

# 从 FAISS docstore 提取所有文档
docstore_dict = store.store.docstore._dict
documents = list(docstore_dict.values())
print(f"提取 {len(documents)} 条文档")

# 写入 pgvector
print("写入 pgvector ...")
pg = PgvectorRetriever(embedder=embedder, conn_string=PG_CONN)
pg.build_from_documents(documents, batch_size=EMBED_BATCH_SIZE)
print("迁移完成!")

# 验证
pg_count = 0
with pg._conn.cursor() as cur:
    cur.execute(f"SELECT COUNT(*) FROM {pg._table}")
    pg_count = cur.fetchone()[0]
pg.close()

print(f"FAISS: {len(documents)} | pgvector: {pg_count}")
assert len(documents) == pg_count, "数量不匹配!"
print("验证通过 ✓")
