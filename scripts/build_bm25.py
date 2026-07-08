"""单独构建 BM25 语料（不重建 FAISS 索引）"""
import sys, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import INDEX_DIR, INDEX_NAME
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.retriever import FAISSRetriever
from src.embedding.embedder import LawEmbedder
from src.embedding.vector_store import VectorStore
from src.config import EMBED_MODEL, EMBED_BASE_URL

store_dir = INDEX_DIR / INDEX_NAME
pkl_path = store_dir / "index.pkl"
corpus_path = store_dir / "bm25_corpus.pkl"

print(f"加载 FAISS 索引: {store_dir}")
embedder = LawEmbedder(model=EMBED_MODEL, base_url=EMBED_BASE_URL)
store = VectorStore(embedder=embedder, persist_dir=INDEX_DIR, index_name=INDEX_NAME)
store.load()

# 从 FAISS 的 docstore 提取文本
faiss_index = store.store
texts = []
ret_docs = []

# FAISS 的 docstore 是一个 dict: {id: Document}
docstore = faiss_index.docstore._dict if hasattr(faiss_index.docstore, '_dict') else {}
for doc_id, doc in docstore.items():
    texts.append(doc.page_content)
    ret_docs.append(FAISSRetriever._to_retrieved(doc, 0.0))

print(f"提取 {len(texts)} 条文档文本")
print("构建 BM25 索引 ...")

# 创建临时 FAISS retriever
faiss = FAISSRetriever(store)
hr = HybridRetriever(
    vector_retriever=faiss,
    corpus_texts=texts,
    corpus_docs=ret_docs,
)
hr.save_corpus(corpus_path)
print(f"BM25 语料已保存: {corpus_path} ({corpus_path.stat().st_size / 1024:.0f} KB)")
