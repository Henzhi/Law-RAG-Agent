"""
API 依赖注入。

管理 LLM、向量库、RAG 引擎等单例，所有可配参数从 src.config 读取。
"""
from __future__ import annotations

import logging

from src.config import (
    LLM_MODEL, LLM_BASE_URL, LLM_TEMPERATURE, LLM_TOP_P, LLM_MAX_TOKENS,
    EMBED_MODEL, EMBED_BASE_URL, EMBED_BATCH_SIZE,
    RETRIEVAL_TOP_K, RETRIEVAL_HYBRID_ENABLED,
    INDEX_NAME, INDEX_DIR,
)
from src.embedding.embedder import LawEmbedder
from src.embedding.vector_store import VectorStore
from src.llm.client import LawLLM, LLMConfig
from src.rag.engine import RAGEngine
from src.rag.retriever import FAISSRetriever
from src.rag.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)

_engine: RAGEngine | None = None
_llm: LawLLM | None = None


def get_llm() -> LawLLM:
    """获取 LLM 单例"""
    global _llm
    if _llm is None:
        logger.info(f"初始化 LLM: {LLM_MODEL} @ {LLM_BASE_URL}")
        _llm = LawLLM(
            model=LLM_MODEL,
            base_url=LLM_BASE_URL,
            config=LLMConfig(
                temperature=LLM_TEMPERATURE,
                top_p=LLM_TOP_P,
                num_predict=LLM_MAX_TOKENS,
            ),
        )
    return _llm


def get_engine() -> RAGEngine:
    """获取 RAG 引擎单例"""
    global _engine
    if _engine is None:
        llm = get_llm()

        logger.info(f"加载向量库: {INDEX_DIR / INDEX_NAME} (模型: {EMBED_MODEL})")

        embedder = LawEmbedder(
            model=EMBED_MODEL,
            base_url=EMBED_BASE_URL,
            batch_size=EMBED_BATCH_SIZE,
        )
        store = VectorStore(
            embedder=embedder,
            persist_dir=INDEX_DIR,
            index_name=INDEX_NAME,
        )
        if store.load() is None:
            raise RuntimeError(
                f"FAISS 索引不存在: {INDEX_DIR / INDEX_NAME}\n"
                f"请先运行: uv run python scripts/build_index.py build"
            )

        # 构建检索器（纯向量 / 混合检索，由 .env 控制）
        from pathlib import Path
        corpus_path = Path(store.store_dir) / "bm25_corpus.pkl"
        if RETRIEVAL_HYBRID_ENABLED and corpus_path.exists():
            faiss = FAISSRetriever(store)
            retriever = HybridRetriever.from_corpus_file(
                vector_retriever=faiss,
                corpus_path=corpus_path,
            )
            logger.info(f"混合检索就绪 (BM25 + 向量, BM25权重={RETRIEVAL_BM25_WEIGHT})")
        else:
            retriever = FAISSRetriever(store)
            logger.info("纯向量检索就绪")

        _engine = RAGEngine(retriever=retriever, llm=llm, top_k=RETRIEVAL_TOP_K)
        logger.info(f"RAG 引擎就绪 (索引: {store.doc_count} 条, top_k={RETRIEVAL_TOP_K})")

    return _engine
