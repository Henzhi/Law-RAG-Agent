"""
API 依赖注入。

管理 LLM、向量库、RAG 引擎 / Agent 等单例，所有可配参数从 src.config 读取。
支持 FAISS 和 pgvector 两种后端。
"""
from __future__ import annotations

import logging

from src.config import (
    LLM_MODEL, LLM_BASE_URL, LLM_TEMPERATURE, LLM_TOP_P, LLM_MAX_TOKENS,
    EMBED_MODEL, EMBED_BASE_URL, EMBED_BATCH_SIZE,
    RETRIEVAL_TOP_K, RETRIEVAL_HYBRID_ENABLED,
    RERANK_ENABLED, RERANK_MODEL, RERANK_RECALL_K, RERANK_TOP_K,
    AGENT_ENABLED, AGENT_MAX_RETRIES,
    PG_ENABLED, PG_CONN,
    INDEX_NAME, INDEX_DIR,
)
from src.embedding.embedder import LawEmbedder
from src.embedding.vector_store import VectorStore
from src.llm.client import LawLLM, LLMConfig
from src.rag.engine import RAGEngine
from src.rag.retriever import FAISSRetriever, PgvectorRetriever
from src.rag.hybrid_retriever import HybridRetriever
from src.agents.graph import LawAgentGraph

logger = logging.getLogger(__name__)

_engine: RAGEngine | None = None
_agent: LawAgentGraph | None = None
_llm: LawLLM | None = None


def get_llm() -> LawLLM:
    global _llm
    if _llm is None:
        logger.info(f"LLM 初始化: {LLM_MODEL}")
        _llm = LawLLM(
            model=LLM_MODEL, base_url=LLM_BASE_URL,
            config=LLMConfig(temperature=LLM_TEMPERATURE, top_p=LLM_TOP_P, num_predict=LLM_MAX_TOKENS),
        )
    return _llm


def _create_retriever(embedder: LawEmbedder):
    """根据配置创建检索器 (FAISS/pgvector)"""
    from pathlib import Path

    if PG_ENABLED:
        logger.info("使用 pgvector 检索")
        retriever = PgvectorRetriever(embedder=embedder, conn_string=PG_CONN)
        return retriever

    # FAISS 模式
    logger.info(f"加载 FAISS: {INDEX_DIR / INDEX_NAME}")
    store = VectorStore(embedder=embedder, persist_dir=INDEX_DIR, index_name=INDEX_NAME)
    if store.load() is None:
        raise RuntimeError(f"索引不存在: {INDEX_DIR / INDEX_NAME}\n请先运行: uv run python scripts/build_index.py build")

    retriever = FAISSRetriever(store)

    # 混合检索
    corpus_path = Path(store.store_dir) / "bm25_corpus.pkl"
    if RETRIEVAL_HYBRID_ENABLED and corpus_path.exists():
        faiss = FAISSRetriever(store)
        retriever = HybridRetriever.from_corpus_file(vector_retriever=faiss, corpus_path=corpus_path)
        logger.info("混合检索就绪")

    # Reranker
    if RERANK_ENABLED:
        from src.rag.reranker import Reranker, RerankRetriever
        reranker = Reranker(model_name=RERANK_MODEL)
        retriever = RerankRetriever(base_retriever=retriever, reranker=reranker, recall_k=RERANK_RECALL_K, top_k=RERANK_TOP_K)
        logger.info(f"Reranker 就绪: 粗排{RERANK_RECALL_K} → 精排{RERANK_TOP_K}")

    return retriever


def get_engine() -> RAGEngine:
    global _engine
    if _engine is None:
        llm = get_llm()
        embedder = LawEmbedder(model=EMBED_MODEL, base_url=EMBED_BASE_URL, batch_size=EMBED_BATCH_SIZE)
        retriever = _create_retriever(embedder)
        _engine = RAGEngine(retriever=retriever, llm=llm, top_k=RETRIEVAL_TOP_K)
        logger.info("RAG 引擎就绪")
    return _engine


def get_agent() -> LawAgentGraph:
    """获取 LangGraph 多 Agent 引擎"""
    global _agent
    if _agent is None:
        llm = get_llm()
        embedder = LawEmbedder(model=EMBED_MODEL, base_url=EMBED_BASE_URL, batch_size=EMBED_BATCH_SIZE)
        retriever = _create_retriever(embedder)
        _agent = LawAgentGraph(retriever=retriever, llm=llm, top_k=RETRIEVAL_TOP_K, max_retries=AGENT_MAX_RETRIES)
        logger.info("LangGraph Agent 就绪")
    return _agent
