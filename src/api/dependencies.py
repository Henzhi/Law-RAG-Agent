"""
API 依赖注入。

管理 LLM、向量库、RAG 引擎 / Agent 等单例，所有可配参数从 src.config 读取。
支持 FAISS 和 pgvector 两种后端。

v0.5: 引入多后端工厂函数 + 适配器，可通过 .env 切换 Ollama/OpenAI。
"""
from __future__ import annotations

import logging

from src.config import (
    LLM_MODEL, LLM_TEMPERATURE, LLM_TOP_P, LLM_MAX_TOKENS,
    LLM_BACKEND, LLM_MAX_RETRIES,
    EMBED_MODEL, EMBED_BATCH_SIZE, EMBED_MAX_RETRIES,
    RETRIEVAL_TOP_K, RETRIEVAL_HYBRID_ENABLED,
    RERANK_ENABLED, RERANK_MODEL, RERANK_RECALL_K, RERANK_TOP_K,
    AGENT_MAX_RETRIES,
    PG_ENABLED, PG_CONN,
    INDEX_NAME, INDEX_DIR,
    ADJACENT_ENABLED, ADJACENT_WINDOW,
)
from src.llm.adapter import LLMAdapter, EmbeddingAdapter
from src.llm.factory import create_llm_backend
from src.embedding.factory import create_embedding_backend
from src.embedding.vector_store import VectorStore
from src.rag.engine import RAGEngine
from src.rag.retriever import FAISSRetriever, PgvectorRetriever, PgvectorStoreRetriever
from src.rag.hybrid_retriever import HybridRetriever
from src.agents.graph import LawAgentGraph

logger = logging.getLogger(__name__)

_engine: RAGEngine | None = None
_agent: LawAgentGraph | None = None
_llm: object | None = None  # LLMAdapter，兼容旧 LawLLM 接口


def get_llm():
    """获取 LLM 实例（通过适配器兼容旧 API）

    根据 LLM_BACKEND 环境变量自动选择后端:
      ollama → OllamaBackend（本地）
      openai → OpenAICompatibleBackend（API）

    base_url 等连接参数由工厂函数从环境变量读取，
    此处只传模型无关的通用参数（temperature/top_p/max_tokens）。
    """
    global _llm
    if _llm is None:
        backend = create_llm_backend(
            backend_type=LLM_BACKEND,
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE, top_p=LLM_TOP_P,
            max_tokens=LLM_MAX_TOKENS, max_retries=LLM_MAX_RETRIES,
        )
        _llm = LLMAdapter(backend)
        logger.info(f"LLM 就绪: {LLM_BACKEND}:{LLM_MODEL} (context_window={backend.get_context_window()})")
    return _llm


def _create_embedder():
    """根据配置创建 Embedding 实例（通过适配器兼容旧 API）

    根据 EMBED_BACKEND 环境变量自动选择后端（默认 ollama，独立于 LLM_BACKEND）。
    base_url 等连接参数由工厂函数从环境变量读取，此处只传模型无关参数。
    """
    backend = create_embedding_backend(
        backend_type=None,  # 自动从环境变量读取
        model=EMBED_MODEL,
        batch_size=EMBED_BATCH_SIZE, max_retries=EMBED_MAX_RETRIES,
    )
    return EmbeddingAdapter(backend)


def _create_retriever(embedder):
    """根据配置创建检索器 (FAISS / pgvector v2)

    当 PG_ENABLED=true 时使用新的 PgvectorStore + halfvec + HNSW，
    否则回退到 FAISS。
    """
    from pathlib import Path

    store_dir = INDEX_DIR / INDEX_NAME

    if PG_ENABLED:
        from src.knowledge.pgvector_store import PgvectorStore
        logger.info("使用 pgvector v2 检索 (halfvec + HNSW)")
        store = PgvectorStore(PG_CONN)
        store.ensure_tables()
        retriever = PgvectorStoreRetriever(
            store=store,
            embedder=embedder,
            embedding_model=embedder.model,
        )
        return _wrap_adjacent(retriever, store_dir)

    # FAISS 模式
    logger.info(f"加载 FAISS: {store_dir}")
    store = VectorStore(embedder=embedder, persist_dir=INDEX_DIR, index_name=INDEX_NAME)
    if store.load() is None:
        raise RuntimeError(f"索引不存在: {store_dir}\n请先运行: uv run python scripts/build_index.py build")

    retriever = FAISSRetriever(store)

    # 混合检索
    corpus_path = Path(store.store_dir) / "bm25_corpus.pkl"
    if RETRIEVAL_HYBRID_ENABLED and corpus_path.exists():
        faiss = FAISSRetriever(store)
        retriever = HybridRetriever.from_corpus_file(vector_retriever=faiss, corpus_path=corpus_path)
        logger.info("混合检索就绪")

    # 相邻扩展
    retriever = _wrap_adjacent(retriever, store_dir)

    # Reranker 兜底精排
    if RERANK_ENABLED:
        from src.rag.reranker import Reranker, RerankRetriever
        reranker = Reranker(model_name=RERANK_MODEL)
        retriever = RerankRetriever(base_retriever=retriever, reranker=reranker, recall_k=RERANK_RECALL_K, top_k=RERANK_TOP_K)
        logger.info(f"Reranker 就绪: 粗排{RERANK_RECALL_K} → 精排{RERANK_TOP_K}")

    return retriever


def _wrap_adjacent(retriever, store_dir):
    """如果启用，包裹相邻扩展检索器（最外层）"""
    if ADJACENT_ENABLED:
        from pathlib import Path
        from src.rag.adjacent_expander import AdjacentExpander
        map_path = Path(store_dir) / "article_map.json"
        retriever = AdjacentExpander(base_retriever=retriever, article_map_path=map_path, window=ADJACENT_WINDOW)
    return retriever


def get_engine() -> RAGEngine:
    global _engine
    if _engine is None:
        llm = get_llm()
        embedder = _create_embedder()
        retriever = _create_retriever(embedder)
        _engine = RAGEngine(retriever=retriever, llm=llm, top_k=RETRIEVAL_TOP_K)
        logger.info("RAG 引擎就绪")
    return _engine


def _create_memory_manager(llm, embedder):
    """创建对话记忆管理器（需要 pgvector 环境）"""
    if not PG_ENABLED:
        return None
    try:
        from src.memory.conversation import ConversationMemoryManager
        return ConversationMemoryManager(conn_string=PG_CONN, embedder=embedder, llm=llm)
    except Exception as e:
        logger.warning(f"记忆管理器初始化失败（pgvector 未就绪？）: {e}")
        return None


def _create_faq_cache(embedder):
    """创建 FAQ 语义缓存管理器（需要 pgvector 环境）"""
    if not PG_ENABLED:
        return None
    try:
        from src.memory.faq_cache import FAQCache
        return FAQCache(conn_string=PG_CONN, embedder=embedder)
    except Exception as e:
        logger.warning(f"FAQ缓存初始化失败（pgvector 未就绪？）: {e}")
        return None


def get_agent(force_reload: bool = False) -> LawAgentGraph:
    """获取 LangGraph 多 Agent 引擎（含记忆管理器 + FAQ 缓存）"""
    global _agent
    if force_reload:
        _agent = None
    if _agent is None:
        llm = get_llm()
        embedder = _create_embedder()
        retriever = _create_retriever(embedder)
        memory_mgr = _create_memory_manager(llm, embedder)
        faq_cache = _create_faq_cache(embedder)
        _agent = LawAgentGraph(
            retriever=retriever, llm=llm,
            top_k=RETRIEVAL_TOP_K, max_retries=AGENT_MAX_RETRIES,
            memory_manager=memory_mgr,
            faq_cache=faq_cache,
        )
        extras = []
        if memory_mgr:
            extras.append("记忆")
        if faq_cache:
            extras.append("FAQ缓存")
        logger.info(f"LangGraph Agent 就绪 ({'/'.join(extras) if extras else '基础模式'})")
    return _agent
