"""
API 依赖注入。

管理 LLM、向量库、RAG 引擎 / Agent 等单例，所有可配参数从 src.config 读取。
v0.6: 纯 PG 架构，检索后端统一为 pgvector（已移除 FAISS）。
"""
from __future__ import annotations

import logging

from src.config import (
    LLM_MODEL, LLM_TEMPERATURE, LLM_TOP_P, LLM_MAX_TOKENS,
    LLM_BACKEND, LLM_MAX_RETRIES,
    EMBED_MODEL, EMBED_BATCH_SIZE, EMBED_MAX_RETRIES,
    RETRIEVAL_TOP_K,
    RERANK_ENABLED, RERANK_MODEL, RERANK_RECALL_K, RERANK_TOP_K,
    AGENT_MAX_RETRIES,
    PG_CONN,
    ADJACENT_ENABLED, ADJACENT_WINDOW,
    HYBRID_ENABLED, HYBRID_RRF_K, HYBRID_BM25_WEIGHT,
)
from src.llm.adapter import LLMAdapter, EmbeddingAdapter
from src.llm.factory import create_llm_backend
from src.embedding.factory import create_embedding_backend
from src.rag.engine import RAGEngine
from src.rag.retriever import PgvectorStoreRetriever
from src.agents.graph import LawAgentGraph

logger = logging.getLogger(__name__)

_engine: RAGEngine | None = None
_agent: LawAgentGraph | None = None
_llm: object | None = None  # LLMAdapter，兼容旧 LawLLM 接口
_memory_mgr: object | None = None  # ConversationMemoryManager | None


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
    """创建检索器（纯 pgvector：PgvectorStore + halfvec + HNSW）

    v0.6 起强制 pgvector，不再支持 FAISS 回退。
    PG 连接失败将直接抛错（不静默降级），保证部署配置正确性。
    """
    from pathlib import Path

    from src.knowledge.pgvector_store import PgvectorStore
    logger.info("使用 pgvector 检索 (halfvec + HNSW)")
    store = PgvectorStore(PG_CONN)
    store.ensure_tables()
    retriever = PgvectorStoreRetriever(
        store=store,
        embedder=embedder,
        embedding_model=embedder.model,
    )

    # Reranker 精排（若启用）
    if RERANK_ENABLED:
        from src.rag.reranker import Reranker, RerankRetriever
        reranker = Reranker(model_name=RERANK_MODEL)
        retriever = RerankRetriever(base_retriever=retriever, reranker=reranker, recall_k=RERANK_RECALL_K, top_k=RERANK_TOP_K)
        logger.info(f"Reranker 就绪: 粗排{RERANK_RECALL_K} → 精排{RERANK_TOP_K}")

    # 相邻扩展（article_map 缺失时自动降级为空转）
    if ADJACENT_ENABLED:
        from src.rag.adjacent_expander import AdjacentExpander
        map_path = Path(__file__).resolve().parents[2] / "data" / "vector_store" / "article_map.json"
        retriever = AdjacentExpander(base_retriever=retriever, article_map_path=map_path, window=ADJACENT_WINDOW)

    # BM25 关键词混合（rank-based 条件激活）：仅法名/条款查询参与，BM25 索引懒加载
    if HYBRID_ENABLED:
        from src.rag.bm25_retriever import Bm25Retriever
        from src.rag.hybrid_retriever import HybridRetriever
        bm25 = Bm25Retriever(store)
        retriever = HybridRetriever(
            base_retriever=retriever,
            bm25_retriever=bm25,
            rrf_k=HYBRID_RRF_K,
            bm25_weight=HYBRID_BM25_WEIGHT,
        )
        logger.info(f"BM25 条件混合就绪: RRF k={HYBRID_RRF_K}, bm25_w={HYBRID_BM25_WEIGHT} (懒加载)")

    # 条款号精确路由（最外层）：对"法名+第X条"查询做精确置顶，弥补纯向量对条款号查询的失配
    from src.rag.article_router import ArticleRouter
    retriever = ArticleRouter(base_retriever=retriever, store=store)

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
    """创建对话记忆管理器（纯 PG，需要 pgvector 环境）"""
    try:
        from src.memory.conversation import ConversationMemoryManager
        return ConversationMemoryManager(conn_string=PG_CONN, embedder=embedder, llm=llm)
    except Exception as e:
        logger.warning(f"记忆管理器初始化失败（pgvector 未就绪？）: {e}")
        return None


def _create_faq_cache(embedder):
    """创建 FAQ 语义缓存管理器（纯 PG，需要 pgvector 环境）"""
    try:
        from src.memory.faq_cache import FAQCache
        return FAQCache(conn_string=PG_CONN, embedder=embedder)
    except Exception as e:
        logger.warning(f"FAQ缓存初始化失败（pgvector 未就绪？）: {e}")
        return None


def get_memory_manager():
    """获取对话记忆管理器单例（会话保存时异步固化记忆用）。

    与 get_agent 内创建的记忆管理器复用同一实例，避免重复连接 PG。
    """
    global _memory_mgr
    if _memory_mgr is None:
        llm = get_llm()
        embedder = _create_embedder()
        _memory_mgr = _create_memory_manager(llm, embedder)
    return _memory_mgr


def get_agent(force_reload: bool = False) -> LawAgentGraph:
    """获取 LangGraph 多 Agent 引擎（含记忆管理器 + FAQ 缓存）"""
    global _agent
    if force_reload:
        _agent = None
    if _agent is None:
        llm = get_llm()
        embedder = _create_embedder()
        retriever = _create_retriever(embedder)
        memory_mgr = get_memory_manager()
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
