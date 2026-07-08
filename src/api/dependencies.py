"""
API 依赖注入。

管理 LLM、向量库、RAG 引擎等单例，确保只初始化一次。
"""
from __future__ import annotations

import os
import logging
from functools import lru_cache
from pathlib import Path

from src.embedding.embedder import LawEmbedder
from src.embedding.vector_store import VectorStore
from src.llm.client import LawLLM
from src.rag.engine import create_rag_engine, RAGEngine

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"
INDEX_NAME = os.getenv("LAW_INDEX_NAME", "law_index")
LLM_MODEL = os.getenv("LAW_LLM_MODEL", "qwen2.5:7b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
TOP_K = int(os.getenv("LAW_TOP_K", "5"))


_engine: RAGEngine | None = None
_llm: LawLLM | None = None


def get_llm() -> LawLLM:
    """获取 LLM 单例"""
    global _llm
    if _llm is None:
        logger.info(f"初始化 LLM: {LLM_MODEL} @ {OLLAMA_URL}")
        _llm = LawLLM(model=LLM_MODEL, base_url=OLLAMA_URL)
    return _llm


def get_engine() -> RAGEngine:
    """获取 RAG 引擎单例"""
    global _engine
    if _engine is None:
        llm = get_llm()

        logger.info(f"加载向量库: {VECTOR_STORE_DIR / INDEX_NAME}")
        embedder = LawEmbedder(base_url=OLLAMA_URL)
        store = VectorStore(
            embedder=embedder,
            persist_dir=VECTOR_STORE_DIR,
            index_name=INDEX_NAME,
        )
        if store.load() is None:
            raise RuntimeError(
                f"FAISS 索引不存在: {VECTOR_STORE_DIR / INDEX_NAME}\n"
                f"请先运行: uv run python scripts/build_index.py build"
            )

        _engine = create_rag_engine(store, llm, top_k=TOP_K)
        logger.info(f"RAG 引擎就绪 (索引: {store.doc_count} 条)")
    return _engine
