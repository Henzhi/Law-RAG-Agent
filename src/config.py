"""
统一配置模块。

从 .env 文件和环境变量加载所有可配参数，提供一站式配置入口。
模块级变量可直接 from src.config import xxx 使用。
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# 自动加载项目根目录的 .env 文件
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.9"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "http://localhost:11434")
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))


# ---------------------------------------------------------------------------
# 检索
# ---------------------------------------------------------------------------

RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
RETRIEVAL_HYBRID_ENABLED = os.getenv("RETRIEVAL_HYBRID_ENABLED", "false").lower() == "true"
RETRIEVAL_BM25_WEIGHT = float(os.getenv("RETRIEVAL_BM25_WEIGHT", "0.0"))

# Reranker 二次精排 (CPU 较慢，建议有 GPU 再开启)
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "false").lower() == "true"
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_RECALL_K = int(os.getenv("RERANK_RECALL_K", "10"))  # 粗排召回数
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))          # 精排后返回数

# 连续片段扩展：检索后自动拉取相邻 ±N 条条文
ADJACENT_ENABLED = os.getenv("ADJACENT_ENABLED", "false").lower() == "true"
ADJACENT_WINDOW = int(os.getenv("ADJACENT_WINDOW", "2"))     # ±N 条


# ---------------------------------------------------------------------------
# 向量索引
# ---------------------------------------------------------------------------

INDEX_NAME = os.getenv("INDEX_NAME", "law_index")
INDEX_DIR = _PROJECT_ROOT / os.getenv("INDEX_DIR", "data/vector_store")


# ---------------------------------------------------------------------------
# LangGraph Agent
# ---------------------------------------------------------------------------

AGENT_ENABLED = os.getenv("AGENT_ENABLED", "false").lower() == "true"
AGENT_MAX_RETRIES = int(os.getenv("AGENT_MAX_RETRIES", "1"))

# ---------------------------------------------------------------------------
# pgvector
# ---------------------------------------------------------------------------

PG_ENABLED = os.getenv("PG_ENABLED", "false").lower() == "true"
PG_CONN = os.getenv("PG_CONN", "postgresql://lawrag:lawrag123@localhost:5432/lawrag")

# ---------------------------------------------------------------------------
# 服务
# ---------------------------------------------------------------------------

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
