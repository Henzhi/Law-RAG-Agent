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

EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "http://localhost:11434")
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))


# ---------------------------------------------------------------------------
# 检索
# ---------------------------------------------------------------------------

RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
RETRIEVAL_HYBRID_ENABLED = os.getenv("RETRIEVAL_HYBRID_ENABLED", "false").lower() == "true"
RETRIEVAL_BM25_WEIGHT = float(os.getenv("RETRIEVAL_BM25_WEIGHT", "0.3"))


# ---------------------------------------------------------------------------
# 向量索引
# ---------------------------------------------------------------------------

INDEX_NAME = os.getenv("INDEX_NAME", "law_index")
INDEX_DIR = _PROJECT_ROOT / os.getenv("INDEX_DIR", "data/vector_store")


# ---------------------------------------------------------------------------
# 服务
# ---------------------------------------------------------------------------

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
