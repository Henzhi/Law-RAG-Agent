"""
统一配置模块。

从 .env 文件和环境变量加载所有可配参数，提供一站式配置入口。
模块级变量可直接 from src.config import xxx 使用。
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# 自动加载项目根目录的 .env 文件
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# ---- 日志 ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

# 强制离线模式 — 必须在任何 HuggingFace 相关 import 之前设置
# sentence_transformers 5.x 的某些版本不完全尊重 HF_HUB_OFFLINE，
# 所以这里同时设三个环境变量 + 后续传给 CrossEncoder 的 local_files_only
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


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

# 检索时是否过滤章级摘要 chunk（噪声大，评测已验证应过滤）
# 在检索层统一拦截，避免运行时出现 30+ 条无关条文被召回的问题
RETRIEVAL_DROP_SUMMARY_CHUNKS = os.getenv("RETRIEVAL_DROP_SUMMARY_CHUNKS", "true").lower() == "true"

# Reranker 二次精排 (Cross-Encoder)。评测验证可显著提升召回质量、消除噪声；
# 纯 CPU 推理会增加少量延迟，有 GPU 更佳。默认开启以对齐评测验证过的配置。
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() == "true"
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_RECALL_K = int(os.getenv("RERANK_RECALL_K", "15"))  # 粗排召回数
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))          # 精排后返回数

# 连续片段扩展：检索后自动拉取相邻 ±N 条条文
ADJACENT_ENABLED = os.getenv("ADJACENT_ENABLED", "true").lower() == "true"
ADJACENT_WINDOW = int(os.getenv("ADJACENT_WINDOW", "3"))     # ±N 条


# ---------------------------------------------------------------------------
# 向量索引
# ---------------------------------------------------------------------------

INDEX_NAME = os.getenv("INDEX_NAME", "law_index")
INDEX_DIR = _PROJECT_ROOT / os.getenv("INDEX_DIR", "data/vector_store")


# ---------------------------------------------------------------------------
# LangGraph Agent
# ---------------------------------------------------------------------------

# LangGraph Agent 路径（含查询改写 + 答案校验）。默认关闭：
# 开启后每条查询会额外发起 rewrite + validate 两次 LLM 调用，延迟显著上升；
# 追求最高回答质量时可设为 true（需 GPU 或接受慢速）。检索质量与噪声过滤不依赖它。
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
