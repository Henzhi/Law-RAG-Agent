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


def _safe_float(key: str, default: float) -> float:
    """安全获取浮点型环境变量，格式错误时使用默认值并警告"""
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        logging.getLogger(__name__).warning(
            f"环境变量 {key}='{val}' 不是合法浮点数，使用默认值 {default}"
        )
        return default


def _safe_int(key: str, default: int) -> int:
    """安全获取整型环境变量，格式错误时使用默认值并警告"""
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        logging.getLogger(__name__).warning(
            f"环境变量 {key}='{val}' 不是合法整数，使用默认值 {default}"
        )
        return default


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

# 后端类型：ollama | openai
# 未设置 EMBED_BACKEND 时也会回退到此值
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")
LLM_TEMPERATURE = _safe_float("LLM_TEMPERATURE", 0.1)
LLM_TOP_P = _safe_float("LLM_TOP_P", 0.9)
LLM_MAX_TOKENS = _safe_int("LLM_MAX_TOKENS", 2048)
LLM_MAX_RETRIES = _safe_int("LLM_MAX_RETRIES", 3)

# OpenAI 兼容后端配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "deepseek-chat")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

# 后端类型：ollama | openai
# 未设置时回退到 LLM_BACKEND 的值
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "http://localhost:11434")
EMBED_BATCH_SIZE = _safe_int("EMBED_BATCH_SIZE", 32)
EMBED_MAX_RETRIES = _safe_int("EMBED_MAX_RETRIES", 3)


# ---------------------------------------------------------------------------
# 检索
# ---------------------------------------------------------------------------

RETRIEVAL_TOP_K = _safe_int("RETRIEVAL_TOP_K", 5)

# 检索时是否过滤章级摘要 chunk（噪声大，评测已验证应过滤）
# 在检索层统一拦截，避免运行时出现 30+ 条无关条文被召回的问题
RETRIEVAL_DROP_SUMMARY_CHUNKS = os.getenv("RETRIEVAL_DROP_SUMMARY_CHUNKS", "true").lower() == "true"

# 向量相似度召回阈值（bge-m3 归一化内积，范围约 [-1, 1]，0.95≈强相关、<0.4 视为较差）。
# 仅作为召回质量闸门：向量分数低于阈值的结果被丢弃；若过滤后无候选则回退保留原结果（避免哑火）。
# 0.0 表示关闭（默认，保持评测指标 Recall@5=73% 不变）。建议启用值 0.3~0.5。
RETRIEVAL_SIM_THRESHOLD = _safe_float("RETRIEVAL_SIM_THRESHOLD", 0.0)

# Reranker 二次精排 (Cross-Encoder)。评测验证可显著提升召回质量、消除噪声；
# 纯 CPU 推理会增加少量延迟，有 GPU 更佳。默认开启以对齐评测验证过的配置。
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() == "true"
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_RECALL_K = _safe_int("RERANK_RECALL_K", 15)  # 粗排召回数
RERANK_TOP_K = _safe_int("RERANK_TOP_K", 15)          # 精排后返回数

# 连续片段扩展：检索后自动拉取相邻 ±N 条条文
ADJACENT_ENABLED = os.getenv("ADJACENT_ENABLED", "true").lower() == "true"
# 相邻扩展窗口：原默认 ±3 会把每条命中扩展成 7 条，引用列表被大量
# "相邻但不相关"的条文污染；±1 仅保留紧邻上下文（引用仍以检索命中为主）
ADJACENT_WINDOW = _safe_int("ADJACENT_WINDOW", 1)     # ±N 条


# ---------------------------------------------------------------------------
# 向量索引
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# LangGraph Agent
# ---------------------------------------------------------------------------

# LangGraph Agent 路径（含答案校验/自动重试）。默认关闭：
# 开启后每条查询会额外发起一次 validate LLM 调用，延迟上升；
# 追求更高回答质量（幻觉审核 + 自动重试）时可设为 true。检索质量与噪声过滤不依赖它。
AGENT_ENABLED = os.getenv("AGENT_ENABLED", "false").lower() == "true"
AGENT_MAX_RETRIES = _safe_int("AGENT_MAX_RETRIES", 1)

# ---------------------------------------------------------------------------
# pgvector
# ---------------------------------------------------------------------------

PG_ENABLED = os.getenv("PG_ENABLED", "false").lower() == "true"
# 安全基线：连接串默认值不含密码，生产环境必须通过 PG_CONN 环境变量显式提供
PG_CONN = os.getenv("PG_CONN", "postgresql://lawrag@localhost:5432/lawrag")

# ---------------------------------------------------------------------------
# 服务
# ---------------------------------------------------------------------------

HOST = os.getenv("HOST", "0.0.0.0")
PORT = _safe_int("PORT", 8000)
