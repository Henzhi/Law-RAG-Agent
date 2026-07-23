"""
LLM 后端工厂函数。

根据环境变量配置自动选择并创建 LLM 后端实例。
支持 Ollama 和 OpenAI 兼容 API 两种后端。
"""
from __future__ import annotations

import logging
import os

from src.llm.base import LLMBackend
from src.llm.ollama_backend import OllamaBackend
from src.llm.openai_backend import OpenAICompatibleBackend

logger = logging.getLogger(__name__)

# 默认系统提示词
LAW_SYSTEM_PROMPT = """你是一位专业的中国法律助手，具备以下能力：

1. 引用具体的法律条文回答用户问题
2. 根据用户提供的历史消息上下文中的法律条文进行推理
3. 回答简洁准确，优先使用法律原文
4. 如果被问及法律条文中没有涉及的内容，明确指出缺乏依据
5. 用中文回答，条理清晰"""


def create_llm_backend(
    backend_type: str | None = None,
    **kwargs,
) -> LLMBackend:
    """创建 LLM 后端实例

    根据 backend_type 或环境变量 LLM_BACKEND 自动选择后端:

    - "ollama": OllamaBackend（本地部署）
    - "openai": OpenAICompatibleBackend（API 调用）

    Args:
        backend_type: 后端类型，为 None 时从环境变量 LLM_BACKEND 读取
        **kwargs: 传给具体后端的参数

    Returns:
        LLMBackend 实例
    """
    if backend_type is None:
        backend_type = os.getenv("LLM_BACKEND", "ollama")

    backend_type = backend_type.lower()

    if backend_type == "ollama":
        return _create_ollama(**kwargs)
    elif backend_type in ("openai", "openai_compatible"):
        return _create_openai(**kwargs)
    else:
        raise ValueError(
            f"不支持的 LLM 后端类型: '{backend_type}'。"
            f"支持的类型: ollama, openai"
        )


def _create_ollama(**kwargs) -> OllamaBackend:
    model = kwargs.get("model") or os.getenv("LLM_MODEL", "qwen2.5:7b")
    base_url = kwargs.get("base_url") or os.getenv("LLM_BASE_URL", "http://localhost:11434")
    temperature = kwargs.get("temperature", float(os.getenv("LLM_TEMPERATURE", "0.1")))
    top_p = kwargs.get("top_p", float(os.getenv("LLM_TOP_P", "0.9")))
    max_tokens = kwargs.get("max_tokens", int(os.getenv("LLM_MAX_TOKENS", "2048")))
    max_retries = kwargs.get("max_retries", int(os.getenv("LLM_MAX_RETRIES", "3")))

    logger.info(f"创建 Ollama 后端: model={model}, base_url={base_url}")
    return OllamaBackend(
        model=model,
        base_url=base_url,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        max_retries=max_retries,
    )


def _create_openai(**kwargs) -> OpenAICompatibleBackend:
    model = kwargs.get("model") or os.getenv("OPENAI_MODEL", "deepseek-chat")
    api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY", "")
    base_url = kwargs.get("base_url") or os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    temperature = kwargs.get("temperature", float(os.getenv("LLM_TEMPERATURE", "0.1")))
    top_p = kwargs.get("top_p", float(os.getenv("LLM_TOP_P", "0.9")))
    max_tokens = kwargs.get("max_tokens", int(os.getenv("LLM_MAX_TOKENS", "2048")))
    max_retries = kwargs.get("max_retries", int(os.getenv("LLM_MAX_RETRIES", "3")))

    if not api_key:
        raise ValueError(
            "使用 OpenAI 兼容后端必须设置 OPENAI_API_KEY 环境变量"
        )

    safe_key = api_key[:8] + "..." if len(api_key) > 8 else "***"
    logger.info(f"创建 OpenAI 兼容后端: model={model}, base_url={base_url}, api_key={safe_key}")
    return OpenAICompatibleBackend(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        max_retries=max_retries,
    )
