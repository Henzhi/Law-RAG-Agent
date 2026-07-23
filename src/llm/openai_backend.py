"""
OpenAI 兼容 API LLM 后端实现。

支持所有兼容 OpenAI Chat Completions API 的服务:
  - OpenAI (gpt-4o, gpt-4o-mini)
  - DeepSeek (deepseek-chat, deepseek-reasoner)
  - 通义千问 (qwen-turbo, qwen-plus, qwen-max)
  - 本地 vLLM / Ollama OpenAI 兼容端点
  - 其他兼容服务

通过 openai Python SDK 调用，支持同步和流式。
"""
from __future__ import annotations

import logging
from typing import Iterator

from openai import OpenAI

from src.llm.base import LLMBackend

logger = logging.getLogger(__name__)

# 上下文窗口映射
_OPENAI_CONTEXT_WINDOWS = {
    "gpt-4o":                120000,
    "gpt-4o-mini":           120000,
    "gpt-4-turbo":           120000,
    "gpt-3.5-turbo":          16000,
    "deepseek-chat":          60000,
    "deepseek-v4-flash":      32000,
    "deepseek-reasoner":      60000,
    "qwen-turbo":             32000,
    "qwen-plus":             32000,
    "qwen-max":              32000,
    "qwen2.5:7b":            28000,
    "qwen2.5:14b":           60000,
}


class OpenAICompatibleBackend(LLMBackend):
    """OpenAI 兼容 API LLM 后端

    用法:
        backend = OpenAICompatibleBackend(
            model="deepseek-chat",
            api_key="sk-xxx",
            base_url="https://api.deepseek.com/v1",
        )
        reply = backend.chat("请解释一下什么是不正当竞争")
        for token in backend.chat_stream("请解释..."):
            print(token, end="")
    """

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: str = "",
        base_url: str = "https://api.deepseek.com/v1",
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 2048,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        super().__init__(
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        self.api_key = api_key
        self.base_url = base_url
        self._client = self._init_client()

    def _init_client(self) -> OpenAI:
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            max_retries=0,  # 我们自己管理重试
            timeout=120.0,
        )

    def get_context_window(self) -> int:
        return _OPENAI_CONTEXT_WINDOWS.get(self.model, 32000)

    # ------------------------------------------------------------------
    # LLMBackend 抽象方法实现
    # ------------------------------------------------------------------

    def _generate_impl(self, messages: list[dict[str, str]]) -> str:
        import time as _time

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    stream=False,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                last_error = e
                logger.warning(
                    f"OpenAI API 调用失败 (尝试 {attempt}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries:
                    _time.sleep(self.retry_delay * attempt)

        raise RuntimeError(
            f"OpenAI API 调用失败，已重试 {self.max_retries} 次: {last_error}"
        )

    def _stream_impl(self, messages: list[dict[str, str]]) -> Iterator[str]:
        import time as _time

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                stream = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    stream=True,
                )
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                return
            except Exception as e:
                last_error = e
                logger.warning(
                    f"OpenAI API 流式调用失败 (尝试 {attempt}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries:
                    _time.sleep(self.retry_delay * attempt)

        raise RuntimeError(
            f"OpenAI API 流式调用失败，已重试 {self.max_retries} 次: {last_error}"
        )
