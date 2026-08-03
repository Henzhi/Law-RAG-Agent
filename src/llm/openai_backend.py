"""
OpenAI 兼容 API LLM 后端实现。

支持所有兼容 OpenAI Chat Completions API 的服务:
  - OpenAI (gpt-4o, gpt-4o-mini)
  - DeepSeek (deepseek-chat, deepseek-reasoner)
  - 通义千问 (qwen-turbo, qwen-plus, qwen-max)
  - 本地 vLLM / Ollama OpenAI 兼容端点
  - 其他兼容服务

通过 openai Python SDK 调用，支持同步和流式。

重试策略（src.llm.retry）:
  - 仅重试可重试异常（429 / 5xx / 网络 / 超时），4xx 业务错误直接抛出
  - 指数退避 + 全抖动 + 尊重 Retry-After 头，避免 429 惊群放大限流
  - 流式请求已产出内容后失败不再重试（避免重复 token / 重复计费），
    并在 finally 中关闭底层流以尽快释放连接
"""
from __future__ import annotations

import logging
from typing import Iterator

from openai import OpenAI

from src.llm.base import LLMBackend
from src.llm.retry import is_retryable, wait_and_log

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
        last_error: Exception | None = None
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
                if not is_retryable(e):
                    logger.warning(
                        f"OpenAI API 调用失败（不可重试）: {e}"
                    )
                    raise
                wait_and_log(e, attempt, self.max_retries, logger_name=__name__)

        raise RuntimeError(
            f"OpenAI API 调用失败，已重试 {self.max_retries} 次: {last_error}"
        )

    def _stream_impl(self, messages: list[dict[str, str]]) -> Iterator[str]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            stream = None
            yielded_any = False
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
                        yielded_any = True
                        yield chunk.choices[0].delta.content
                return
            except GeneratorExit:
                # 调用方中断（客户端断开 / 桥接取消）：关闭底层连接立即停止消耗
                raise
            except Exception as e:
                last_error = e
                if yielded_any:
                    # 已向用户输出过内容，不能从头重试（会重复 / 重复计费），
                    # 直接抛给上层处理。
                    logger.warning(
                        f"OpenAI API 流式中途失败（已输出内容，不重试）: {e}"
                    )
                    raise
                if not is_retryable(e):
                    logger.warning(
                        f"OpenAI API 流式调用失败（不可重试）: {e}"
                    )
                    raise
                wait_and_log(e, attempt, self.max_retries, logger_name=__name__)
            finally:
                if stream is not None:
                    try:
                        # openai SDK 的 Stream 支持 close()，立即断开 HTTP 连接
                        close = getattr(stream, "close", None)
                        if callable(close):
                            close()
                    except Exception:
                        pass

        raise RuntimeError(
            f"OpenAI API 流式调用失败，已重试 {self.max_retries} 次: {last_error}"
        )
