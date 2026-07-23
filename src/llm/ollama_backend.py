"""
Ollama LLM 后端实现。

通过 ollama Python SDK 调用本地 Ollama 服务，支持同步和流式调用，
内置自动重试机制。

兼容 LangChain BaseChatModel 接口，可无缝集成到 LangGraph Agent。
"""
from __future__ import annotations

import time
import logging
from typing import Any, Iterator

import ollama
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from src.llm.base import LLMBackend

logger = logging.getLogger(__name__)

# 上下文窗口映射: 模型名 → token数
_OLLAMA_CONTEXT_WINDOWS = {
    "qwen2.5:7b":         28000,
    "qwen2.5:14b":        60000,
    "qwen2.5:32b":        80000,
    "qwen2.5:72b":        80000,
    "qwen3:8b":           32000,
    "llama3.1:8b":        32000,
    "deepseek-r1:7b":     32000,
    "deepseek-r1:14b":    64000,
}


class OllamaBackend(LLMBackend):
    """Ollama LLM 后端

    用法:
        backend = OllamaBackend(model="qwen2.5:7b", base_url="http://localhost:11434")
        reply = backend.chat("请解释一下什么是不正当竞争")
        for token in backend.chat_stream("请解释..."):
            print(token, end="")
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 2048,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        repeat_penalty: float = 1.05,
        seed: int = 42,
    ):
        super().__init__(
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        self.base_url = base_url
        self.repeat_penalty = repeat_penalty
        self.seed = seed
        self._client = self._init_client()

    def _init_client(self) -> ollama.Client:
        host = self.base_url.replace("http://", "").replace("https://", "")
        return ollama.Client(host=host, timeout=300.0)

    def get_context_window(self) -> int:
        return _OLLAMA_CONTEXT_WINDOWS.get(self.model, 28000)

    # ------------------------------------------------------------------
    # LLMBackend 抽象方法实现
    # ------------------------------------------------------------------

    def _generate_impl(self, messages: list[dict[str, str]]) -> str:
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.chat(
                    model=self.model,
                    messages=messages,
                    options={
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "num_predict": self.max_tokens,
                        "repeat_penalty": self.repeat_penalty,
                        "seed": self.seed,
                    },
                )
                return response["message"]["content"]
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Ollama 调用失败 (尝试 {attempt}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)

        raise RuntimeError(
            f"Ollama 调用失败，已重试 {self.max_retries} 次: {last_error}"
        )

    def _stream_impl(self, messages: list[dict[str, str]]) -> Iterator[str]:
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                stream = self._client.chat(
                    model=self.model,
                    messages=messages,
                    options={
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "num_predict": self.max_tokens,
                        "repeat_penalty": self.repeat_penalty,
                        "seed": self.seed,
                    },
                    stream=True,
                )
                for chunk in stream:
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        yield content
                return
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Ollama 流式调用失败 (尝试 {attempt}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)

        raise RuntimeError(
            f"Ollama 流式调用失败，已重试 {self.max_retries} 次: {last_error}"
        )


# ---------------------------------------------------------------------------
# LangChain 兼容包装器
# ---------------------------------------------------------------------------

class OllamaLangChainWrapper(BaseChatModel):
    """将 OllamaBackend 包装为 LangChain BaseChatModel

    使 Ollama 后端可以无缝用于 LangGraph Agent 和其他 LangChain 组件。
    """

    model_name: str = "qwen2.5:7b"
    temperature: float = 0.1

    _backend: OllamaBackend | None = None

    def __init__(self, backend: OllamaBackend):
        super().__init__(
            model_name=backend.model,
            temperature=backend.temperature,
        )
        self._backend = backend

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        ollama_msgs = self._langchain_to_dict(messages)
        response = self._backend._generate_impl(ollama_msgs)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        ollama_msgs = self._langchain_to_dict(messages)
        for token_text in self._backend._stream_impl(ollama_msgs):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=token_text))
            if run_manager:
                run_manager.on_llm_new_token(token_text, chunk=chunk)
            yield chunk

    @property
    def _llm_type(self) -> str:
        return "ollama-law-llm"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "temperature": self.temperature,
            "base_url": self._backend.base_url,
        }

    @staticmethod
    def _langchain_to_dict(messages: list[BaseMessage]) -> list[dict[str, str]]:
        result = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                result.append({"role": "system", "content": str(msg.content)})
            elif isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": str(msg.content)})
            elif isinstance(msg, AIMessage):
                result.append({"role": "assistant", "content": str(msg.content)})
            else:
                result.append({"role": "user", "content": str(msg.content)})
        return result
