"""
Ollama LLM 后端实现。

通过 ollama Python SDK 调用本地 Ollama 服务，支持同步和流式调用。

重试策略（src.llm.retry）:
  - 仅重试可重试异常（429 / 5xx / 网络 / 超时），4xx 业务错误直接抛出
  - 指数退避 + 全抖动 + 尊重 Retry-After 头
  - 流式请求已产出内容后失败不再重试（避免重复 token），
    并在 finally 中尽力关闭底层流以尽快释放连接
"""
from __future__ import annotations

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
from src.llm.retry import is_retryable, wait_and_log

logger = logging.getLogger(__name__)

# 上下文窗口映射: 模型名 → token数
# 说明: 窗口应取模型真实上下文能力且略保守(留 KV Cache 余量)，
#       与 Ollama 服务端 num_ctx 保持一致；可用 OLLAMA_NUM_CTX 显式覆盖。
_OLLAMA_CONTEXT_WINDOWS = {
    "qwen2.5:3b":         32000,  # 真实 32768，取保守值
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
        num_ctx: int = 0,
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
        # 请求上下文窗口：0 = 自动使用模型声明窗口（get_context_window()）。
        # Ollama 服务端 num_ctx 默认仅 2048，不显式下发会静默截断输入。
        self.num_ctx = num_ctx or self.get_context_window()
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
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.chat(
                    model=self.model,
                    messages=messages,
                    options={
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "num_predict": self.max_tokens,
                        "num_ctx": self.num_ctx,
                        "repeat_penalty": self.repeat_penalty,
                        "seed": self.seed,
                    },
                )
                return response["message"]["content"]
            except Exception as e:
                last_error = e
                if not is_retryable(e):
                    logger.warning(f"Ollama 调用失败（不可重试）: {e}")
                    raise
                wait_and_log(e, attempt, self.max_retries, logger_name=__name__)

        raise RuntimeError(
            f"Ollama 调用失败，已重试 {self.max_retries} 次: {last_error}"
        )

    def _stream_impl(self, messages: list[dict[str, str]]) -> Iterator[str]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            stream = None
            yielded_any = False
            try:
                stream = self._client.chat(
                    model=self.model,
                    messages=messages,
                    options={
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "num_predict": self.max_tokens,
                        "num_ctx": self.num_ctx,
                        "repeat_penalty": self.repeat_penalty,
                        "seed": self.seed,
                    },
                    stream=True,
                )
                for chunk in stream:
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        yielded_any = True
                        yield content
                return
            except GeneratorExit:
                # 调用方中断：释放底层连接后继续抛出，由生成器框架处理
                raise
            except Exception as e:
                last_error = e
                if yielded_any:
                    logger.warning(
                        f"Ollama 流式中途失败（已输出内容，不重试）: {e}"
                    )
                    raise
                if not is_retryable(e):
                    logger.warning(f"Ollama 流式调用失败（不可重试）: {e}")
                    raise
                wait_and_log(e, attempt, self.max_retries, logger_name=__name__)
            finally:
                if stream is not None:
                    try:
                        # ollama SDK 迭代器没有 close()，但可通过 close 底层响应释放连接
                        close = getattr(stream, "close", None)
                        if callable(close):
                            close()
                        resp = getattr(stream, "_response", None) or getattr(stream, "response", None)
                        if resp is not None:
                            rc = getattr(resp, "close", None)
                            if callable(rc):
                                rc()
                    except Exception:
                        pass

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
