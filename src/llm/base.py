"""
LLM 后端抽象基类。

定义统一的 LLM 调用接口，支持 Ollama 和 OpenAI 兼容 API 两种后端。
所有后端实现必须继承此基类并实现对应方法。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator


class LLMBackend(ABC):
    """LLM 后端抽象基类

    子类需实现:
      - _generate_impl(): 同步生成
      - _stream_impl(): 流式生成
      - context_window: 返回上下文窗口大小
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 2048,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def chat(
        self,
        user_message: str,
        history: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """单轮对话（同步）

        Args:
            user_message: 用户消息
            history: 历史对话 [{"role": "...", "content": "..."}, ...]
            system_prompt: 系统提示词

        Returns:
            LLM 响应文本
        """
        messages = self._build_messages(user_message, history, system_prompt)
        return self._generate_impl(messages)

    def chat_stream(
        self,
        user_message: str,
        history: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
    ) -> Iterator[str]:
        """流式对话

        Yields:
            逐个 token 的输出文本
        """
        messages = self._build_messages(user_message, history, system_prompt)
        yield from self._stream_impl(messages)

    # ------------------------------------------------------------------
    # 抽象方法（子类必须实现）
    # ------------------------------------------------------------------

    @abstractmethod
    def _generate_impl(self, messages: list[dict[str, str]]) -> str:
        """同步生成实现"""
        ...

    @abstractmethod
    def _stream_impl(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """流式生成实现"""
        ...

    @abstractmethod
    def get_context_window(self) -> int:
        """返回该模型的有效上下文窗口大小 (tokens)"""
        ...

    # ------------------------------------------------------------------
    # 公共工具方法
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        user_message: str,
        history: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
    ) -> list[dict[str, str]]:
        """构建标准消息列表"""
        messages: list[dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if history:
            for msg in history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role in ("system", "user", "assistant"):
                    messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": user_message})
        return messages

    @staticmethod
    def _build_rag_prompt(query: str, context: str) -> str:
        """构建 RAG 问答 prompt"""
        return f"""请根据以下法律条文回答用户的问题。

## 相关法律条文
{context}

## 用户问题
{query}

## 要求
1. 回答中必须引用具体的法律条文（注明法律名称和条款号）
2. 如果条文中没有直接答案，指出现有条文的规定和相关联的情况
3. 保持回答简洁，不要凭空编造"""
