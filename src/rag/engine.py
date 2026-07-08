"""
RAG 问答引擎。

串联完整管线：查询分类 → 闲聊直回 / 检索 → 构建 Prompt → LLM 回答
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator, Optional

from src.llm.client import LawLLM, Message as LLMMessage
from .retriever import BaseRetriever, RetrievedDoc


# ---------------------------------------------------------------------------
# 查询分类
# ---------------------------------------------------------------------------

# 闲聊类关键词/模式 — 匹配后直接走 LLM 回复，跳过检索
_CASUAL_PATTERNS = [
    # 问候
    r'^(你好|您好|hi|hello|嗨|早上好|下午好|晚上好|大家好)',
    r'^(在吗|在不|在不在)$',
    # 感谢
    r'^(谢谢|感谢|多谢|thanks|thank)',
    # 告别
    r'^(再见|拜拜|bye|晚安|回头见)',
    # 自我介绍
    r'^(你是谁|你叫什么|你是什么|你的名字|介绍.*自己)',
    r'^(你能做什么|你会什么|你有什么功能|你能干什么)',
    # 纯闲聊
    r'^(今天天气|天气怎么样|讲个笑话|说个笑话)',
    r'^(嗯|哦|好吧|好的|OK|ok)$',
    # 短问候
    r'^.{1,4}$',  # 1-4字符的超短消息
]

CASUAL_SYSTEM_PROMPT = """你是一位友好的法律助手，同时也能进行日常交流。

## 回复原则
- 问候类：热情简洁地回应，并简要说明你可以帮助解答法律问题
- 感谢类：礼貌回应，鼓励继续提问
- 自我介绍：说明你是基于中国法律法规的智能问答助手，可以查询30多部法律
- 闲聊类：简短回应后，引导用户提出法律问题

请自然友好地回复。"""


def is_casual_query(query: str) -> bool:
    """判断是否为闲聊/问候类查询（无需检索）"""
    q = query.strip().lower()
    if not q:
        return True
    for pattern in _CASUAL_PATTERNS:
        if re.match(pattern, q):
            return True
    return False


# ---------------------------------------------------------------------------
# 问答结果
# ---------------------------------------------------------------------------

@dataclass
class RAGAnswer:
    """一次 RAG 问答的完整结果"""
    query: str
    answer: str
    sources: list[RetrievedDoc] = field(default_factory=list)
    is_casual: bool = False

    def format_sources(self) -> str:
        """格式化引用来源"""
        if self.is_casual:
            return "  （闲聊模式，无引用）"
        lines = []
        seen = set()
        for doc in self.sources:
            key = (doc.law_name, doc.article_range)
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"  - {doc.citation}")
        return "\n".join(lines) if lines else "  无引用来源"


# ---------------------------------------------------------------------------
# RAG 提示词模板
# ---------------------------------------------------------------------------

RAG_PROMPT_TEMPLATE = """你是一位精通中国法律的专业法律助手。请根据以下提供的法律条文，准确回答用户的问题。

## 要求
1. **必须引用条文**：回答中必须标明引用的法律名称和具体条款号（如「治安管理处罚法 第十条」）
2. **以条文为准**：答案必须基于提供的条文内容，不要编造
3. **条文未覆盖时**：如果提供的条文不足以回答，诚实说明并给出最相关的条文作为参考
4. **简洁清晰**：先给结论，再列依据

---

## 相关法律条文
{context}

---

## 用户问题
{query}

---

请回答："""


# ---------------------------------------------------------------------------
# RAG 引擎
# ---------------------------------------------------------------------------

class RAGEngine:
    """RAG 问答引擎：检索 + LLM 回答"""

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: LawLLM,
        top_k: int = 5,
        prompt_template: str = RAG_PROMPT_TEMPLATE,
    ):
        """
        Args:
            retriever: 检索器（FAISS 或 pgvector）
            llm: LLM 客户端
            top_k: 每次检索返回的文档数
            prompt_template: 自定义提示词模板
        """
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.prompt_template = prompt_template

    # ------------------------------------------------------------------
    # 问答
    # ------------------------------------------------------------------

    def ask(self, query: str) -> RAGAnswer:
        """单次问答，自动分类：闲聊直回 / 法律RAG"""

        # 闲聊：跳过检索，直接 LLM 回复
        if is_casual_query(query):
            answer = self.llm.chat(query, system_prompt=CASUAL_SYSTEM_PROMPT)
            return RAGAnswer(query=query, answer=answer, is_casual=True)

        # 法律 RAG
        docs = self.retriever.search(query, top_k=self.top_k)
        prompt = self._build_prompt(query, docs)
        answer = self.llm.chat(prompt)
        return RAGAnswer(query=query, answer=answer, sources=docs)

    def ask_stream(self, query: str) -> Iterator[str]:
        """流式问答，自动分类"""
        if is_casual_query(query):
            yield from self.llm.chat_stream(query, system_prompt=CASUAL_SYSTEM_PROMPT)
            return

        docs = self.retriever.search(query, top_k=self.top_k)
        prompt = self._build_prompt(query, docs)
        yield from self.llm.chat_stream(prompt)

    # ------------------------------------------------------------------
    # 多轮对话
    # ------------------------------------------------------------------

    def chat(
        self,
        query: str,
        history: list[LLMMessage] | None = None,
    ) -> RAGAnswer:
        """多轮对话（带历史，每次重新检索）

        Args:
            query: 当前问题
            history: 历史消息

        Returns:
            RAGAnswer
        """
        docs = self.retriever.search(query, top_k=self.top_k)
        prompt = self._build_prompt(query, docs)

        history = history or []
        answer = self.llm.chat(prompt, history=history)

        return RAGAnswer(query=query, answer=answer, sources=docs)

    def chat_stream(
        self,
        query: str,
        history: list[LLMMessage] | None = None,
    ) -> Iterator[str]:
        """多轮流式对话"""
        docs = self.retriever.search(query, top_k=self.top_k)
        prompt = self._build_prompt(query, docs)
        history = history or []
        yield from self.llm.chat_stream(prompt, history=history)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _build_prompt(self, query: str, docs: list[RetrievedDoc]) -> str:
        """将检索结果格式化为 prompt 中的上下文"""
        context_parts = []
        seen = set()

        for i, doc in enumerate(docs, 1):
            # 去重：同一法条只保留一次
            key = (doc.law_name, doc.article_range)
            if key in seen:
                continue
            seen.add(key)

            # 构建带引用的上下文条目
            header = f"### {i}. {doc.citation}"
            # 取 core content（去掉 chapter_summary 的冗余部分）
            content = self._extract_core(doc)
            context_parts.append(f"{header}\n{content}")

        context = "\n\n".join(context_parts) if context_parts else "（未找到相关条文）"

        return self.prompt_template.format(context=context, query=query)

    @staticmethod
    def _extract_core(doc: RetrievedDoc) -> str:
        """从 chunk 内容中提取核心文本（去掉前缀元数据）"""
        content = doc.content
        # chapter_summary chunk 的格式是 "【法律名】／章\n第一条: xxx\n第二条: xxx"
        # article chunk 的格式是 "【法律名】／章\n条文内容"
        # 去掉第一行前缀，保留实质内容
        if "\n" in content and content.startswith("【"):
            content = content.split("\n", 1)[1]
        return content.strip()


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------

def create_rag_engine(
    vector_store,
    llm: LawLLM,
    top_k: int = 5,
) -> RAGEngine:
    """快速创建 RAG 引擎（FAISS 检索器）"""
    from .retriever import FAISSRetriever

    retriever = FAISSRetriever(vector_store)
    return RAGEngine(retriever=retriever, llm=llm, top_k=top_k)
