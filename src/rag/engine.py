"""
RAG 问答引擎。

串联完整管线：查询分类 → 闲聊直回 / 检索 → 构建 Prompt → LLM 回答
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator

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
]

CASUAL_SYSTEM_PROMPT = """你是一位友好的法律助手，同时也能进行日常交流。

## 回复原则
- 问候类：热情简洁地回应，并简要说明你可以帮助解答法律问题
- 感谢类：礼貌回应，鼓励继续提问
- 自我介绍：说明你是基于中国法律法规的智能问答助手，可以查询30多部法律
- 闲聊类：简短回应后，引导用户提出法律问题

请自然友好地回复。"""


def is_casual_query(query: str) -> bool:
    """快速正则判断是否为明显的闲聊/问候（用于响应元数据标记）"""
    q = query.strip().lower()
    if not q:
        return True
    for pattern in _CASUAL_PATTERNS:
        if re.match(pattern, q):
            return True
    return False


# ---------------------------------------------------------------------------
# LLM 自省路由：判断是否需要检索
# ---------------------------------------------------------------------------

ROUTE_PROMPT = """判断以下用户消息是否需要用法律知识库检索来回答。

## 规则
- YES: 涉及法律条文、法规、处罚、程序、权利等法律专业知识
- NO: 问候、感谢、告别、自我介绍、纯闲聊、日常对话

只输出 YES 或 NO，不要解释。

用户消息: {query}"""


def needs_retrieval(query: str, llm: LawLLM) -> bool:
    """LLM 自省：是否需要检索法律知识库？

    1. 正则命中 → 明确闲聊，不检索
    2. 问题超过 8 个字 → 大概率法律问题，直接检索（零延迟）
    3. 短模糊查询 → LLM 自省判断（正则误杀和真实闲聊的中间地带）
    """
    if is_casual_query(query):
        return False

    # 长查询大概率是正经问题，不走 LLM 路由省一次调用
    if len(query.strip()) > 8:
        return True

    # 短模糊查询：LLM 判断
    prompt = ROUTE_PROMPT.format(query=query)
    result = llm.chat(
        prompt,
        system_prompt="你是一个查询路由判断器。只输出 YES 或 NO。",
    ).strip().upper()

    if "NO" in result:
        return False
    return True


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
1. 引用法律名称和条款号
2. 基于条文内容，不编造
3. 条文不足时诚实说明
4. 回答简洁清晰

## 示例
用户: 治安处罚有哪些种类？
条文: 第十条: 治安管理处罚的种类分为：(一)警告；(二)罚款；(三)行政拘留；(四)吊销公安机关发放的许可证。
回答: 根据治安管理处罚法第十条，治安处罚共有四种：警告、罚款、行政拘留、吊销公安机关发放的许可证。

用户: 酒驾怎么处罚？
条文: 第九十一条: 饮酒后驾驶机动车的，处暂扣六个月机动车驾驶证，并处一千元以上二千元以下罚款。
回答: 根据道路交通安全法第九十一条，饮酒驾驶机动车，处暂扣六个月驾驶证，并处一千元以上二千元以下罚款。

---

## 相关法律条文
{context}

---

## 用户问题
{query}

---"""


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
        """单次问答，LLM 自省路由：闲聊直回 / 法律RAG"""

        # LLM 自省：是否需要检索？
        if not needs_retrieval(query, self.llm):
            answer = self.llm.chat(query, system_prompt=CASUAL_SYSTEM_PROMPT)
            return RAGAnswer(query=query, answer=answer, is_casual=True)

        # 法律 RAG
        docs = self.retriever.search(query, top_k=self.top_k)
        prompt = self._build_prompt(query, docs)
        answer = self.llm.chat(prompt)
        return RAGAnswer(query=query, answer=answer, sources=docs)

    def ask_stream(self, query: str) -> Iterator[str]:
        """流式问答，LLM 自省路由"""
        if not needs_retrieval(query, self.llm):
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
        """将检索结果格式化为 prompt 中的上下文，按法律+章节分组"""
        # 按 (法律名, 章) 分组，保留层次结构
        groups: dict[str, dict[str, list]] = {}  # law_name → chapter → [docs]
        seen = set()
        for doc in docs:
            key = (doc.law_name, doc.article_range)
            if key in seen:
                continue
            seen.add(key)
            chapter = doc.chapter or "总则"
            groups.setdefault(doc.law_name, {}).setdefault(chapter, []).append(doc)

        # 构建分组上下文
        parts = []
        idx = 0
        for law_name, chapters in groups.items():
            for chapter, ch_docs in chapters.items():
                # 章节头
                section = ch_docs[0].section if ch_docs and ch_docs[0].section else ""
                if section:
                    parts.append(f"## 《{law_name}》{chapter} → {section}")
                else:
                    parts.append(f"## 《{law_name}》{chapter}")
                for doc in ch_docs:
                    idx += 1
                    content = self._extract_core(doc)
                    parts.append(f"### {idx}. {doc.article_range}\n{content}")

        context = "\n\n".join(parts) if parts else "（未找到相关条文）"

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
