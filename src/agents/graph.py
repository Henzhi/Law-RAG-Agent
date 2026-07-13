"""
LangGraph 多 Agent 工作流。

流程:
    用户查询 → 查询改写 → 检索 → 生成回答 → 答案校验
                                      ↑              │
                                      └── 不通过 ────┘
"""
from __future__ import annotations

import logging
from typing import TypedDict, Annotated, Iterator

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from src.llm.client import LawLLM, Message as LLMMessage
from src.rag.retriever import BaseRetriever, RetrievedDoc
from src.rag.engine import RAG_PROMPT_TEMPLATE, is_casual_query, needs_retrieval

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    query: str                      # 原始用户查询
    rewritten_query: str            # 改写后的查询
    messages: Annotated[list, add_messages]  # 对话历史
    retrieved_docs: list[dict]      # 检索结果
    answer: str                     # 生成的回答
    validation_passed: bool         # 校验是否通过
    retry_count: int                # 重试次数


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

REWRITE_PROMPT = """你是一个法律查询改写助手。将用户的自然语言问题改写为更适合法律条文检索的查询。

## 规则
1. 保留原问题的核心法律概念
2. 补充不完整的法律名称（如"那个法"要补全）
3. 使用正式的法律术语
4. 只输出改写后的查询，不要解释

## 示例
用户: 打人怎么处罚 → 故意伤害他人的治安处罚标准
用户: 那个治安法里咋说的 → 治安管理处罚法相关规定
用户: 酒驾吊销驾照多久 → 饮酒后驾驶机动车驾驶证吊销期限

## 对话历史
{history}

## 用户问题
{query}

改写后的查询："""


VALIDATOR_PROMPT = """你是一个法律回答质量审核员。审核以下回答是否合格。

## 审核标准（宽松）
只要回答涉及法律内容、与检索到的条文相关、没有明显编造，就 PASS。
只有回答完全无关、明显胡说八道时才 FAIL。

## 检索到的条文
{context}

## LLM 生成的回答
{answer}

## 判定
只输出 PASS 或 FAIL，不要解释。"""


# ---------------------------------------------------------------------------
# 消息工具（兼容 dict 和 LangChain message 对象）
# ---------------------------------------------------------------------------

def _msg_role(m) -> str:
    """获取消息角色，兼容 dict 和 LangChain message 对象"""
    if hasattr(m, "type"):
        type_map = {"human": "user", "ai": "assistant", "system": "system"}
        return type_map.get(m.type, m.type or "user")
    if isinstance(m, dict):
        return m.get("role", "user")
    return "user"


def _msg_content(m) -> str:
    """获取消息内容，兼容 dict 和 LangChain message 对象"""
    if hasattr(m, "content"):
        return str(m.content) if m.content else ""
    if isinstance(m, dict):
        return str(m.get("content", ""))
    return str(m)


# ---------------------------------------------------------------------------
# 多 Agent 引擎
# ---------------------------------------------------------------------------

class LawAgentGraph:
    """LangGraph 多 Agent 法律问答引擎

    用法:
        agent = LawAgentGraph(retriever, llm)
        for token in agent.stream("行政拘留最长多久", history=[]):
            print(token, end="")
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: LawLLM,
        top_k: int = 5,
        max_retries: int = 1,
    ):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.max_retries = max_retries
        self._graph = self._build_graph()

    # ------------------------------------------------------------------
    # 图构建
    # ------------------------------------------------------------------

    def _build_graph(self) -> StateGraph:
        builder = StateGraph(AgentState)

        builder.add_node("rewrite", self._rewrite_query)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("generate", self._generate)
        builder.add_node("validate", self._validate)

        builder.set_entry_point("rewrite")
        builder.add_edge("rewrite", "retrieve")
        builder.add_edge("retrieve", "generate")
        builder.add_conditional_edges(
            "validate",
            self._should_retry,
            {"retry": "generate", "end": END},
        )
        builder.add_edge("generate", "validate")

        return builder.compile()

    # ------------------------------------------------------------------
    # 节点实现
    # ------------------------------------------------------------------

    def _rewrite_query(self, state: AgentState) -> dict:
        query = state["query"]
        history = state.get("messages", [])

        # 简短查询 + LLM 自省：不需要检索的查询跳过改写
        if len(query) <= 8 or not needs_retrieval(query, self.llm):
            return {"rewritten_query": query}

        # 构建历史文本
        hist_text = ""
        if history:
            recent = history[-6:]  # 最近 3 轮
            hist_text = "\n".join(
                f"{_msg_role(m)}: {str(_msg_content(m))[:200]}"
                for m in recent
            )

        prompt = REWRITE_PROMPT.format(query=query, history=hist_text or "（首次对话）")
        rewritten = self.llm.chat(prompt, system_prompt="你是一个法律查询改写助手，只输出改写后的查询。").strip()
        # 去掉可能的引号包裹
        rewritten = rewritten.strip('"').strip("'").strip()
        if not rewritten or len(rewritten) < 2:
            rewritten = query

        logger.info(f"查询改写: '{query}' → '{rewritten}'")
        return {"rewritten_query": rewritten}

    def _retrieve(self, state: AgentState) -> dict:
        q = state.get("rewritten_query", state["query"])
        docs = self.retriever.search(q, top_k=self.top_k)
        return {
            "retrieved_docs": [
                {"content": d.content, "law_name": d.law_name,
                 "article_range": d.article_range, "citation": d.citation}
                for d in docs
            ]
        }

    def _generate(self, state: AgentState) -> dict:
        docs = state.get("retrieved_docs", [])
        query = state.get("rewritten_query", state["query"])
        feedback = state.get("validation_feedback", "")

        ctx = _build_hierarchical_context(docs)

        # 重试时追加质量提醒
        extra = ""
        if feedback:
            extra = f"\n\n## ⚠️ 上次回答不合格\n原因: {feedback}\n请确保本次回答: 引用法律名称、标注条款号、不编造内容。"

        prompt = RAG_PROMPT_TEMPLATE.format(context=ctx, query=query) + extra

        # 附加历史（兼容 LangChain Message 和 dict）
        history = []
        for m in state.get("messages", [])[-6:]:
            role = _msg_role(m)
            content = _msg_content(m)[:300]
            if role in ("human", "ai", "user", "assistant"):
                role = "user" if role == "human" else "assistant" if role == "ai" else role
                history.append(LLMMessage(role, content))

        answer = self.llm.chat(prompt, history=history if history else None)
        return {"answer": answer}

    def _validate(self, state: AgentState) -> dict:
        answer = state.get("answer", "")
        docs = state.get("retrieved_docs", [])
        retry = state.get("retry_count", 0)

        # 无检索结果时直接通过
        if not docs:
            return {"validation_passed": True}

        # 构建简化上下文
        ctx = "\n".join(
            f"- {d.get('citation','')}: {d.get('content','')[:100]}"
            for d in docs[:5]
        )

        prompt = VALIDATOR_PROMPT.format(context=ctx, answer=answer[:800])
        result = self.llm.chat(prompt, system_prompt="你是一个审核员。只输出 PASS 或 FAIL。").strip().upper()

        passed = "PASS" in result
        if not passed and retry < self.max_retries:
            logger.info(f"校验未通过，重试 {retry + 1}/{self.max_retries}")
            return {"validation_passed": False, "retry_count": retry + 1}

        return {"validation_passed": True}

    def _should_retry(self, state: AgentState) -> str:
        if not state.get("validation_passed", True):
            return "retry"
        return "end"

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def ask(self, query: str, history: list[dict] | None = None) -> dict:
        """同步问答，返回完整 state"""
        initial = {
            "query": query,
            "messages": history or [],
            "rewritten_query": "",
            "retrieved_docs": [],
            "answer": "",
            "validation_passed": False,
            "retry_count": 0,
        }
        result = self._graph.invoke(initial)
        return result

    def stream(self, query: str, history: list[dict] | None = None) -> Iterator[dict]:
        """流式问答 - 手动步进 + LLM 真实流式输出"""
        yield {"type": "thinking", "content": "🔧 正在初始化 Agent..."}

        if not needs_retrieval(query, self.llm):
            yield {"type": "thinking", "content": "📝 直接回复，无需检索"}
            for token in self.llm.chat_stream(query):
                yield {"type": "token", "content": token}
            yield {"type": "thinking", "content": "✅ 完成"}
            return

        state = {
            "query": query, "messages": history or [], "rewritten_query": "",
            "retrieved_docs": [], "answer": "", "validation_passed": False,
            "retry_count": 0, "validation_feedback": "",
        }

        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                yield {"type": "clear", "content": ""}
                yield {"type": "thinking", "content": f"--- 第 {attempt + 1} 次尝试 ---"}

            # 1. Rewrite
            yield {"type": "thinking", "content": "⏳ 正在理解问题..."}
            state.update(self._rewrite_query(state))
            rw = state.get("rewritten_query", query)
            if rw != query:
                yield {"type": "thinking", "content": f"📝 查询改写: {rw}"}
            else:
                yield {"type": "thinking", "content": "📝 使用原始查询"}

            # 2. Retrieve
            yield {"type": "thinking", "content": "🔍 正在检索法律条文..."}
            state.update(self._retrieve(state))
            docs = state.get("retrieved_docs", [])
            yield {"type": "thinking", "content": f"📚 检索完成，找到 {len(docs)} 条相关条文"}
            if docs:
                citations = [d.get("citation", "") for d in docs[:5]]
                yield {"type": "thinking", "content": f"📖 引用: {', '.join(citations)}"}
            sources = [{"law_name": d.get("law_name", ""), "citation": d.get("citation", ""), "score": 0.0} for d in docs]
            yield {"type": "meta", "sources": sources, "is_casual": False, "rewritten": rw}

            # 3. Generate
            yield {"type": "thinking", "content": "💭 模型正在思考..."}
            fb = state.get("validation_feedback", "")
            ctx = _build_hierarchical_context(docs)
            extra = f"\n\n## ⚠️ 上次回答不合格\n原因: {fb}\n请确保本次回答: 引用法律名称、标注条款号、不编造内容。" if fb else ""
            prompt = RAG_PROMPT_TEMPLATE.format(context=ctx, query=rw) + extra
            hist = []
            for m in state.get("messages", [])[-6:]:
                r = _msg_role(m); c = _msg_content(m)[:300]
                if r in ("human", "ai", "user", "assistant"):
                    hist.append(LLMMessage("user" if r == "human" else "assistant" if r == "ai" else r, c))

            raw = ""
            for token in self.llm.chat_stream(prompt, history=hist if hist else None):
                raw += token

            import re
            think_match = re.search(r'<think>\s*(.*?)\s*</think>', raw, re.DOTALL)
            answer = raw
            if think_match:
                think_content = think_match.group(1).strip()
                if think_content:
                    yield {"type": "thinking", "content": think_content}
                answer = raw[think_match.end():].strip()

            state["answer"] = answer
            yield {"type": "thinking", "content": "💬 输出回答"}

            if answer:
                for i in range(0, len(answer), 4):
                    yield {"type": "token", "content": answer[i:i + 4]}

            # 4. Validate
            yield {"type": "thinking", "content": "🔎 审核回答质量..."}
            state.update(self._validate(state))
            if state.get("validation_passed", True):
                yield {"type": "thinking", "content": "✅ 审核通过"}
                break
            yield {"type": "thinking", "content": "❌ 未通过，重新生成..."}
            state["validation_feedback"] = "回答未引用法律名称或条款号"

        yield {"type": "thinking", "content": "✅ 全部完成"}


# ---------------------------------------------------------------------------
# 层级上下文构建（engine 和 agent 共用）
# ---------------------------------------------------------------------------

def _build_hierarchical_context(docs: list[dict]) -> str:
    """将检索结果按 (法律名, 章) 分组构建层级结构化上下文"""
    groups: dict[str, dict[str, list]] = {}  # law → chapter → [docs]
    seen = set()
    for doc in docs:
        law = doc.get("law_name", "")
        article = doc.get("article_range", "")
        key = (law, article)
        if key in seen:
            continue
        seen.add(key)
        chapter = doc.get("chapter", "") or "总则"
        groups.setdefault(law, {}).setdefault(chapter, []).append(doc)

    parts = []
    idx = 0
    for law_name, chapters in groups.items():
        for chapter, ch_docs in chapters.items():
            section = ch_docs[0].get("section", "") if ch_docs else ""
            if section:
                parts.append(f"## 《{law_name}》{chapter} → {section}")
            else:
                parts.append(f"## 《{law_name}》{chapter}")
            for doc in ch_docs:
                idx += 1
                content = doc.get("content", "")
                if "\n" in content and content.startswith("【"):
                    content = content.split("\n", 1)[1]
                parts.append(f"### {idx}. {doc.get('article_range', '')}\n{content.strip()}")

    return "\n\n".join(parts) if parts else "（未找到相关条文）"
