"""
LangGraph 多 Agent 工作流编排 (v0.5)。

职责单一：构建和编译 StateGraph，将节点连接成工作流。
节点实现 → agents/nodes.py
状态定义 → agents/state.py
提示词   → agents/prompts.py

流程:
    intent → FAQ缓存检查 → memory_retrieve → rewrite → retrieve → generate → validate
                ↓ 命中返回    ↑ 新增节点                             ↓
                              └── 检索历史对话摘要   ├─ PASS → 存缓存 → END
                                                    └─ FAIL → generate (重试)
"""
from __future__ import annotations

import logging
from typing import Iterator

from langgraph.graph import StateGraph, END

from src.agents.state import AgentState
from src.agents.nodes import make_nodes, build_hierarchical_context, _msg_role, _msg_content
from src.rag.retriever import BaseRetriever
from src.rag.engine import RAG_PROMPT_TEMPLATE
from src.rag.intent import classify_intent, classify_query_type
from src.llm.client import Message as LLMMessage

logger = logging.getLogger(__name__)


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
        llm,                    # LLMAdapter
        top_k: int = 5,
        max_retries: int = 1,
        memory_manager = None,  # ConversationMemoryManager | None
        faq_cache = None,       # FAQCache | None
    ):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.max_retries = max_retries
        self._memory = memory_manager
        self._faq_cache = faq_cache

        # 通过工厂函数注入依赖，节点本身无状态
        nodes = make_nodes(llm, retriever, memory_manager, top_k, max_retries)
        self._nodes = nodes
        self._graph = self._build_graph(nodes)

    # ------------------------------------------------------------------
    # 图构建
    # ------------------------------------------------------------------

    def _build_graph(self, nodes: dict) -> StateGraph:
        builder = StateGraph(AgentState)

        builder.add_node("intent", nodes["classify_intent"])
        builder.add_node("casual_reply", nodes["casual_reply"])
        builder.add_node("memory_retrieve", nodes["memory_retrieve"])
        builder.add_node("rewrite", nodes["rewrite_query"])
        builder.add_node("retrieve", nodes["retrieve"])
        builder.add_node("generate", nodes["generate"])
        builder.add_node("validate", nodes["validate"])

        builder.set_entry_point("intent")
        builder.add_conditional_edges(
            "intent", nodes["route_by_intent"],
            {"legal": "memory_retrieve", "casual": "casual_reply"},
        )
        builder.add_edge("casual_reply", END)
        builder.add_edge("memory_retrieve", "rewrite")
        builder.add_edge("rewrite", "retrieve")
        builder.add_edge("retrieve", "generate")
        builder.add_conditional_edges(
            "validate", nodes["should_retry"],
            {"retry": "generate", "end": END},
        )
        builder.add_edge("generate", "validate")

        return builder.compile()

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def ask(self, query: str, history: list[dict] | None = None, user_id: str = "") -> dict:
        """同步问答 — 含 FAQ 缓存检查（与 stream() 路径行为一致）"""
        query_type = classify_query_type(query)

        # FAQ 缓存检查
        if self._faq_cache:
            try:
                cached = self._faq_cache.check(query)
                if cached:
                    logger.info(f"ask() FAQ缓存命中: score={cached['score']}")
                    return {
                        "query": query, "answer": cached["answer"],
                        "retrieved_docs": [], "is_legal_query": True,
                        "cached": True,
                    }
            except Exception as e:
                logger.warning(f"FAQ缓存检查失败: {e}")

        initial: AgentState = {
            "query": query,
            "messages": history or [],
            "rewritten_query": "",
            "retrieved_docs": [],
            "answer": "",
            "validation_passed": False,
            "validation_feedback": "",
            "retry_count": 0,
            "is_legal_query": query_type != "casual",
            "query_type": query_type,
            "memory_context": "",
            "user_id": user_id,
        }
        return self._graph.invoke(initial)

    def stream(self, query: str, history: list[dict] | None = None, user_id: str = "") -> Iterator[dict]:
        """流式问答 - 手动步进 + LLM 真实流式输出"""
        yield {"type": "thinking", "content": "🔧 正在初始化 Agent..."}

        # 1. 意图识别（三分类）
        query_type = classify_query_type(query)
        is_legal = query_type != "casual"
        type_label = {"law_lookup": "法律条文查询", "case_query": "案例检索", "casual": "闲聊"}
        yield {"type": "thinking", "content": f"🎯 意图识别: {type_label.get(query_type, query_type)}"}

        if not is_legal:
            yield {"type": "thinking", "content": "📝 直接回复，无需检索"}
            for token in self.llm.chat_stream(query):
                yield {"type": "token", "content": token}
            yield {"type": "thinking", "content": "✅ 完成"}
            return

        # 2. FAQ 缓存检查（命中则直接返回，未命中继续 RAG 流程）
        if self._faq_cache:
            yield {"type": "thinking", "content": "⚡ 检查 FAQ 缓存..."}
            try:
                cached = self._faq_cache.check(query)
                if cached:
                    yield {"type": "FAQ", "content": f"⚡ FAQ 缓存命中 (相似度: {cached['score']:.3f})"}
                    yield {"type": "token", "content": cached["answer"]}
                    yield {"type": "meta", "sources": cached.get("sources", []), "is_casual": False, "cache_hit": True}
                    yield {"type": "thinking", "content": "✅ 完成（来自缓存）"}
                    return
            except Exception as e:
                logger.warning(f"FAQ缓存检查失败: {e}")

        state: dict = {
            "query": query, "messages": history or [], "rewritten_query": "",
            "retrieved_docs": [], "answer": "", "validation_passed": False,
            "retry_count": 0, "validation_feedback": "", "is_legal_query": True,
            "query_type": query_type, "memory_context": "", "user_id": user_id,
        }

        # 3. 记忆检索
        if self._memory and user_id:
            yield {"type": "thinking", "content": "🧠 检索历史记忆..."}
            try:
                memories = self._memory.retrieve(user_id, query)
                ctx = self._memory.build_context(memories)
                if ctx:
                    state["memory_context"] = ctx
                    yield {"type": "thinking", "content": f"🧠 找到 {len(memories)} 条相关历史记忆"}
            except Exception as e:
                logger.warning(f"流式: 记忆检索失败: {e}")

        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                yield {"type": "clear", "content": ""}
                yield {"type": "thinking", "content": f"--- 第 {attempt + 1} 次尝试 ---"}

            # 4. Rewrite
            yield {"type": "thinking", "content": "⏳ 正在理解问题..."}
            state.update(self._nodes["rewrite_query"](state))
            rw = state.get("rewritten_query", query)
            if rw != query:
                yield {"type": "thinking", "content": f"📝 查询改写: {rw}"}
            else:
                yield {"type": "thinking", "content": "📝 使用原始查询"}

            # 5. Retrieve（按意图路由文档类型）
            yield {"type": "thinking", "content": "🔍 正在检索法律条文..."}
            state.update(self._nodes["retrieve"](state))
            docs = state.get("retrieved_docs", [])
            type_hint = "案例" if query_type == "case_query" else "条文"
            yield {"type": "thinking", "content": f"📚 检索完成，找到 {len(docs)} 条相关{type_hint}"}
            if docs:
                citations = [d.get("citation", "") for d in docs[:5]]
                yield {"type": "thinking", "content": f"📖 引用: {', '.join(citations)}"}
            sources = [{"law_name": d.get("law_name", ""), "citation": d.get("citation", ""), "score": 0.0} for d in docs]
            yield {"type": "meta", "sources": sources, "is_casual": False, "rewritten": rw}

            # 6. Generate
            yield {"type": "thinking", "content": "💭 模型正在思考..."}
            fb = state.get("validation_feedback", "")
            memory_ctx = state.get("memory_context", "")
            ctx = build_hierarchical_context(docs)
            if memory_ctx:
                ctx = memory_ctx + "\n\n" + ctx
            extra = f"\n\n## ⚠️ 上次回答不合格\n原因: {fb}\n请确保本次回答: 引用法律名称、标注条款号、不编造内容。" if fb else ""
            prompt = RAG_PROMPT_TEMPLATE.format(context=ctx, query=rw) + extra

            hist = []
            for m in state.get("messages", [])[-6:]:
                r = _msg_role(m); c = _msg_content(m)[:300]
                if r in ("human", "ai", "user", "assistant"):
                    hist.append(LLMMessage("user" if r == "human" else "assistant" if r == "ai" else r, c))

            answer_raw = ""
            for token in self.llm.chat_stream(prompt, history=hist if hist else None):
                yield {"type": "token", "content": token}
                answer_raw += token
            state["answer"] = answer_raw.strip() or "(未能生成回答)"

            # 7. Validate
            yield {"type": "thinking", "content": "🔎 审核回答质量..."}
            state.update(self._nodes["validate"](state))
            if state.get("validation_passed", True):
                yield {"type": "thinking", "content": "✅ 审核通过"}
                # 校验通过 → 存入 FAQ 缓存
                if self._faq_cache:
                    try:
                        related_laws = list(set(d.get("law_name", "") for d in docs if d.get("law_name")))
                        self._faq_cache.store(
                            question=query,
                            answer=state["answer"],
                            sources=sources,
                            related_laws=related_laws,
                            confidence=0.9,
                        )
                    except Exception as e:
                        logger.warning(f"FAQ缓存写入失败: {e}")
                break
            fb = state.get("validation_feedback", "")
            yield {"type": "thinking", "content": f"❌ 未通过{f': {fb}' if fb else ''}，重新生成..."}

        yield {"type": "thinking", "content": "✅ 全部完成"}


# _msg_role / _msg_content 统一从 nodes 模块导入（单一来源，见文件顶部 import）
