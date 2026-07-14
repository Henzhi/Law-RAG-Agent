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


VALIDATOR_PROMPT = """
你是一个严格的法律回答幻觉审核器。你的唯一任务是审查 AI 生成的法律回答是否存在**事实性幻觉**，包括：

- 虚构、篡改或引用不存在的法律条文、司法解释、案例
- 将已失效的法律当作现行有效法律使用
- 法律原则、法律后果、程序规则等出现根本性错误
- 关键事实假设毫无根据，严重误导用户

**审核标准**：
- 仅关注**客观、可验证的法律事实错误**，不评判表达风格、详略程度、主观建议是否最优。
- 只要回答中包含任何一处确定的幻觉或重大法律错误，即判定为 **FAIL**。
- 如果回答没有幻觉和重大错误（即使不够全面、有遗漏），判定为 **PASS**。

**输入格式**：
用户问题：
{query}

AI 回答：
{answer}

**输出要求**：
只输出以下格式，不要添加额外解释或礼貌用语：

`PASS` 或 `FAIL`
理由：（一句话，指出具体哪一点错误，若无则写“未发现幻觉”）

---

**示例**

输入：
用户问题：借条上没写还款日期，诉讼时效怎么算？
AI 回答：根据《民法典》第188条，诉讼时效为3年，从您知道权利受侵害之日起计算。如果借条没有还款日期，诉讼时效从您第一次要账时起算。

输出：
PASS
理由：未发现幻觉

---

输入：
用户问题：试用期单位不交社保合法吗？
AI 回答：根据《劳动合同法》第17条，试用期属于劳动合同的协商条款，单位可以不交社保。

输出：
FAIL
理由：虚构试用期可不交社保的规定，与实际法律要求不符
"""


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
        query = state.get("query", "")

        # 无检索结果时直接通过
        if not docs:
            return {"validation_passed": True}

        prompt = VALIDATOR_PROMPT.format(query=query, answer=answer[:800])
        result = self.llm.chat(prompt, system_prompt="你是一个法律回答审核员。").strip().upper()

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

    # ------------------------------------------------------------------
    # 流式生成（实时解析 <think> 标签）
    # ------------------------------------------------------------------
    THINK_START = "<think>"
    THINK_END = "</think>"

    @staticmethod
    def _tag_partial(s: str, tag: str) -> bool:
        """检查字符串 s 的末尾是否为 tag 的部分前缀（如 '<th' 是 '<think>' 的前缀）"""
        for i in range(1, len(tag)):
            if s.endswith(tag[:i]):
                return True
        return False

    def _stream_generate(self, prompt: str, history: list) -> Iterator[dict]:
        """流式调用 LLM，实时解析 <think> 标签并 yield token/thinking 事件

        状态机：
            OUTSIDE — 寻找 <think>，中间文本作为 thinking 输出
            INSIDE  — 收集思考内容，直到 </think>
            ANSWER  — </think> 之后的内容，作为 token 实时输出

        输出粒度：按 LLM token 而不是逐字输出，避免前端一行一个字。
        """
        OUTSIDE, INSIDE, ANSWER = 0, 1, 2
        state = OUTSIDE
        buf = ""         # 字符缓冲区，用于标签检测
        out = ""         # 输出缓冲区，积累后再 yield
        answer_raw = ""  # 累积完整回答（不含 think），供校验使用
        think_raw = ""   # 累积所有思考内容，用于兜底时重新作为 token 输出

        def _flush(th: bool = False):
            """刷新输出缓冲区"""
            nonlocal out, answer_raw, think_raw
            if out:
                if th:
                    think_raw += out
                else:
                    answer_raw += out  # 先累加，不依赖 yield 后 resume
                if th:
                    yield {"type": "thinking", "content": out}
                else:
                    yield {"type": "token", "content": out}
                out = ""

        for token in self.llm.chat_stream(prompt, history=history if history else None):
            for ch in token:
                buf += ch

                if state == OUTSIDE:
                    if self.THINK_START in buf:
                        idx = buf.index(self.THINK_START)
                        before = buf[:idx]
                        if before.strip():
                            out += before
                        buf = buf[idx + len(self.THINK_START):]
                        state = INSIDE
                    elif not self._tag_partial(buf, self.THINK_START):
                        out += buf
                        buf = ""
                    # 否则等待更多字符拼完整标签

                elif state == INSIDE:
                    if self.THINK_END in buf:
                        idx = buf.index(self.THINK_END)
                        out += buf[:idx]
                        buf = buf[idx + len(self.THINK_END):]
                        state = ANSWER
                        # 刷新思考内容
                        yield from _flush(th=True)
                        yield {"type": "thinking", "content": "💬 输出回答"}
                        if buf:
                            out += buf
                            buf = ""
                    elif not self._tag_partial(buf, self.THINK_END):
                        out += buf
                        buf = ""
                    # 否则等待更多字符拼完整标签

                else:  # ANSWER
                    if not self._tag_partial(buf, self.THINK_START):
                        out += buf
                        buf = ""
                    # 理论上答完后不会再出现 <think>，但防御一下

            # 每个 LLM token 处理完后，刷新输出缓冲区
            if state == INSIDE:
                yield from _flush(th=True)
            elif state == ANSWER:
                yield from _flush(th=False)
            # OUTSIDE 状态不急于输出，等 <think> 或积累一批再输出

        # 切换状态时已在 _flush 中输出，这里只处理残留
        if out:
            yield from _flush(th=(state != ANSWER))

        # 缓冲区残留
        if buf:
            if state == ANSWER:
                yield {"type": "token", "content": buf}
                answer_raw += buf
            else:
                yield {"type": "thinking", "content": buf}
                think_raw += buf

        # 兜底：模型未关闭 </think>，所有内容被吞进思考框
        # → 把 think_raw 重新作为 token 事件发出去，前端 answer 区域就有内容了
        if state != ANSWER and think_raw and not answer_raw:
            yield {"type": "thinking", "content": "⚠️ 模型未输出 </think>，已将思考内容作为回答"}
            yield {"type": "token", "content": think_raw}
            answer_raw = think_raw

        return answer_raw

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

            # 3. Generate — 真流式，token 逐个到达即输出
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

            answer_raw = yield from self._stream_generate(prompt, hist)
            state["answer"] = answer_raw.strip() or "(未能生成回答)"

            # 4. Validate（需要完整答案，在流式输出之后进行）
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
