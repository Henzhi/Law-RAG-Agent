"""
LangGraph 多 Agent 工作流。

流程:
    意图识别 → 闲聊? → 直接回复
              → 法律? → 查询改写 → 检索 → 生成回答 → 答案校验
                                                  ↑              │
                                                  └── 不通过 ────┘
"""
from __future__ import annotations

import logging
from typing import TypedDict, Annotated, Iterator

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from src.llm.client import LawLLM, Message as LLMMessage
from src.rag.retriever import BaseRetriever
from src.rag.engine import RAG_PROMPT_TEMPLATE

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
    is_legal_query: bool            # 意图识别：是否法律问题


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

检索到的法律条文：
{context}

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
# 意图识别
# ---------------------------------------------------------------------------

# 闲聊短语 — 精确匹配即跳过检索
_CASUAL_PHRASES = {
    # 问候
    "你好", "您好", "hi", "hello", "嗨", "哈喽",
    "早上好", "下午好", "晚上好", "中午好", "晚安", "早",
    "你是谁", "你叫什么", "你是什么", "你的名字",
    "介绍自己", "自我介绍", "你是啥",
    "在吗", "在不在", "在线吗",
    "你能做什么", "你会什么", "你有什么功能",
    "开始", "开始咨询", "测试", "test", "试试",
    # 感谢
    "谢谢", "感谢", "多谢", "thanks", "thank you",
    "非常感谢", "十分感谢", "万分感谢",
    # 告别
    "再见", "拜拜", "bye", "goodbye", "走了", "告辞",
}

# 法律关键词 — 包含任一即走检索
_LEGAL_KEYWORDS = [
    # 法律概念
    "法律", "法条", "法规", "条文", "条款", "规定（法律",
    # 处罚
    "处罚", "罚款", "拘留", "判刑", "刑期", "有期徒刑",
    "无期徒刑", "死刑", "拘役", "管制", "没收", "吊销",
    # 责任赔偿
    "赔偿", "责任", "侵权", "违约", "损害", "损失",
    # 权利
    "权利", "义务", "隐私", "名誉", "肖像", "人身",
    # 法律关系
    "合同", "协议", "婚姻", "离婚", "继承", "遗嘱",
    "收养", "抚养", "赡养", "劳动", "社保", "工伤",
    # 诉讼
    "诉讼", "仲裁", "起诉", "上诉", "判决", "裁定", "执行",
    "证据", "时效", "管辖", "法院",
    # 犯罪
    "犯罪", "罪名", "故意", "过失", "自首", "累犯",
    "盗窃", "诈骗", "抢劫", "伤害", "杀人",
    # 法律名称简称
    "民法典", "刑法", "宪法", "公司法", "劳动法",
    "治安管理", "道路交通", "行政法", "刑事法",
    # 法律问句模式
    "怎么罚", "判多久", "合法吗", "违法吗", "要不要赔",
    "能告吗", "算不算", "有没有责任",
]


def _normalize(text: str) -> str:
    """标准化：去标点、去空格、小写"""
    import re
    return re.sub(r'[^\w\u4e00-\u9fff]', '', text.lower().strip())


def classify_intent(query: str) -> bool:
    """意图识别：是否为法律相关问题？

    1. 标准化后精确匹配闲聊短语 → 闲聊
    2. 标准化后包含法律关键词 → 法律
    3. 短查询（≤4字）精确匹配闲聊 → 闲聊（二次检查）
    4. 都不匹配 → 默认检索（宁可多检）
    """
    q = query.strip()
    nq = _normalize(q)

    # 1. 精确匹配闲聊短语
    for phrase in _CASUAL_PHRASES:
        if _normalize(phrase) == nq:
            return False

    # 2. 包含法律关键词
    for kw in _LEGAL_KEYWORDS:
        if _normalize(kw) in nq:
            return True

    # 3. 短查询二次检查：包含闲聊短语
    if len(nq) <= 4:
        for phrase in _CASUAL_PHRASES:
            if _normalize(phrase) in nq:
                return False

    # 4. 默认走检索
    return True


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

        builder.add_node("intent", self._classify_intent)
        builder.add_node("casual_reply", self._casual_reply)
        builder.add_node("rewrite", self._rewrite_query)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("generate", self._generate)
        builder.add_node("validate", self._validate)

        builder.set_entry_point("intent")
        # 意图 → 闲聊 / 法律
        builder.add_conditional_edges(
            "intent", self._route_by_intent,
            {"legal": "rewrite", "casual": "casual_reply"},
        )
        builder.add_edge("casual_reply", END)
        # 法律路径
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

    def _classify_intent(self, state: AgentState) -> dict:
        """意图识别节点：判断是闲聊还是法律问题"""
        is_legal = classify_intent(state["query"])
        logger.info(f"意图识别: '{state['query']}' → {'法律' if is_legal else '闲聊'}")
        return {"is_legal_query": is_legal}

    def _route_by_intent(self, state: AgentState) -> str:
        """根据意图路由到不同分支"""
        return "legal" if state.get("is_legal_query", True) else "casual"

    def _casual_reply(self, state: AgentState) -> dict:
        """闲聊直接回复，不走检索"""
        from src.rag.engine import CASUAL_SYSTEM_PROMPT
        answer = self.llm.chat(state["query"], system_prompt=CASUAL_SYSTEM_PROMPT)
        return {"answer": answer, "validation_passed": True}

    def _rewrite_query(self, state: AgentState) -> dict:
        query = state["query"]
        history = state.get("messages", [])

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

        ctx = "\n".join(
            f"- {d.get('citation','')}: {d.get('content','')[:120]}"
            for d in docs[:5]
        )
        prompt = VALIDATOR_PROMPT.format(query=query, context=ctx, answer=answer[:800])
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
            "is_legal_query": True,
        }
        result = self._graph.invoke(initial)
        return result

    def stream(self, query: str, history: list[dict] | None = None) -> Iterator[dict]:
        """流式问答 - 手动步进 + LLM 真实流式输出"""
        yield {"type": "thinking", "content": "🔧 正在初始化 Agent..."}

        # 1. 意图识别
        is_legal = classify_intent(query)
        yield {"type": "thinking", "content": f"🎯 意图识别: {'法律问题 → 检索法条' if is_legal else '闲聊 → 直接回复'}"}

        if not is_legal:
            yield {"type": "thinking", "content": "📝 直接回复，无需检索"}
            for token in self.llm.chat_stream(query):
                yield {"type": "token", "content": token}
            yield {"type": "thinking", "content": "✅ 完成"}
            return

        state = {
            "query": query, "messages": history or [], "rewritten_query": "",
            "retrieved_docs": [], "answer": "", "validation_passed": False,
            "retry_count": 0, "validation_feedback": "", "is_legal_query": True,
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

            # 3. Generate — LLM 直接流式输出回答
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

            answer_raw = ""
            for token in self.llm.chat_stream(prompt, history=hist if hist else None):
                yield {"type": "token", "content": token}
                answer_raw += token
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
