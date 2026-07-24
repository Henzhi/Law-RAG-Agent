"""
Agent 共享状态定义。

所有节点通过 TypedDict 约定的 state 进行通信。
LangGraph 的 add_messages reducer 自动合并多轮消息。
"""
from __future__ import annotations

from typing import TypedDict, Annotated

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """多 Agent 工作流共享状态

    各节点读取/写入对应的 key，LangGraph 自动处理状态传递。
    """
    query: str                      # 原始用户查询
    rewritten_query: str            # 改写后的查询（用于检索）
    messages: Annotated[list, add_messages]  # 当前会话对话历史
    retrieved_docs: list[dict]      # 检索结果 [{"content", "law_name", "article_range", "citation"}]
    answer: str                     # 生成的最终回答
    validation_passed: bool         # 校验是否通过
    validation_feedback: str        # 校验失败时的反馈信息（用于重试）
    retry_count: int                # 已重试次数
    is_legal_query: bool            # 意图识别：是否法律问题
    query_type: str                 # 查询类型: law_lookup | case_query | casual
    memory_context: str             # 历史对话记忆上下文（注入 Prompt）
    user_id: str                    # 用户 ID（用于记忆检索）
