"""
对话记忆管理器 (v0.5)

实现跨会话的记忆存储和检索:
  - 摘要生成: 对话 > 6 轮时，LLM 自动生成结构化摘要 + 关键实体提取
  - 存储: 摘要向量写入 pgvector conversation_memories 表，TTL 30 天
  - 检索: 新问题时语义检索 Top-3 历史摘要，时间衰减排序
  - 注入: 结果拼入 Prompt [历史参考] 段，让 LLM 带着记忆回答

用法:
    mgr = ConversationMemoryManager(store, embedder, llm)
    mgr.save_memory(user_id, session_id, messages)   # 对话结束时调用
    summaries = mgr.retrieve(user_id, query)          # 新问题时调用
"""
from __future__ import annotations

import logging
from typing import List

from src.knowledge.pgvector_store import PgvectorStore

logger = logging.getLogger(__name__)

# 摘要生成 Prompt
_SUMMARY_PROMPT = """请将以下法律咨询对话总结为一段结构化摘要，用于后续记忆检索。

## 对话内容
{conversation}

## 摘要格式（严格按此结构输出，每个字段一行）
案件类型: <继承纠纷/合同纠纷/劳动争议/刑事/婚姻/行政/其他>
涉及法律: <列举涉及的法律名称，逗号分隔>
关键事实: <用户描述的核心案情，1-2句话>
已回答问题: <系统已经回答了什么>
未解决问题: <还有什么待回答>
"""

# 触发条件：对话轮数阈值
SUMMARY_TRIGGER_ROUNDS = 6

# 检索参数
DEFAULT_TOP_K = 3
TIME_DECAY_DAYS = 7  # 7 天内的记忆权重不变，之后线性衰减


class ConversationMemoryManager:
    """对话记忆管理器

    Attributes:
        _store: pgvector 存储实例
        _embedder: Embedding 适配器
        _llm: LLM 适配器（用于生成摘要）
    """

    def __init__(
        self,
        store: PgvectorStore,
        embedder,   # EmbeddingAdapter
        llm,        # LLMAdapter
    ):
        self._store = store
        self._embedder = embedder
        self._llm = llm

    # ------------------------------------------------------------------
    # 记忆写入
    # ------------------------------------------------------------------

    def should_summarize(self, message_count: int) -> bool:
        """判断是否需要生成摘要（>6 轮时触发）"""
        return message_count >= SUMMARY_TRIGGER_ROUNDS

    def save_memory(
        self,
        user_id: str,
        session_id: str,
        messages: list[dict],
    ) -> str | None:
        """保存对话记忆

        1. 检查是否达到触发条件
        2. LLM 生成结构化摘要
        3. 提取关键实体
        4. embedding → 写入 pgvector

        Args:
            user_id: 用户 ID
            session_id: 会话 ID
            messages: 完整对话消息 [{"role": ..., "content": ...}, ...]

        Returns:
            摘要文本（用于日志），如果未触发则返回 None
        """
        if len(messages) < SUMMARY_TRIGGER_ROUNDS:
            return None

        # 拼对话文本
        conv_text = self._format_conversation(messages)

        # LLM 生成摘要
        summary = self._llm.chat(conv_text, system_prompt=_SUMMARY_PROMPT)

        # 解析实体
        entities = self._parse_entities(summary)

        # 向量化
        summary_vec = self._embedder.embed_query(summary)

        # 写入 pgvector
        self._store._ensure_connection()
        with self._store._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO conversation_memories "
                "(user_id, session_id, summary, summary_embed, entities, message_count) "
                "VALUES (%s, %s, %s, %s::halfvec, %s, %s)",
                (
                    user_id,
                    session_id,
                    summary,
                    summary_vec,
                    entities,
                    len(messages),
                ),
            )
        self._store._conn.commit()
        logger.info(f"对话记忆已保存: user={user_id[:8]}..., session={session_id[:8]}..., msg_count={len(messages)}")
        return summary

    # ------------------------------------------------------------------
    # 记忆检索
    # ------------------------------------------------------------------

    def retrieve(
        self,
        user_id: str,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[dict]:
        """检索与当前问题最相关的历史对话摘要

        1. embedding 查询
        2. pgvector 余弦相似度检索（仅查该用户的记忆）
        3. 时间衰减排序

        Returns:
            [{"summary", "entities", "score", "message_count", "created_at"}, ...]
        """
        query_vec = self._embedder.embed_query(query)

        self._store._ensure_connection()
        with self._store._conn.cursor() as cur:
            cur.execute(
                "SELECT summary, entities, message_count, created_at, "
                "1 - (summary_embed <=> %s::halfvec) AS score "
                "FROM conversation_memories "
                "WHERE user_id = %s "
                "  AND expires_at > NOW() "
                "ORDER BY summary_embed <=> %s::halfvec "
                "LIMIT %s",
                (query_vec, user_id, query_vec, top_k),
            )
            rows = cur.fetchall()

        if not rows:
            return []

        import datetime
        now = datetime.datetime.now(datetime.timezone.utc)
        results = []
        for row in rows:
            summary, entities, msg_count, created_at, score = row
            # 时间衰减
            if created_at:
                age_days = (now - created_at).days
                if age_days > TIME_DECAY_DAYS:
                    decay = TIME_DECAY_DAYS / age_days
                    score = score * decay

            results.append({
                "summary": summary,
                "entities": entities or {},
                "score": round(float(score), 4),
                "message_count": msg_count,
                "created_at": created_at.isoformat() if created_at else None,
            })

        # 按衰减后分数重排
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def build_context(self, memories: list[dict]) -> str:
        """将检索到的记忆组装为 Prompt 上下文片段"""
        if not memories:
            return ""

        lines = ["## 历史对话参考（用户之前咨询过的相关内容）"]
        for i, m in enumerate(memories, 1):
            lines.append(f"### 历史对话 {i}（相关度: {m['score']:.2f}）")
            lines.append(m["summary"])
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _format_conversation(messages: list[dict]) -> str:
        """将消息列表格式化为对话文本"""
        lines = []
        for msg in messages:
            role = "用户" if msg.get("role") == "user" else "AI助手"
            content = msg.get("content", "")
            # 截断过长消息
            if len(content) > 500:
                content = content[:500] + "..."
            lines.append(f"{role}: {content}")
        return "\n\n".join(lines)

    @staticmethod
    def _parse_entities(summary: str) -> dict:
        """从摘要中提取关键实体"""
        entities = {}
        for line in summary.split("\n"):
            line = line.strip()
            if line.startswith("案件类型:"):
                entities["case_type"] = line.replace("案件类型:", "").strip()
            elif line.startswith("涉及法律:"):
                entities["laws_involved"] = line.replace("涉及法律:", "").strip()
            elif line.startswith("关键事实:"):
                entities["key_facts"] = line.replace("关键事实:", "").strip()
        return entities
