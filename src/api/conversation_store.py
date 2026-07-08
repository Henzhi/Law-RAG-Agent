"""对话持久化（pgvector PostgreSQL）"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import psycopg2
from src.config import PG_CONN


class ConversationStore:
    def __init__(self, conn_string: str = PG_CONN):
        self._conn = psycopg2.connect(conn_string)
        self._create_table()

    def _create_table(self):
        with self._conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT now()
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id, created_at)")
        self._conn.commit()

    def save_message(self, session_id: str, role: str, content: str):
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO conversations (session_id, role, content) VALUES (%s, %s, %s)",
                (session_id, role, content),
            )
        self._conn.commit()

    def load_history(self, session_id: str, limit: int = 50) -> list[dict]:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT role, content FROM conversations WHERE session_id = %s ORDER BY created_at DESC LIMIT %s",
                (session_id, limit),
            )
            rows = cur.fetchall()
        return [{"role": r, "content": c} for r, c in reversed(rows)]

    def list_sessions(self, limit: int = 20) -> list[dict]:
        with self._conn.cursor() as cur:
            cur.execute("""
                SELECT session_id, MIN(created_at) as started, COUNT(*) as msgs,
                       (SELECT content FROM conversations c2 WHERE c2.session_id = c.session_id ORDER BY created_at LIMIT 1) as first_msg
                FROM conversations c GROUP BY session_id ORDER BY started DESC LIMIT %s
            """, (limit,))
            rows = cur.fetchall()
        return [
            {"session_id": r[0], "started": r[1].isoformat(), "msg_count": r[2], "first_msg": (r[3] or "")[:50]}
            for r in rows
        ]

    def delete_session(self, session_id: str):
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM conversations WHERE session_id = %s", (session_id,))
        self._conn.commit()

    def close(self):
        self._conn.close()


# 全局单例
_store: ConversationStore | None = None


def get_conversation_store() -> ConversationStore:
    global _store
    if _store is None:
        _store = ConversationStore()
    return _store
