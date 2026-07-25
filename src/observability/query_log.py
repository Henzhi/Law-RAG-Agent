"""
可观测性 — 检索质量日志 (v0.5)。

每次查询记录完整的性能指标和检索链路信息到 query_logs 表。
用于：性能瓶颈分析、检索质量追踪、高频问题发现、成本核算。
"""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)


class QueryLogger:
    """检索质量日志记录器

    用法:
        qlog = QueryLogger(conn_string)

        with qlog.trace("user_001", "工伤怎么认定") as trace:
            trace.stage("intent", 200)
            trace.stage("retrieve", 350)
            trace.stage("generate", 2100)
            trace.finalize(
                retrieved_count=15,
                reranked_count=5,
                faq_cache_hit=False,
                memory_docs_used=2,
                llm_tokens=1240,
            )
    """

    def __init__(self, conn_string: str):
        self._conn_string = conn_string

    @contextmanager
    def trace(self, user_id: str, query: str):
        """创建一次查询追踪上下文

        Usage:
            with qlog.trace(user_id, query) as trace:
                ...
        """
        trace = _QueryTrace(self._conn_string, user_id, query, str(uuid.uuid4()))
        trace._start_time = time.time()
        try:
            yield trace
        finally:
            if not trace._finalized:
                trace._save()


class _QueryTrace:
    """单次查询追踪器"""

    def __init__(self, conn_string: str, user_id: str, query: str, request_id: str):
        self._conn_string = conn_string
        self._user_id = user_id
        self._query = query
        self._request_id = request_id
        self._start_time = 0.0
        self._stages: dict[str, float] = {}
        self._intent = ""
        self._retrieved_count = 0
        self._reranked_count = 0
        self._faq_cache_hit = False
        self._memory_docs_used = 0
        self._llm_tokens = 0
        self._finalized = False

    def stage(self, name: str, duration_ms: int):
        self._stages[name] = float(duration_ms)

    def set_intent(self, intent: str):
        self._intent = intent

    def finalize(
        self,
        retrieved_count: int = 0,
        reranked_count: int = 0,
        faq_cache_hit: bool = False,
        memory_docs_used: int = 0,
        llm_tokens: int = 0,
    ):
        self._retrieved_count = retrieved_count
        self._reranked_count = reranked_count
        self._faq_cache_hit = faq_cache_hit
        self._memory_docs_used = memory_docs_used
        self._llm_tokens = llm_tokens
        self._save()
        self._finalized = True

    def _save(self):
        import json
        import psycopg2
        total_latency = int((time.time() - self._start_time) * 1000)

        try:
            conn = psycopg2.connect(self._conn_string)
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO query_logs "
                    "(request_id, user_id, query, intent, retrieved_count, reranked_count, "
                    " faq_cache_hit, memory_docs_used, llm_tokens_used, total_latency_ms, stage_timings) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (
                        self._request_id, self._user_id, self._query,
                        self._intent or "",
                        self._retrieved_count, self._reranked_count,
                        self._faq_cache_hit, self._memory_docs_used,
                        self._llm_tokens, total_latency,
                        json.dumps(self._stages, ensure_ascii=False) if self._stages else "{}",
                    ),
                )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"QueryLogger 写入失败: {e}")
