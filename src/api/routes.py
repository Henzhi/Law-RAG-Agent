"""
API 路由定义。支持多轮对话 + LangGraph Agent。
"""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from .dependencies import get_engine, get_agent
from .models import ChatRequest, ChatResponse, HealthResponse
from src.config import AGENT_ENABLED
from src.rag.engine import is_casual_query
from src.llm.client import Message

router = APIRouter()


def _dicts_to_messages(history: list[dict]) -> list[Message]:
    return [Message(msg["role"], msg["content"]) for msg in history if msg.get("content")]


@router.get("/health", response_model=HealthResponse)
async def health():
    try:
        eng = get_engine() if not AGENT_ENABLED else get_agent()
        if hasattr(eng, "retriever"):
            r = eng.retriever
            if hasattr(r, "_base"):
                doc_count = r._base._store.doc_count if hasattr(r._base, "_store") else 0
            else:
                doc_count = r._store.doc_count if hasattr(r, "_store") else 0
        else:
            doc_count = 0
        return HealthResponse(
            status="ok", version="0.1.0",
            index_ready=eng.retriever.is_ready() if hasattr(eng, "retriever") else True,
            doc_count=doc_count,
            llm_model="qwen2.5:7b",
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if AGENT_ENABLED:
        agent = get_agent()
        result = agent.ask(req.query, history=req.history)
        return ChatResponse.from_rag_answer(
            query=result["query"], answer=result["answer"],
            sources=_dicts_to_retrieved(result.get("retrieved_docs", [])),
            is_casual=is_casual_query(req.query),
        )

    engine = get_engine()
    history = _dicts_to_messages(req.history)
    if is_casual_query(req.query):
        answer = engine.llm.chat(req.query, history=history)
        return ChatResponse.from_rag_answer(query=req.query, answer=answer, sources=[], is_casual=True)

    docs = engine.retriever.search(req.query, top_k=req.top_k)
    prompt = engine._build_prompt(req.query, docs)
    answer = engine.llm.chat(prompt, history=history)
    return ChatResponse.from_rag_answer(query=req.query, answer=answer, sources=docs)


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    casual = is_casual_query(req.query)

    if AGENT_ENABLED and not casual:
        agent = get_agent()

        def generate():
            q = req.query
            # 查询改写 (agent 内部)
            state = {"query": q, "messages": req.history, "rewritten_query": "", "retrieved_docs": [], "answer": "", "validation_passed": False, "retry_count": 0}
            after_rw = agent._rewrite_query(state)
            rw_q = after_rw.get("rewritten_query", q)

            # 检索
            after_ret = agent._retrieve({"query": q, "rewritten_query": rw_q, "messages": req.history})
            docs = after_ret.get("retrieved_docs", [])

            sources = [
                {"law_name": d.get("law_name", ""), "chapter": d.get("chapter", ""),
                 "article_range": d.get("article_range", ""), "citation": d.get("citation", ""),
                 "score": 0.0}
                for d in docs
            ]
            meta = json.dumps({"type": "meta", "sources": sources, "is_casual": False, "rewritten": rw_q}, ensure_ascii=False)
            yield f"data: {meta}\n\n"

            # 流式输出
            for token in agent.stream(req.query, history=req.history):
                chunk = json.dumps({"type": "token", "content": token}, ensure_ascii=False)
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
    else:
        engine = get_engine()
        history = _dicts_to_messages(req.history)

        if casual:
            def generate():
                meta = json.dumps({"type": "meta", "sources": [], "is_casual": True}, ensure_ascii=False)
                yield f"data: {meta}\n\n"
                for token in engine.llm.chat_stream(req.query, history=history):
                    chunk = json.dumps({"type": "token", "content": token}, ensure_ascii=False)
                    yield f"data: {chunk}\n\n"
                yield "data: [DONE]\n\n"
        else:
            docs = engine.retriever.search(req.query, top_k=engine.top_k)
            prompt = engine._build_prompt(req.query, docs)

            def generate():
                sources = [
                    {"law_name": s.law_name, "chapter": s.chapter,
                     "article_range": s.article_range, "citation": s.citation, "score": float(s.score)}
                    for s in docs
                ]
                meta = json.dumps({"type": "meta", "sources": sources, "is_casual": False}, ensure_ascii=False)
                yield f"data: {meta}\n\n"
                for token in engine.llm.chat_stream(prompt, history=history):
                    chunk = json.dumps({"type": "token", "content": token}, ensure_ascii=False)
                    yield f"data: {chunk}\n\n"
                yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ------------------------------------------------------------------
# 对话持久化
# ------------------------------------------------------------------

@router.get("/conversations")
def list_conversations():
    """列出最近的对话会话"""
    from .conversation_store import get_conversation_store
    store = get_conversation_store()
    return store.list_sessions()


@router.get("/conversations/{session_id}")
def get_conversation(session_id: str):
    """加载指定会话的对话历史"""
    from .conversation_store import get_conversation_store
    store = get_conversation_store()
    return {"session_id": session_id, "history": store.load_history(session_id)}


@router.post("/conversations/{session_id}")
def save_message(session_id: str, msg: dict):
    """保存一条消息到会话"""
    from .conversation_store import get_conversation_store
    store = get_conversation_store()
    store.save_message(session_id, msg.get("role", "user"), msg.get("content", ""))
    return {"ok": True}


def _dicts_to_retrieved(docs: list[dict]) -> list:
    """将 agent 返回的 dict 转为 RetrievedDoc 兼容格式"""
    result = []
    for d in docs:
        result.append(type("RetrievedDoc", (), {
            "law_name": d.get("law_name", ""),
            "chapter": d.get("chapter", ""),
            "section": d.get("section", ""),
            "article_range": d.get("article_range", ""),
            "citation": d.get("citation", ""),
            "content": d.get("content", ""),
            "score": float(d.get("score", 0)),
        })())
    return result
