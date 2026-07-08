"""
API 路由定义。支持多轮对话（通过 history 字段）。
"""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from .dependencies import get_engine
from .models import ChatRequest, ChatResponse, HealthResponse
from src.rag.engine import is_casual_query
from src.llm.client import Message

router = APIRouter()


def _dicts_to_messages(history: list[dict]) -> list[Message]:
    """将前端传来的 [{role, content}] 转为 LLM Message 列表"""
    return [Message(msg["role"], msg["content"]) for msg in history if msg.get("content")]


@router.get("/health", response_model=HealthResponse)
async def health():
    """健康检查"""
    try:
        engine = get_engine()
        # 获取 doc_count: 如果是 RerankRetriever 则透传底层
        retriever = engine.retriever
        if hasattr(retriever, '_base'):
            doc_count = retriever._base._store.doc_count if hasattr(retriever._base, '_store') else 0
        else:
            doc_count = retriever._store.doc_count if hasattr(retriever, '_store') else 0
        return HealthResponse(
            status="ok",
            version="0.1.0",
            index_ready=retriever.is_ready(),
            doc_count=doc_count,
            llm_model=engine.llm.model_name,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """法律问答（支持多轮）"""
    import traceback
    try:
        engine = get_engine()
        history = _dicts_to_messages(req.history)

        # 闲聊：不检索
        if is_casual_query(req.query):
            answer = engine.llm.chat(req.query, history=history)
            return ChatResponse.from_rag_answer(query=req.query, answer=answer, sources=[], is_casual=True)

        # 法律 RAG
        docs = engine.retriever.search(req.query, top_k=req.top_k)
        prompt = engine._build_prompt(req.query, docs)
        answer = engine.llm.chat(prompt, history=history)
        return ChatResponse.from_rag_answer(query=req.query, answer=answer, sources=docs)
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """流式问答（支持多轮）"""
    engine = get_engine()
    history = _dicts_to_messages(req.history)
    casual = is_casual_query(req.query)

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
                 "article_range": s.article_range, "citation": s.citation,
                 "score": float(s.score)}
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
