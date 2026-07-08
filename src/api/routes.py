"""
API 路由定义。
"""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from .dependencies import get_engine
from .models import ChatRequest, ChatResponse, HealthResponse
from src.rag.engine import is_casual_query

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health():
    """健康检查"""
    try:
        engine = get_engine()
        return HealthResponse(
            status="ok",
            version="0.1.0",
            index_ready=engine.retriever.is_ready(),
            doc_count=engine.retriever._store.doc_count if engine.retriever._store else 0,
            llm_model=engine.llm.model_name,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """单次法律问答（自动区分闲聊/法律）"""
    engine = get_engine()
    result = engine.ask(req.query)
    return ChatResponse.from_rag_answer(
        query=result.query,
        answer=result.answer,
        sources=result.sources,
        is_casual=result.is_casual,
    )


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """流式问答（自动区分闲聊/法律）"""
    engine = get_engine()

    # 判断是否为闲聊
    casual = is_casual_query(req.query)

    if casual:
        # 闲聊：不检索，直接流式回复
        def generate():
            meta = json.dumps({
                "type": "meta",
                "sources": [],
                "is_casual": True,
            }, ensure_ascii=False)
            yield f"data: {meta}\n\n"

            for token in engine.llm.chat_stream(
                req.query,
                system_prompt=engine._system_prompt if hasattr(engine, '_system_prompt') else None
            ):
                chunk = json.dumps({"type": "token", "content": token}, ensure_ascii=False)
                yield f"data: {chunk}\n\n"

            yield "data: [DONE]\n\n"
    else:
        # 法律：检索 + 流式回复
        docs = engine.retriever.search(req.query, top_k=engine.top_k)
        prompt = engine._build_prompt(req.query, docs)

        def generate():
            sources = [
                {
                    "law_name": s.law_name,
                    "chapter": s.chapter,
                    "article_range": s.article_range,
                    "citation": s.citation,
                    "score": float(s.score),
                }
                for s in docs
            ]
            meta = json.dumps({
                "type": "meta",
                "sources": sources,
                "is_casual": False,
            }, ensure_ascii=False)
            yield f"data: {meta}\n\n"

            for token in engine.llm.chat_stream(prompt):
                chunk = json.dumps({"type": "token", "content": token}, ensure_ascii=False)
                yield f"data: {chunk}\n\n"

            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
