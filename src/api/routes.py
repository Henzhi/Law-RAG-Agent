"""
API 路由定义。
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from .dependencies import get_engine
from .models import ChatRequest, ChatResponse, HealthResponse

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
    """单次法律问答"""
    engine = get_engine()
    docs = engine.retriever.search(req.query, top_k=req.top_k)
    prompt = engine._build_prompt(req.query, docs)
    answer = engine.llm.chat(prompt)
    return ChatResponse.from_rag_answer(req.query, answer, docs)


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """流式法律问答"""
    import json
    engine = get_engine()
    docs = engine.retriever.search(req.query, top_k=req.top_k)
    prompt = engine._build_prompt(req.query, docs)

    def generate():
        # 先发 JSON 元数据行（引用来源）
        sources = [
            {
                "law_name": s.law_name,
                "chapter": s.chapter,
                "article_range": s.article_range,
                "citation": s.citation,
                "score": float(s.score),  # numpy.float32 → python float
            }
            for s in docs
        ]
        meta = json.dumps({"type": "meta", "sources": sources}, ensure_ascii=False)
        yield f"data: {meta}\n\n"

        # 流式输出答案
        for token in engine.llm.chat_stream(prompt):
            chunk = json.dumps({"type": "token", "content": token}, ensure_ascii=False)
            yield f"data: {chunk}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
