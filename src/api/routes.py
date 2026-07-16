"""
API 路由定义。支持多轮对话 + LangGraph Agent + 用户会话隔离。
"""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse

from .dependencies import get_engine, get_agent
from .models import ChatRequest, ChatResponse, HealthResponse, RegisterRequest, LoginRequest, AuthResponse
from .auth import get_current_user, register_user, login_user
from src.config import AGENT_ENABLED
from src.rag.engine import needs_retrieval
from src.llm.client import Message

router = APIRouter()
auth_router = APIRouter()


def _dicts_to_messages(history: list[dict]) -> list[Message]:
    return [Message(msg["role"], msg["content"]) for msg in history if msg.get("content")]


@router.get("/health", response_model=HealthResponse)
async def health():
    try:
        from src.config import LLM_MODEL
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
            llm_model=LLM_MODEL,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        if AGENT_ENABLED:
            # Agent 图内部有 intent → casual_reply/rewrite 路由
            agent = get_agent()
            result = agent.ask(req.query, history=req.history)
            return ChatResponse.from_rag_answer(
                query=result["query"], answer=result["answer"],
                sources=_dicts_to_retrieved(result.get("retrieved_docs", [])),
                is_casual=not result.get("is_legal_query", True),
            )

        engine = get_engine()
        history = _dicts_to_messages(req.history)
        if not needs_retrieval(req.query, engine.llm):
            answer = engine.llm.chat(req.query, history=history)
            return ChatResponse.from_rag_answer(query=req.query, answer=answer, sources=[], is_casual=True)

        docs = engine.retriever.search(req.query, top_k=req.top_k)
        prompt = engine._build_prompt(req.query, docs)
        answer = engine.llm.chat(prompt, history=history)
        return ChatResponse.from_rag_answer(query=req.query, answer=answer, sources=docs)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理请求失败: {str(e)}")


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    if AGENT_ENABLED:
        # Agent.stream() 内部有意图识别，自己处理闲聊/法律路由
        agent = get_agent()

        def generate():
            try:
                for event in agent.stream(req.query, history=req.history):
                    chunk = json.dumps(event, ensure_ascii=False)
                    yield f"data: {chunk}\n\n"
            except Exception as e:
                err = json.dumps({"type": "error", "content": f"处理失败: {str(e)}"}, ensure_ascii=False)
                yield f"data: {err}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    else:
        engine = get_engine()
        history = _dicts_to_messages(req.history)
        casual = not needs_retrieval(req.query, engine.llm)
        if casual:
            def generate():
                try:
                    meta = json.dumps({"type": "meta", "sources": [], "is_casual": True}, ensure_ascii=False)
                    yield f"data: {meta}\n\n"
                    for token in engine.llm.chat_stream(req.query, history=history):
                        chunk = json.dumps({"type": "token", "content": token}, ensure_ascii=False)
                        yield f"data: {chunk}\n\n"
                except Exception as e:
                    err = json.dumps({"type": "error", "content": f"处理失败: {str(e)}"}, ensure_ascii=False)
                    yield f"data: {err}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        docs = engine.retriever.search(req.query, top_k=engine.top_k)
        prompt = engine._build_prompt(req.query, docs)

        def generate():
            try:
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
            except Exception as e:
                err = json.dumps({"type": "error", "content": f"处理失败: {str(e)}"}, ensure_ascii=False)
                yield f"data: {err}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ------------------------------------------------------------------
# 对话持久化（全部按 user_id 隔离）
# ------------------------------------------------------------------

@router.get("/conversations")
def list_conversations(user_id: str = Depends(get_current_user)):
    """列出当前用户的对话会话"""
    from .conversation_store import get_conversation_store
    store = get_conversation_store()
    return store.list_sessions(user_id=user_id)


@router.get("/conversations/{session_id}")
def get_conversation(session_id: str, user_id: str = Depends(get_current_user)):
    """加载指定会话的对话历史（仅限当前用户）"""
    from .conversation_store import get_conversation_store
    store = get_conversation_store()
    history = store.load_history(user_id=user_id, session_id=session_id)
    return {"session_id": session_id, "history": history}


@router.post("/conversations/{session_id}")
def save_session(session_id: str, body: dict, user_id: str = Depends(get_current_user)):
    """保存整个会话的 JSON 消息数组（每次整体覆盖，不逐条插入）"""
    from .conversation_store import get_conversation_store
    store = get_conversation_store()
    messages = body.get("messages", [])
    store.save_session(user_id=user_id, session_id=session_id, messages=messages)
    return {"ok": True}


# ------------------------------------------------------------------
# 认证路由
# ------------------------------------------------------------------

@auth_router.post("/register", response_model=AuthResponse)
def register(req: RegisterRequest):
    """注册新用户（需要用户名+密码），返回 Bearer Token"""
    return register_user(username=req.username, password=req.password)


@auth_router.post("/login", response_model=AuthResponse)
def login(req: LoginRequest):
    """用用户名+密码登录，返回 Bearer Token"""
    return login_user(username=req.username, password=req.password)


@auth_router.get("/me")
def get_me(user_id: str = Depends(get_current_user)):
    """获取当前用户信息"""
    from .auth import ANONYMOUS_USER_ID
    is_anonymous = user_id == ANONYMOUS_USER_ID
    return {"user_id": user_id, "anonymous": is_anonymous}


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
