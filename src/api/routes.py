"""
API 路由定义。支持多轮对话 + LangGraph Agent + 用户会话隔离。
"""
from __future__ import annotations

import json
import time
import logging

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
perf_logger = logging.getLogger("api.perf")


def _dicts_to_messages(history: list[dict]) -> list[Message]:
    return [Message(msg["role"], msg["content"]) for msg in history if msg.get("content")]


@router.get("/health", response_model=HealthResponse)
async def health():
    try:
        from src.config import LLM_MODEL
        eng = get_engine() if not AGENT_ENABLED else get_agent()

        # 遍历检索器链找到最内层的 FAISS/PG retriever
        doc_count = 0
        index_ready = True
        retriever = getattr(eng, "retriever", None)
        if retriever:
            index_ready = retriever.is_ready()
            # 穿透装饰器链: AdjacentExpander → Reranker → Hybrid → FAISS
            chain = retriever
            while hasattr(chain, "_base"):
                chain = chain._base
            if hasattr(chain, "_store"):
                doc_count = getattr(chain._store, "doc_count", 0)

        return HealthResponse(
            status="ok", version="0.1.0",
            index_ready=index_ready,
            doc_count=doc_count,
            llm_model=LLM_MODEL,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    t_start = time.perf_counter()
    try:
        if AGENT_ENABLED:
            agent = get_agent()
            result = agent.ask(req.query, history=req.history)
            elapsed = (time.perf_counter() - t_start) * 1000
            ret_docs = result.get("retrieved_docs", [])
            perf_logger.info(
                f"[chat] mode=agent query_len={len(req.query)} "
                f"legal={result.get('is_legal_query', True)} "
                f"retrieved={len(ret_docs)} elapsed={elapsed:.0f}ms"
            )
            return ChatResponse.from_rag_answer(
                query=result["query"], answer=result["answer"],
                sources=_dicts_to_retrieved(ret_docs),
                is_casual=not result.get("is_legal_query", True),
            )

        engine = get_engine()
        history = _dicts_to_messages(req.history)

        t_route = time.perf_counter()
        if not needs_retrieval(req.query, engine.llm):
            answer = engine.llm.chat(req.query, history=history)
            elapsed = (time.perf_counter() - t_start) * 1000
            perf_logger.info(
                f"[chat] mode=casual query_len={len(req.query)} "
                f"route_ms={(time.perf_counter()-t_route)*1000:.0f} elapsed={elapsed:.0f}ms"
            )
            return ChatResponse.from_rag_answer(query=req.query, answer=answer, sources=[], is_casual=True)

        t_ret = time.perf_counter()
        docs = engine.retriever.search(req.query, top_k=req.top_k)
        ret_ms = (time.perf_counter() - t_ret) * 1000

        t_llm = time.perf_counter()
        prompt = engine._build_prompt(req.query, docs)
        answer = engine.llm.chat(prompt, history=history)
        llm_ms = (time.perf_counter() - t_llm) * 1000

        elapsed = (time.perf_counter() - t_start) * 1000
        top_score = round(docs[0].score, 4) if docs else 0
        perf_logger.info(
            f"[chat] mode=rag query_len={len(req.query)} "
            f"retrieved={len(docs)} top_score={top_score} "
            f"ret_ms={ret_ms:.0f} llm_ms={llm_ms:.0f} elapsed={elapsed:.0f}ms"
        )
        return ChatResponse.from_rag_answer(query=req.query, answer=answer, sources=docs)
    except HTTPException:
        raise
    except Exception as e:
        elapsed = (time.perf_counter() - t_start) * 1000
        perf_logger.error(f"[chat] error={type(e).__name__} elapsed={elapsed:.0f}ms")
        raise HTTPException(status_code=500, detail=f"处理请求失败: {str(e)}")


def _sse(data: dict) -> str:
    """将 dict 序列化为 SSE 格式的一行"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    t_start = time.perf_counter()
    if AGENT_ENABLED:
        agent = get_agent()

        def generate():
            try:
                for event in agent.stream(req.query, history=req.history):
                    yield _sse(event)
            except Exception as e:
                elapsed = (time.perf_counter() - t_start) * 1000
                perf_logger.error(f"[stream] mode=agent error={type(e).__name__} elapsed={elapsed:.0f}ms")
                yield _sse({"type": "error", "content": f"处理失败: {str(e)}"})
            elapsed = (time.perf_counter() - t_start) * 1000
            perf_logger.info(f"[stream] mode=agent query_len={len(req.query)} elapsed={elapsed:.0f}ms")
            yield "data: [DONE]\n\n"
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ---- 非 Agent 路径：统一用一个 generate() 发出 thinking 事件 ----
    engine = get_engine()
    history = _dicts_to_messages(req.history)

    def generate():
        try:
            yield _sse({"type": "thinking", "content": "正在分析问题..."})
            casual = not needs_retrieval(req.query, engine.llm)
            yield _sse({"type": "thinking", "content": f"意图识别: {'闲聊 → 直接回复' if casual else '法律问题 → 检索法条'}"})

            if casual:
                yield _sse({"type": "meta", "sources": [], "is_casual": True})
                yield _sse({"type": "thinking", "content": "直接回复，无需检索"})
                for token in engine.llm.chat_stream(req.query, history=history):
                    yield _sse({"type": "token", "content": token})
                yield _sse({"type": "thinking", "content": "完成"})
                return

            t_ret = time.perf_counter()
            yield _sse({"type": "thinking", "content": "正在检索法律条文..."})
            docs = engine.retriever.search(req.query, top_k=engine.top_k)
            ret_ms = (time.perf_counter() - t_ret) * 1000
            prompt = engine._build_prompt(req.query, docs)
            top_score = round(docs[0].score, 4) if docs else 0
            perf_logger.info(
                f"[stream] mode=rag retrieved={len(docs)} top_score={top_score} ret_ms={ret_ms:.0f}ms"
            )
            yield _sse({"type": "thinking", "content": f"检索完成，找到 {len(docs)} 条相关条文"})
            if docs:
                citations = [f"{d.law_name} {d.article_range}" for d in docs[:5]]
                yield _sse({"type": "thinking", "content": f"引用: {', '.join(citations)}"})

            sources = [
                {"law_name": s.law_name, "chapter": s.chapter,
                 "article_range": s.article_range, "citation": s.citation, "score": float(s.score)}
                for s in docs
            ]
            yield _sse({"type": "meta", "sources": sources, "is_casual": False})
            yield _sse({"type": "thinking", "content": "模型正在生成回答..."})
            for token in engine.llm.chat_stream(prompt, history=history):
                yield _sse({"type": "token", "content": token})

        except Exception as e:
            elapsed = (time.perf_counter() - t_start) * 1000
            perf_logger.error(f"[stream] error={type(e).__name__} elapsed={elapsed:.0f}ms")
            yield _sse({"type": "error", "content": f"处理失败: {str(e)}"})

        elapsed = (time.perf_counter() - t_start) * 1000
        perf_logger.info(f"[stream] query_len={len(req.query)} elapsed={elapsed:.0f}ms")
        yield _sse({"type": "thinking", "content": "全部完成"})
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
