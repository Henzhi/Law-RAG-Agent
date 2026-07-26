"""
API 路由定义。支持多轮对话 + LangGraph Agent + 用户会话隔离。
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import StreamingResponse

from .dependencies import get_engine, get_agent, _create_embedder
from .models import ChatRequest, ChatResponse, HealthResponse, RegisterRequest, LoginRequest, AuthResponse, CrawlRequest, CrawlTaskResponse, CrawlStatusResponse
from .auth import get_current_user, register_user, login_user
from src.config import AGENT_ENABLED
from src.rag.engine import needs_retrieval
from src.rag.intent import sanitize_input
from src.llm.client import Message

router = APIRouter()
auth_router = APIRouter()
perf_logger = logging.getLogger("api.perf")
logger = logging.getLogger(__name__)


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

    # 输入安全过滤（Prompt 注入 + 敏感内容检测）
    safe_query, is_safe, reject_reason = sanitize_input(req.query)
    if not is_safe:
        perf_logger.warning(f"[chat] blocked: reason={reject_reason} query_preview={req.query[:100]}")
        return ChatResponse(query=req.query, answer=safe_query, sources=[], is_casual=True)

    try:
        if AGENT_ENABLED:
            agent = get_agent()
            result = agent.ask(safe_query, history=req.history)
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

    # 输入安全过滤（Prompt 注入 + 敏感内容检测）
    safe_query, is_safe, reject_reason = sanitize_input(req.query)
    if not is_safe:
        async def _reject_stream():
            yield _sse({"type": "error", "content": safe_query})
            yield "data: [DONE]\n\n"
        return StreamingResponse(
            _reject_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    if AGENT_ENABLED:
        agent = get_agent()

        def generate():
            try:
                for event in agent.stream(safe_query, history=req.history):
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


@router.delete("/conversations/{session_id}")
def delete_session(session_id: str, user_id: str = Depends(get_current_user)):
    """删除指定会话"""
    from .conversation_store import get_conversation_store
    store = get_conversation_store()
    store.delete_session(user_id=user_id, session_id=session_id)
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


# ---------------------------------------------------------------------------
# 4. 知识库 — 文档上传
# ---------------------------------------------------------------------------
# 解析管道单例（任务状态跨请求共享）
_ingestion_pipeline: object | None = None


def _get_ingestion_pipeline():
    """获取解析管道单例"""
    global _ingestion_pipeline
    if _ingestion_pipeline is None:
        embedder = _create_embedder()
        from src.knowledge.pgvector_store import PgvectorStore
        from src.config import PG_CONN as _pg_conn
        store = PgvectorStore(_pg_conn)
        store.ensure_tables()
        from src.knowledge.ingestion.pipeline import IngestionPipeline
        _ingestion_pipeline = IngestionPipeline(store, embedder)
    return _ingestion_pipeline


@router.post("/knowledge/upload")
async def upload_document(
    file: UploadFile = File(...),
    doc_type: str = Form("law"),
    source: str = Form(""),
    effective_date: str = Form(""),
):
    """上传法律文档（PDF/DOCX/TXT）

    文件被保存到临时目录后由解析管道处理，
    返回 task_id 用于查询处理进度。
    """
    import tempfile
    import asyncio

    # 验证文件扩展名
    ext = Path(file.filename or "").suffix.lower()
    allowed = {".pdf", ".docx", ".txt"}
    if ext not in allowed:
        raise HTTPException(400, f"不支持的文件格式: {ext}，支持: {', '.join(allowed)}")

    # 检查文件大小
    content = await file.read()
    max_size = 50 * 1024 * 1024  # 50MB
    if len(content) > max_size:
        raise HTTPException(400, f"文件过大: {len(content) / 1024 / 1024:.1f}MB（限制 50MB）")

    # 保存到临时文件
    suffix = ext if ext else ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    # 提交解析任务
    pipeline = _get_ingestion_pipeline()
    task_id = pipeline.submit(
        file_path=tmp_path,
        doc_type=doc_type,
        source=source,
        effective_date=effective_date or None,
    )

    # 后台异步处理 — to_thread 避免同步解析阻塞事件循环
    asyncio.create_task(asyncio.to_thread(_run_ingestion_sync, pipeline, task_id, tmp_path))

    return {
        "task_id": task_id,
        "filename": file.filename,
        "doc_type": doc_type,
        "status": "pending",
        "message": f"文档 {file.filename} 已提交解析",
    }


def _run_ingestion_sync(pipeline, task_id: str, tmp_path: str):
    """后台同步执行解析任务（运行在 asyncio.to_thread 线程中）"""
    import os
    try:
        chunk_count = pipeline.run(task_id)
        logger.info(f"后台解析完成: task={task_id[:8]}..., chunks={chunk_count}")
    except Exception as e:
        logger.error(f"后台解析失败: task={task_id[:8]}..., error={e}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@router.get("/knowledge/status/{task_id}")
async def get_ingestion_status(task_id: str):
    """查询文档解析任务状态"""
    pipeline = _get_ingestion_pipeline()
    status = pipeline.get_status(task_id)
    if status is None:
        raise HTTPException(404, "任务不存在")
    return status


# ---------------------------------------------------------------------------
# 5. 知识库 — 文档管理
# ---------------------------------------------------------------------------

def _get_store():
    """获取 pgvector store 单例"""
    from src.knowledge.pgvector_store import PgvectorStore
    from src.config import PG_CONN as _pg_conn
    store = PgvectorStore(_pg_conn)
    store.ensure_tables()
    return store


@router.get("/knowledge/documents")
def list_knowledge_documents(doc_type: str | None = None):
    """列出知识库中的所有文档

    Query:
        doc_type: 按类型过滤 (law/interpretation/case/regulation)，不传则返回全部
    """
    store = _get_store()
    docs = store.list_documents(doc_type=doc_type)
    return {"documents": docs, "total": len(docs)}


@router.delete("/knowledge/documents/{doc_id}")
def delete_knowledge_document(doc_id: str):
    """删除文档及其所有向量块"""
    store = _get_store()
    ok = store.delete_document(doc_id)
    if not ok:
        raise HTTPException(404, "文档不存在")
    # 删除后重建索引
    store.reindex()
    return {"ok": True, "message": f"文档 {doc_id[:8]}... 已删除"}


@router.get("/knowledge/documents/{doc_id}/chunks")
def get_document_chunks(doc_id: str):
    """获取文档的所有文本块"""
    store = _get_store()
    chunks = store.get_document_chunks(doc_id)
    if not chunks:
        raise HTTPException(404, "文档不存在或无内容")
    return {"doc_id": doc_id, "chunks": chunks, "total": len(chunks)}


# ----------------------------------------------------------------------------
# 6. 爬虫 — 国家法律法规数据库增量爬取
# ----------------------------------------------------------------------------
_crawl_tasks: dict[str, dict] = {}


@router.post("/crawl", response_model=CrawlTaskResponse)
async def crawl_laws(req: CrawlRequest):
    """触发爬取（后台任务）。

    数据源现仅支持 npc（全国人大「国家法律法规数据库」）。任务提交后返回
    task_id，通过 GET /api/crawl/status/{task_id} 查询进度与结果。
    爬取的文档会落地到 LawData/<子目录>/ 并做增量去重。
    """
    if req.source != "npc":
        raise HTTPException(400, "暂仅支持 source=npc（国家法律法规数据库）")
    task_id = uuid4().hex[:12]
    _crawl_tasks[task_id] = {
        "status": "pending",
        "progress": {"total": 0, "added": 0, "updated": 0, "skipped": 0, "failed": 0},
        "errors": [], "files": [], "finished": False,
        "result": None, "rebuild": None,
    }
    asyncio.create_task(asyncio.to_thread(_run_crawl, task_id, req))
    return CrawlTaskResponse(
        task_id=task_id, status="pending",
        message="爬取任务已提交，请用 GET /api/crawl/status/{task_id} 查询进度",
    )


def _run_crawl(task_id: str, req: CrawlRequest) -> None:
    from dataclasses import asdict

    from src.knowledge.crawler import NpcLawCrawler

    state = _crawl_tasks.get(task_id)
    if state is None:
        return
    state["status"] = "running"
    try:
        crawler = NpcLawCrawler()

        def _on_progress(r) -> None:
            state["progress"] = {
                "total": r.total, "added": r.added, "updated": r.updated,
                "skipped": r.skipped, "failed": r.failed,
            }

        res = crawler.crawl(
            doc_type=req.doc_type, keyword=req.keyword, limit=req.limit,
            force=req.force, subdir=req.subdir, store=req.store,
            progress_cb=_on_progress,
        )
        state["result"] = asdict(res)
        state["errors"] = res.errors
        state["files"] = res.files
        state["progress"] = {
            "total": res.total, "added": res.added, "updated": res.updated,
            "skipped": res.skipped, "failed": res.failed,
        }
        state["finished"] = True
        state["status"] = "done"
        if req.rebuild and "pg" not in (req.store or "txt").lower():
            _trigger_rebuild(task_id)
    except Exception as e:
        state["status"] = "error"
        state["errors"] = [str(e)]
        state["finished"] = True
        logger.error(f"[crawl] task {task_id} 失败: {e}")


def _trigger_rebuild(task_id: str) -> None:
    import subprocess

    state = _crawl_tasks.get(task_id)
    if state is None:
        return
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    script = PROJECT_ROOT / "scripts" / "build_index.py"
    state["rebuild"] = "running"
    try:
        subprocess.run([sys.executable, str(script), "build"], cwd=str(PROJECT_ROOT), check=True)
        state["rebuild"] = "done"
    except Exception as e:
        state["rebuild"] = f"error: {e}"
        logger.error(f"[crawl] 重建索引失败: {e}")


@router.get("/crawl/status/{task_id}", response_model=CrawlStatusResponse)
async def get_crawl_status(task_id: str):
    """查询爬取任务状态与结果"""
    state = _crawl_tasks.get(task_id)
    if state is None:
        raise HTTPException(404, "任务不存在")
    return CrawlStatusResponse(
        task_id=task_id,
        status=state["status"],
        progress=state["progress"],
        errors=state["errors"],
        files=state["files"],
        finished=state["finished"],
        rebuild=state.get("rebuild"),
        result=state.get("result"),
    )


@router.get("/crawl/types")
async def list_crawl_types():
    """列出支持的爬取类型与说明"""
    return {
        "source": "npc",
        "types": {
            "law": "法律法规", "regulation": "行政法规", "judicial": "司法解释",
            "local": "地方性法规", "constitution": "宪法", "supervision": "监察法规",
            "all": "全部（依次爬取上述类型）",
        },
        "unsupported": ["case（案例 / 裁判文书，该数据源不提供）"],
    }
