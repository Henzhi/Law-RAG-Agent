"""
FastAPI 应用入口。

启动:
    uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from uuid import uuid4

from .routes import router as api_router
from .routes import auth_router
from .models import ErrorResponse

STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时预热（FAISS 加载 ~3s 提前到启动时消耗）"""
    from src.config import AGENT_ENABLED
    if AGENT_ENABLED:
        from .dependencies import get_agent
        get_agent(force_reload=True)
        logger.info("Agent 图已重建")
    else:
        from .dependencies import get_engine
        get_engine()
        logger.info("RAG 引擎已预热")
    yield


app = FastAPI(
    title="Law-RAG-Agent",
    description="基于本地 LLM 的法律法规智能问答系统",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — 本地部署只允许 Vite 开发服务器和同源访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# request_id 追踪中间件
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    rid = uuid4().hex[:8]
    request.state.request_id = rid
    logger.info(f"[{rid}] {request.method} {request.url.path}")
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response


# ------------------------------------------------------------------
# 全局异常处理
# ------------------------------------------------------------------

@app.exception_handler(TimeoutError)
async def timeout_exception_handler(request: Request, exc: TimeoutError):
    logger.error(f"请求超时: {request.url} - {exc}")
    return JSONResponse(
        status_code=504,
        content=ErrorResponse(
            error="LLM 响应超时，请稍后重试",
            detail=str(exc),
            code="TIMEOUT",
        ).model_dump(),
    )


@app.exception_handler(ConnectionError)
async def connection_exception_handler(request: Request, exc: ConnectionError):
    logger.error(f"连接失败: {request.url} - {exc}")
    return JSONResponse(
        status_code=503,
        content=ErrorResponse(
            error="服务暂不可用，请检查 Ollama / PostgreSQL 是否正常运行",
            detail=str(exc),
            code="SERVICE_UNAVAILABLE",
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """兜底：捕获所有未处理的异常"""
    logger.exception(f"未处理的异常: {request.url}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="服务器内部错误",
            detail=str(exc),
            code="INTERNAL_ERROR",
        ).model_dump(),
    )

# API 路由
app.include_router(api_router, prefix="/api")
app.include_router(auth_router, prefix="/api/auth")

# 启动时加载 Token 缓存（PG 不可用时跳过，不影响测试）
from .auth import load_token_cache
try:
    load_token_cache()
except Exception:
    logger.warning("Token 缓存加载失败（PG 未启动？），认证功能可能受限")

# 静态文件（前端页面）
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
