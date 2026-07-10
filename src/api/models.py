"""
API 请求/响应模型。
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """单次问答请求"""
    query: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    top_k: int = Field(default=5, ge=1, le=20, description="检索条文数")
    history: list[dict] = Field(default_factory=list, description="多轮对话历史 [{role, content}]")
    session_id: str = Field(default="", description="会话 ID，客户端传入，服务端按用户隔离校验")


class ChatResponse(BaseModel):
    """单次问答响应"""
    query: str
    answer: str
    sources: list[dict] = Field(default_factory=list)
    is_casual: bool = False

    @classmethod
    def from_rag_answer(cls, query: str, answer: str, sources: list, is_casual: bool = False) -> "ChatResponse":
        return cls(
            query=query,
            answer=answer,
            is_casual=is_casual,
            sources=[
                {
                    "law_name": s.law_name,
                    "chapter": s.chapter,
                    "article_range": s.article_range,
                    "citation": s.citation,
                    "score": float(s.score),  # FAISS 返回 numpy.float32，需转 Python float
                }
                for s in sources
            ],
        )


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    version: str
    index_ready: bool
    doc_count: int
    llm_model: str


class RegisterRequest(BaseModel):
    """用户注册请求"""
    username: str = Field(..., min_length=2, max_length=64, description="用户名")
    password: str = Field(..., min_length=6, max_length=128, description="密码，至少 6 位")


class LoginRequest(BaseModel):
    """用户登录请求"""
    username: str = Field(..., min_length=1, max_length=64, description="用户名")
    password: str = Field(..., min_length=1, max_length=128, description="密码")


class AuthResponse(BaseModel):
    """认证响应（注册/登录共用）"""
    user_id: str
    token: str
    username: str
