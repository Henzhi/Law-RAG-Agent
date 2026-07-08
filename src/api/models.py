"""
API 请求/响应模型。
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """单次问答请求"""
    query: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    top_k: int = Field(default=5, ge=1, le=20, description="检索条文数")


class ChatResponse(BaseModel):
    """单次问答响应"""
    query: str
    answer: str
    sources: list[dict] = Field(default_factory=list)

    @classmethod
    def from_rag_answer(cls, query: str, answer: str, sources: list) -> "ChatResponse":
        return cls(
            query=query,
            answer=answer,
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
