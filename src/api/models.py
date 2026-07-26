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


class RewriteRequest(BaseModel):
    """查询改写请求：把口语化问题规范化为法律检索查询"""
    query: str = Field(..., min_length=1, max_length=2000, description="用户原始提问")


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


class ErrorResponse(BaseModel):
    """统一错误响应"""
    error: str
    detail: str = ""
    code: str = "INTERNAL_ERROR"


class CrawlRequest(BaseModel):
    """爬取请求（数据源: 国家法律法规数据库）"""
    source: str = Field(default="npc", description="数据源，目前仅支持 npc(国家法律法规数据库)")
    doc_type: str = Field(
        default="law",
        description="文档类型: law/regulation/judicial/local/constitution/supervision/all/case",
    )
    keyword: str = Field(default="", description="标题模糊搜索关键词（空=该类型全部）")
    limit: int = Field(default=50, ge=0, le=1000, description="最多爬取条数，0=不限")
    force: bool = Field(default=False, description="是否强制重爬已存在的文档")
    subdir: str = Field(default="", description="覆盖输出子目录名（默认按 doc_type 自动）")
    store: str = Field(
        default="txt",
        description="输出目标: txt(LawData/FAISS) / pg(pgvector) / both。可组合如 pg,txt",
    )
    rebuild: bool = Field(default=False, description="爬完后是否自动重建 FAISS 索引（store 不含 txt 时无效）")


class CrawlTaskResponse(BaseModel):
    """爬取任务提交响应"""
    task_id: str
    status: str
    message: str


class CrawlStatusResponse(BaseModel):
    """爬取任务状态 / 结果"""
    task_id: str
    status: str
    progress: dict
    errors: list[str] = []
    files: list[str] = []
    finished: bool
    rebuild: str | None = None
    result: dict | None = None
