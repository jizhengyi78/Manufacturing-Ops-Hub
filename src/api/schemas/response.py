"""
response.py — API 响应模型 (Pydantic)
======================================
定义所有 API 端点的响应体结构。

统一响应格式:
  成功: {"code": 0, "message": "success", "data": {...}}
  分页: {"code": 0, "message": "success", "data": {"items": [...], "total": N, "page": 1, "page_size": 20}}
  错误: {"code": 40001, "message": "权限不足", "detail": {...}}
"""

from typing import Any, Generic, Optional, TypeVar
from pydantic import BaseModel, Field

T = TypeVar("T")


class APIResponse(BaseModel, Generic[T]):
    """通用 API 响应。

    示例 (成功):
        APIResponse(code=0, message="success", data={"answer": "..."})

    示例 (错误):
        APIResponse(code=40001, message="权限不足", detail={"permission": "cost:read"})
    """
    code: int = Field(default=0, description="错误码，0=成功")
    message: str = Field(default="success", description="提示信息")
    data: Optional[T] = Field(default=None, description="响应数据")
    detail: Optional[dict] = Field(default=None, description="错误详情")


class PageData(BaseModel, Generic[T]):
    """分页数据。"""
    items: list[T] = Field(default_factory=list)
    total: int = Field(default=0)
    page: int = Field(default=1)
    page_size: int = Field(default=20)


class ChatResponse(BaseModel):
    """对话完成响应 (非流式)。"""
    session_id: str
    answer: str
    model_used: str = ""
    citations: list[dict] = Field(default_factory=list)
    tokens_used: int = 0
    fallback_used: bool = False
    latency_ms: float = 0.0


class ConversationMessage(BaseModel):
    """对话历史中的单条消息。"""
    role: str
    content: str
    citations: list[dict] = Field(default_factory=list)
    created_at: str = ""


class ConversationHistoryResponse(BaseModel):
    """对话历史响应。"""
    session_id: str
    messages: list[ConversationMessage] = Field(default_factory=list)
    message_count: int = 0


class HealthResponse(BaseModel):
    """健康检查响应。"""
    status: str = "healthy"
    version: str = "0.1.0"
    mode: str = "online"
    checks: dict[str, str] = Field(default_factory=dict)


class DocumentInfo(BaseModel):
    """文档信息。"""
    doc_id: str
    title: str
    doc_type: str
    workshop_id: str = ""
    equipment_model: str = ""
    classification: str = "internal"
    status: str = "active"
    chunk_count: int = 0
    ingested_at: str = ""


class IngestionResponse(BaseModel):
    """文档摄入响应。"""
    doc_id: str
    status: str
    chunk_count: int = 0
    error: str = ""


class KnowledgeSearchResult(BaseModel):
    """检索结果。"""
    rank: int
    chunk_id: str
    content_preview: str
    relevance_score: float
    source: str = ""


class KnowledgeSearchResponse(BaseModel):
    """检索测试响应。"""
    results: list[KnowledgeSearchResult] = Field(default_factory=list)
    latency_ms: float = 0.0


class ErrorResponse(BaseModel):
    """错误响应 (备用)。"""
    code: int
    message: str
    detail: Optional[dict] = None
