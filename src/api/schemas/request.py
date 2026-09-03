"""
request.py — API 请求模型 (Pydantic)
====================================
定义所有 API 端点的请求体结构。

命名规范: XxxRequest
"""

from typing import Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """对话请求。

    示例:
        {
            "session_id": null,
            "message": "注塑机料筒温度异常怎么处理",
            "context": {"equipment_id": "HA-003", "fault_code": "HT-E-0021"}
        }

    session_id 为 null 时服务端自动创建新会话。
    """
    session_id: Optional[str] = Field(
        default=None,
        description="会话ID，不传则新建会话。同一个 session_id 维持上下文连续性",
        max_length=100,
    )
    message: str = Field(
        ...,
        description="用户输入的问题或指令",
        min_length=1,
        max_length=5000,
    )
    user_id: Optional[str] = Field(
        default=None,
        description="用户ID (Phase 1 可选，用于角色模拟)",
    )
    context: Optional[dict] = Field(
        default=None,
        description="可选的上下文信息，如设备ID、报警码等",
        examples=[{"equipment_id": "HA-003", "fault_code": "HT-E-0021"}],
    )
    workshop_id: str = Field(
        default="workshop-a",
        description="车间ID，用于数据隔离",
    )


class ChatStopRequest(BaseModel):
    """停止流式生成请求。"""
    session_id: str = Field(..., description="要停止的会话ID")


class DocumentUploadRequest(BaseModel):
    """文档上传请求 (元数据部分，文件通过 multipart 上传)。"""
    doc_type: str = Field(
        ...,
        description="文档类型: equipment_manual / sop / fault_case / quality_standard / alarm_code",
    )
    workshop_id: str = Field(default="workshop-a", description="所属车间")
    equipment_model: str = Field(default="", description="关联设备型号")
    classification: str = Field(default="internal", description="密级: public / internal / confidential / secret")


class KnowledgeSearchRequest(BaseModel):
    """管理后台检索测试请求。"""
    query: str = Field(..., min_length=1, max_length=500)
    workshop_id: str = Field(default="workshop-a")
    top_k: int = Field(default=10, ge=1, le=50)
    hybrid: bool = Field(default=True, description="是否使用混合检索 (vs 纯向量)")
