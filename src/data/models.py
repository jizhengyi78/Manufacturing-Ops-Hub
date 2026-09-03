"""
models.py — SQLAlchemy ORM 模型
===============================
Phase 2: SQLite (开发)
Phase 3: PostgreSQL (生产, 模型不变, 只改连接串)

10 张表:
  users, documents, ingestion_transactions, dedup_log,
  work_orders, knowledge_cases,
  conversations, token_usage_daily,
  audit_logs, sync_log
"""
import uuid
from datetime import datetime

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, Date,
    ForeignKey, JSON, BigInteger, UniqueConstraint, Index,
)
from sqlalchemy.orm import relationship

from src.core.database import Base


def _uuid():
    return str(uuid.uuid4())


def _now():
    return datetime.now().replace(microsecond=0)


# ── 用户与权限 ─────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=_uuid)
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(128), default="")
    display_name = Column(String(100), nullable=False)
    role = Column(String(30), nullable=False)
    workshop_id = Column(String(50), default="")
    phone = Column(String(20), default="")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=_now)
    updated_at = Column(DateTime, default=_now, onupdate=_now)

    __table_args__ = (
        Index("idx_users_role", "role"),
        Index("idx_users_workshop", "workshop_id"),
    )


# ── 文档与知识库 ───────────────────────────────

class Document(Base):
    """文档元数据表。"""
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True, default=_uuid)
    doc_id = Column(String(100), unique=True, nullable=False)
    title = Column(String(500), nullable=False)
    doc_type = Column(String(50), nullable=False)
    file_format = Column(String(20), default="")
    file_path = Column(Text, default="")
    workshop_id = Column(String(50), default="")
    equipment_model = Column(String(200), default="")
    classification = Column(String(20), default="internal")
    version = Column(String(20), default="1.0")
    status = Column(String(20), default="active")
    chunk_count = Column(Integer, default=0)
    file_size_bytes = Column(BigInteger, default=0)
    ingested_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=_now, onupdate=_now)
    created_at = Column(DateTime, default=_now)

    __table_args__ = (
        Index("idx_docs_type", "doc_type"),
        Index("idx_docs_status", "status"),
        Index("idx_docs_workshop", "workshop_id"),
        Index("idx_docs_equipment", "equipment_model"),
    )


class IngestionTransaction(Base):
    """双写事务记录表。"""
    __tablename__ = "ingestion_transactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(String(100), nullable=False)
    chunk_id = Column(String(100), unique=True, nullable=False)
    milvus_ok = Column(Boolean, default=False)
    es_ok = Column(Boolean, default=False)
    status = Column(String(20), default="pending")
    retry_count = Column(Integer, default=0)
    last_error = Column(Text, default="")
    created_at = Column(DateTime, default=_now)
    updated_at = Column(DateTime, default=_now, onupdate=_now)

    __table_args__ = (Index("idx_ingest_status", "status", "created_at"),)


class DedupLog(Base):
    """离线同步去重表。"""
    __tablename__ = "dedup_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    offline_uuid = Column(String(36), unique=True, nullable=False)
    online_uuid = Column(String(36), nullable=True)
    similarity_score = Column(Float, nullable=True)
    action = Column(String(20), nullable=False)
    reviewer = Column(String(100), default="")
    reviewed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=_now)


# ── 工单 ──────────────────────────────────────

class WorkOrder(Base):
    __tablename__ = "work_orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    mes_event_id = Column(String(100), unique=True, nullable=False)
    mes_order_id = Column(String(100), default="")
    equipment_id = Column(String(100), default="")
    equipment_model = Column(String(200), default="")
    fault_code = Column(String(50), default="")
    fault_desc = Column(Text, default="")
    priority = Column(String(10), default="P2")
    status = Column(String(20), default="pending")
    assigned_to = Column(String(36), nullable=True)
    workshop_id = Column(String(50), default="")
    diagnosis = Column(Text, default="")
    repair_actions = Column(Text, default="")
    resolution = Column(Text, default="")
    created_at = Column(DateTime, default=_now)
    updated_at = Column(DateTime, default=_now, onupdate=_now)
    completed_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("idx_wo_status", "status"),
        Index("idx_wo_workshop", "workshop_id"),
        Index("idx_wo_equipment", "equipment_id"),
        Index("idx_wo_created", "created_at"),
    )


# ── 知识案例 ──────────────────────────────────

class KnowledgeCase(Base):
    __tablename__ = "knowledge_cases"

    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(String(100), unique=True, nullable=False)
    equipment_model = Column(String(200), default="")
    fault_code = Column(String(50), default="")
    fault_category = Column(String(100), default="")
    root_cause = Column(Text, default="")
    repair_steps = Column(JSON, default=list)
    safety_notes = Column(Text, default="")
    source_wo_id = Column(String(100), default="")
    verification = Column(String(20), default="unverified")
    weight = Column(Float, default=1.0)
    status = Column(String(20), default="active")
    review_status = Column(String(20), default="")
    reviewer_id = Column(String(36), nullable=True)
    access_count = Column(Integer, default=0)
    last_accessed = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=_now)
    updated_at = Column(DateTime, default=_now, onupdate=_now)

    __table_args__ = (
        Index("idx_case_model", "equipment_model"),
        Index("idx_case_fault", "fault_code"),
        Index("idx_case_category", "fault_category"),
        Index("idx_case_status", "status"),
    )


# ── 对话与记忆 ────────────────────────────────

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String(36), primary_key=True, default=_uuid)
    session_id = Column(String(100), nullable=False)
    user_id = Column(String(36), nullable=False)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    citations = Column(JSON, default=list)
    token_count = Column(Integer, default=0)
    model_used = Column(String(50), default="")
    model_cost = Column(Float, default=0.0)
    latency_ms = Column(Integer, default=0)
    feedback = Column(String(10), default="")
    created_at = Column(DateTime, default=_now)

    __table_args__ = (
        Index("idx_conv_session", "session_id", "created_at"),
        Index("idx_conv_user", "user_id", "created_at"),
    )


class TokenUsageDaily(Base):
    __tablename__ = "token_usage_daily"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False)
    workshop_id = Column(String(50), nullable=False)
    agent = Column(String(50), nullable=False)
    model = Column(String(50), nullable=False)
    prompt_tokens = Column(BigInteger, default=0)
    completion_tokens = Column(BigInteger, default=0)
    cost = Column(Float, default=0.0)
    request_count = Column(Integer, default=0)

    __table_args__ = (
        UniqueConstraint("date", "workshop_id", "agent", "model", name="uq_token_daily"),
        Index("idx_token_date", "date"),
    )


# ── 审计与监控 ────────────────────────────────

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_type = Column(String(50), nullable=False)
    user_id = Column(String(36), default="")
    agent_id = Column(String(50), default="")
    tool_name = Column(String(100), default="")
    target_table = Column(String(100), default="")
    severity = Column(String(20), default="info")
    detail = Column(JSON, default=dict)
    ip_address = Column(String(45), default="")
    created_at = Column(DateTime, default=_now)

    __table_args__ = (
        Index("idx_audit_type", "event_type", "created_at"),
        Index("idx_audit_severity", "severity", "created_at"),
    )


class SyncLog(Base):
    __tablename__ = "sync_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sync_type = Column(String(50), nullable=False)
    direction = Column(String(20), nullable=False)
    records_synced = Column(Integer, default=0)
    status = Column(String(20), default="success")
    last_sync_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=_now)
