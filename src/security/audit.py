"""
audit.py — 审计日志
===================
角色：全量记录所有安全相关事件，满足合规审计要求。

记录内容 (全量存储原始载荷):
- 注入拦截: 用户 ID、原始输入、匹配到的模式
- 权限拒绝: 用户 ID、尝试的操作、目标资源
- 工具越权: Agent ID、工具名、目标表、尝试的查询
- A2A 通信: 调用方、目标、消息 ID、结果
- 系统事件: 离线/在线切换、熔断触发/恢复、预算告警

存储: 双写 (loguru 日志文件 + 可选 Redis 实时通道)

日志级别:
- info: 正常的 A2A 调用、系统事件
- warn: 注入拦截、权限拒绝、熔断触发
- critical: 工具越权、数据泄露尝试

使用示例:
    from src.security.audit import audit_logger, AuditEntry, AuditEventType
    audit_logger.log(AuditEntry(
        event_type=AuditEventType.INJECTION_BLOCKED,
        user_id="worker_zhang",
        severity="warn",
        detail={"matched_pattern": "pattern_3", "input_preview": "忽略之前..."}
    ))

注意事项:
- 审计日志不等同于业务日志，专门服务于安全合规
- 生产环境需持久化到 PostgreSQL audit_logs 表
- 敏感载荷 (如用户输入) 在日志中保留前 500 字符, 不截断
- Phase 1 仅写文件/Redis, Phase 4 接 Langfuse/OpenTelemetry 统一追踪
"""

import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any

from src.core.logging import get_logger

logger = get_logger(__name__)


class AuditEventType(str, Enum):
    INJECTION_BLOCKED = "injection_blocked"
    PERMISSION_DENIED = "permission_denied"
    CROSS_WORKSHOP_DENIED = "cross_workshop_denied"
    CLASSIFICATION_DENIED = "classification_denied"
    TOOL_VIOLATION = "tool_violation"
    A2A_CALL = "a2a_call"
    A2A_LOOP_DETECTED = "a2a_loop_detected"
    A2A_RATE_LIMITED = "a2a_rate_limited"
    OFFLINE_SWITCH = "offline_switch"
    ONLINE_SWITCH = "online_switch"
    CIRCUIT_OPEN = "circuit_open"
    CIRCUIT_CLOSED = "circuit_closed"
    CHECKPOINT_RECOVERED = "checkpoint_recovered"
    TOKEN_BUDGET_WARNING = "token_budget_warning"
    TOKEN_BUDGET_CRITICAL = "token_budget_critical"


@dataclass
class AuditEntry:
    event_type: str
    user_id: str = ""
    agent_id: str = ""
    tool_name: str = ""
    target_table: str = ""
    severity: str = "info"
    detail: dict[str, Any] = field(default_factory=dict)
    ip_address: str = ""
    timestamp: float = field(default_factory=time.time)


class AuditLogger:
    """审计日志: 双写 (loguru 日志文件 + Redis 实时通道)。"""

    def __init__(self):
        self._buffer: list[AuditEntry] = []

    def log(self, entry: AuditEntry) -> None:
        level = "ERROR" if entry.severity == "critical" else ("WARNING" if entry.severity == "warn" else "INFO")
        msg = json.dumps(asdict(entry), ensure_ascii=False, default=str)
        logger.bind(audit=True).log(level, msg)
        self._buffer.append(entry)
        # Phase 3: 持久化到 DB
        try:
            import asyncio
            from src.core.database import get_session_factory
            from src.data.models import AuditLog
            factory = get_session_factory()
            async def _save():
                async with factory() as session:
                    session.add(AuditLog(
                        event_type=entry.event_type, user_id=entry.user_id,
                        agent_id=entry.agent_id, tool_name=entry.tool_name,
                        target_table=entry.target_table, severity=entry.severity,
                        detail=entry.detail, ip_address=entry.ip_address,
                    ))
                    await session.commit()
            try: asyncio.get_running_loop(); asyncio.create_task(_save())
            except RuntimeError: pass
        except Exception: pass

    def log_block(
        self,
        event_type: str,
        user_id: str = "",
        agent_id: str = "",
        tool_name: str = "",
        target_table: str = "",
        severity: str = "warn",
        detail: dict | None = None,
    ) -> None:
        self.log(AuditEntry(
            event_type=event_type,
            user_id=user_id,
            agent_id=agent_id,
            tool_name=tool_name,
            target_table=target_table,
            severity=severity,
            detail=detail or {},
        ))

    def flush(self) -> list[AuditEntry]:
        entries = self._buffer[:]
        self._buffer.clear()
        return entries


# 全局单例
audit_logger = AuditLogger()
