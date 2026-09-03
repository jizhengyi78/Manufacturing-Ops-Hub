"""
session.py — Redis 短期记忆管理
===============================
角色：管理单个会话的对话上下文，让模型能记住"刚才聊了什么"。

为什么需要短期记忆:
- 用户问"这台机器怎么办" → 需要知道上文说的"这台机器"是"HA-003 注塑机"
- 用户追问"那步骤2呢" → 需要知道步骤2指的是上一轮回答的维修步骤

存储结构:
Redis Key: session:{session_id}
类型: LIST (FIFO 队列)
内容: 每轮对话的 JSON 序列化消息
TTL: 2 小时 (会话活跃时自动续期)

窗口策略:
- 保留最近 N 轮对话 (默认 20 条消息，即 10 轮问答)
- 超过 N 轮的旧消息自动从 LIST 尾部弹出 (LTRIM)
- 20 条消息约 4000-8000 tokens，留给 LLM 回答的空间充足

消息格式:
{
    "role": "user" | "assistant" | "system",
    "content": "...",
    "timestamp": "2026-07-17T14:30:00Z",
    "citations": [{"doc_id": "...", "title": "..."}],  // 仅 assistant
    "model": "deepseek-chat",                           // 仅 assistant
}

使用示例:
    from src.memory.session import SessionMemory, get_session_memory

    memory = get_session_memory()

    # 添加一轮对话
    await memory.add_message("session_abc", {"role": "user", "content": "料筒温度异常"})
    await memory.add_message("session_abc", {"role": "assistant", "content": "根据SOP-HA-12..."})

    # 获取上下文 (给 LLM 用)
    context = await memory.get_context("session_abc", max_messages=20)
    # context = [
    #   {"role": "user", "content": "料筒温度异常"},
    #   {"role": "assistant", "content": "根据SOP-HA-12..."}
    # ]

    # 获取最后一条用户消息 (用于快捷指令)
    last_user_msg = await memory.get_last_user_message("session_abc")

注意事项:
- Phase 1 提供内存实现 (不需要 Redis)，Phase 2 切 Redis
- 会话 ID 由 API 层生成，前端传回来维持同一会话
- system prompt 不存这里，在 graph node 里动态注入
- 消息超过 2000 个字符会被压缩 (长日志粘贴场景)
"""

import json
import time
from typing import Optional

from src.core.logging import get_logger

logger = get_logger(__name__)

MAX_MESSAGE_LENGTH = 2000  # 单条消息最长 2000 字符


class SessionMemory:
    """会话记忆管理器 (Phase 1: 内存, Phase 2: Redis)。

    Redis 可用时自动使用 Redis 持久化，否则降级到内存。

    示例:
        memory = SessionMemory()
        await memory.configure_redis(redis_client)  # Phase 2: 连 Redis
        await memory.add_message("s1", {"role": "user", "content": "你好"})
        ctx = await memory.get_context("s1")
    """

    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
        self._sessions: dict[str, list[dict]] = {}       # {session_id: [messages]}
        self._user_sessions: dict[str, list[str]] = {}   # {user_id: [session_ids]}
        self._session_user: dict[str, str] = {}          # {session_id: user_id}
        self._created_at: dict[str, float] = {}
        self._last_active: dict[str, float] = {}
        self._redis = None

    def configure_redis(self, redis_client) -> None:
        self._redis = redis_client

    def bind_user(self, session_id: str, user_id: str) -> None:
        """Phase 3: 绑定会话到用户。"""
        self._session_user[session_id] = user_id
        if user_id not in self._user_sessions:
            self._user_sessions[user_id] = []
        if session_id not in self._user_sessions[user_id]:
            self._user_sessions[user_id].append(session_id)

    def get_user_sessions(self, user_id: str) -> list[dict]:
        """获取用户的所有会话摘要 (内存 + 数据库合并)。"""
        sids = set(self._user_sessions.get(user_id, []))

        # 1. 从数据库获取全部会话
        db_sessions = self._restore_sessions_from_db(user_id)
        for s in db_sessions:
            sids.add(s["session_id"])

        # 2. 构建结果：优先用内存数据，兜底用数据库
        result = []
        for sid in sids:
            msgs = self._sessions.get(sid, [])
            if msgs:
                first_msg = msgs[0].get("content", "")[:50]
                last_active = self._last_active.get(sid, 0)
                result.append({
                    "session_id": sid,
                    "preview": first_msg,
                    "message_count": len(msgs),
                    "last_active": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last_active)) if last_active else "",
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self._created_at.get(sid, 0))) if self._created_at.get(sid) else "",
                })
            else:
                # 从数据库数据填充
                db_match = next((s for s in db_sessions if s["session_id"] == sid), None)
                if db_match:
                    result.append(db_match)

        result.sort(key=lambda s: s["last_active"], reverse=True)
        return result

    def _restore_sessions_from_db(self, user_id: str) -> list[dict]:
        """从 SQLite conversations 表恢复用户会话列表（同步方式）。"""
        try:
            import sqlite3
            from pathlib import Path
            db_path = Path(__file__).parent.parent.parent / "data" / "manufacturing.db"
            conn = sqlite3.connect(str(db_path))
            rows = conn.execute(
                "SELECT session_id, MIN(created_at), COUNT(*) FROM conversations "
                "WHERE user_id=? GROUP BY session_id ORDER BY MIN(created_at) DESC LIMIT 20",
                (user_id,)
            ).fetchall()
            sessions = []
            for sid, created, count in rows:
                first = conn.execute(
                    "SELECT content FROM conversations WHERE session_id=? AND role='user' ORDER BY created_at LIMIT 1",
                    (sid,)
                ).fetchone()
                preview = (first[0] if first else "(空)")[:50]
                # 截掉毫秒: "2026-07-20 16:26:04.000000" → "2026-07-20 16:26:04"
                ts = str(created)[:19] if created else ""
                sessions.append({
                    "session_id": sid,
                    "preview": preview,
                    "message_count": count,
                    "last_active": ts,
                    "created_at": ts,
                })
            conn.close()
            return sessions
        except Exception:
            return []

    async def save_to_db(self, session_id: str, user_id: str, message: dict) -> None:
        """Phase 3: 持久化消息到 SQLite conversations 表。"""
        try:
            from src.core.database import get_session_factory
            from src.data.models import Conversation
            factory = get_session_factory()
            async with factory() as db_session:
                db_session.add(Conversation(
                    session_id=session_id, user_id=user_id,
                    role=message.get("role", "user"),
                    content=message.get("content", ""),
                    citations=message.get("citations", []),
                    token_count=message.get("token_count", 0),
                    model_used=message.get("model", ""),
                ))
                await db_session.commit()
        except Exception:
            pass  # DB不可用时静默降级

    def _ensure_session(self, session_id: str) -> None:
        """确保会话存在。"""
        if session_id not in self._sessions:
            self._sessions[session_id] = []
            self._created_at[session_id] = time.time()
        self._last_active[session_id] = time.time()

    async def add_message(self, session_id: str, message: dict, user_id: str = "") -> None:
        self._ensure_session(session_id)
        if user_id:
            self.bind_user(session_id, user_id)

        msg = dict(message)
        content = msg.get("content", "")
        if len(content) > MAX_MESSAGE_LENGTH:
            msg["content"] = content[:MAX_MESSAGE_LENGTH] + f"\n... [已截断, 原文{len(content)}字符]"
            msg["truncated"] = True
        if "timestamp" not in msg:
            msg["timestamp"] = time.time()

        if self._redis:
            import json
            key = f"session:{session_id}"
            await self._redis.rpush(key, json.dumps(msg, ensure_ascii=False, default=str))
            await self._redis.ltrim(key, -self.max_messages, -1)
            await self._redis.expire(key, 7200)
        else:
            self._sessions[session_id].append(msg)
            if len(self._sessions[session_id]) > self.max_messages:
                self._sessions[session_id] = self._sessions[session_id][-self.max_messages:]

        # Phase 3: 持久化到数据库
        if user_id:
            await self.save_to_db(session_id, user_id, msg)

        logger.debug(f"会话 [{session_id}] 添加消息")

    async def get_context(self, session_id, max_messages=None, as_llm_format=True) -> list[dict]:
        self._ensure_session(session_id)
        limit = max_messages or self.max_messages

        # Phase 2: Redis
        if self._redis:
            import json
            key = f"session:{session_id}"
            raw = await self._redis.lrange(key, -limit, -1)
            messages = [json.loads(m) for m in raw]
        else:
            messages = self._sessions[session_id][-limit:]

        if as_llm_format:
            return [{"role": m["role"], "content": m["content"]} for m in messages]
        return list(messages)

    async def get_last_user_message(self, session_id: str) -> Optional[dict]:
        """获取最后一条用户消息 (用于快捷指令获取用户输入)。

        示例:
            msg = await memory.get_last_user_message("session_abc")
            if msg:
                print(f"用户最后说: {msg['content']}")
        """
        self._ensure_session(session_id)
        msgs = self._sessions.get(session_id, [])
        for m in reversed(msgs):
            if m.get("role") == "user":
                return m
        return None

    async def get_message_count(self, session_id: str) -> int:
        """获取会话消息数。"""
        return len(self._sessions.get(session_id, []))

    async def delete_session(self, session_id: str) -> None:
        """删除整个会话 (用户主动清除 / 会话结束)。"""
        self._sessions.pop(session_id, None)
        self._created_at.pop(session_id, None)
        self._last_active.pop(session_id, None)
        logger.info(f"会话删除: {session_id}")

    async def clear_inactive(self, ttl_seconds: int = 7200) -> int:
        """清理过期会话 (2 小时无活跃)。

        返回: 清理的会话数

        应用于: 后台定时任务，避免内存泄漏。
        """
        now = time.time()
        expired = [
            sid for sid, last in self._last_active.items()
            if now - last > ttl_seconds
        ]
        for sid in expired:
            self._sessions.pop(sid, None)
            self._created_at.pop(sid, None)
            self._last_active.pop(sid, None)
        if expired:
            logger.info(f"清理过期会话: {len(expired)} 个")
        return len(expired)

    @property
    def session_count(self) -> int:
        return len(self._sessions)


# 全局单例
_session_memory: SessionMemory | None = None


def get_session_memory() -> SessionMemory:
    """获取会话记忆单例。"""
    global _session_memory
    if _session_memory is None:
        _session_memory = SessionMemory()
    return _session_memory
