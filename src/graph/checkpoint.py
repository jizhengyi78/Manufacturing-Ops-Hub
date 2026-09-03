"""
checkpoint.py — LangGraph Checkpoint Redis 持久化
==================================================
角色：在每个 LangGraph Node 执行后保存 AgentState 快照到 Redis，
      服务重启时可从中断点恢复，长流程不会因为重启而丢失。

为什么需要 Checkpoint:
- 故障排查可能是跨班次的长流程 (设备报警 → 维修 → 验证 可能数小时)
- 服务重启=所有进行中的流程中断=用户从头开始，工业场景不可接受
- Checkpoint 让重启后能恢复到中断点继续

存储结构:
Redis Key: checkpoint:{session_id}:{step_name}
类型: HASH
字段:
  state_json: JSON 序列化的完整 AgentState
  created_at: Unix 时间戳
  ttl: 过期秒数

Checkpoint 生命周期:
1. 每个 Node 执行后 → 自动写入 (checkpoint_after decorator)
2. 流程正常完成 → 删除本 session 的 all checkpoints
3. 会话关联工单未完成 → 自动续期 TTL (续 1 小时)
4. 超时未活跃 → 定时清理 (30分钟间隔扫描)

自动续期逻辑:
- 检测 state 关联的工单状态为 pending/created/processing → 续期
- 检测会话最后活跃在 30 分钟内 → 续期

使用示例:
    from src.graph.checkpoint import CheckpointManager

    ckpt = CheckpointManager(redis_client)
    await ckpt.save("session_abc", "guard", state)
    # ... 服务重启 ...
    state = await ckpt.load("session_abc")

注意事项:
- Phase 1 提供内存实现 (不需要 Redis)，Phase 2 切 Redis
- Checkpoint 文件不入 Git (在 .gitignore)
- 生产环境: Checkpoint TTL 1小时, 关联工单时自动续期
"""

import json
import time
from dataclasses import asdict
from typing import Optional

from src.core.logging import get_logger

logger = get_logger(__name__)

DEFAULT_TTL = 3600  # 1 小时
RENEW_THRESHOLD = 1800  # 30 分钟内活跃则续期


class CheckpointManager:
    """Checkpoint 管理器 (Phase 1: 内存实现)。

    Phase 2 切 Redis，接口不变。

    示例:
        ckpt = CheckpointManager()
        await ckpt.save("session_abc", "router", state)
        restored = await ckpt.load("session_abc")
        # restored 是完整的 AgentState 对象
    """

    def __init__(self):
        self._store: dict[str, dict] = {}    # {key: {state_json, created_at, ttl}}
        self._created_at: dict[str, float] = {}

    def _make_key(self, session_id: str, step_name: str = "latest") -> str:
        return f"checkpoint:{session_id}:{step_name}"

    async def save(self, session_id: str, step_name: str, state) -> None:
        """保存 Checkpoint。

        参数:
            session_id: 会话 ID
            step_name: 当前 Node 名 (如 "guard", "router", "knowledge")
            state: AgentState 对象

        存储内容: 完整 AgentState 序列化为 JSON

        示例:
            await ckpt.save("session_abc", "guard", state)
        """
        key = self._make_key(session_id, step_name)
        now = time.time()

        # 保存最新一步
        self._store[self._make_key(session_id, "latest")] = {
            "state_json": json.dumps(asdict(state), ensure_ascii=False, default=str),
            "step_name": step_name,
            "created_at": now,
            "ttl": DEFAULT_TTL,
        }

        # 同时按步骤名保存
        self._store[key] = {
            "state_json": json.dumps(asdict(state), ensure_ascii=False, default=str),
            "step_name": step_name,
            "created_at": now,
            "ttl": DEFAULT_TTL,
        }
        self._created_at[key] = now

        logger.debug(f"Checkpoint 保存: {key}")

    async def load(self, session_id: str) -> Optional[any]:
        """加载最近的 Checkpoint。

        返回: AgentState 对象 或 None (无快照)

        示例:
            state = await ckpt.load("session_abc")
            if state:
                print(f"恢复到步骤: {state.checkpoint_step}")
        """
        key = self._make_key(session_id, "latest")
        data = self._store.get(key)
        if data is None:
            return None

        state_json = data["state_json"]
        # 重新导入 AgentState (避免循环引用)
        from src.graph.state import AgentState
        state_dict = json.loads(state_json)
        return AgentState(**state_dict)

    async def renew(self, session_id: str) -> bool:
        """续期 Checkpoint TTL (关联工单未完成时调用)。

        续期条件: 会话最后活跃在 30 分钟内

        示例:
            await ckpt.renew("session_abc")  # 再续 1 小时
        """
        latest_key = self._make_key(session_id, "latest")
        if latest_key in self._store:
            now = time.time()
            if now - self._created_at.get(latest_key, 0) < RENEW_THRESHOLD:
                self._store[latest_key]["ttl"] = DEFAULT_TTL
                self._store[latest_key]["created_at"] = now
                logger.debug(f"Checkpoint 续期: {latest_key}")
                return True
        return False

    async def delete_session(self, session_id: str) -> int:
        """删除整个会话的全部 Checkpoint。

        调用时机: 用户主动结束对话 / 工单归档 / 流程正常完成

        示例:
            deleted = await ckpt.delete_session("session_abc")
            print(f"清理了 {deleted} 个 checkpoint")
        """
        prefix = f"checkpoint:{session_id}:"
        to_delete = [k for k in self._store if k.startswith(prefix)]
        for k in to_delete:
            self._store.pop(k, None)
            self._created_at.pop(k, None)
        if to_delete:
            logger.info(f"Checkpoint 清理: {session_id}, {len(to_delete)} 个")
        return len(to_delete)

    async def clean_expired(self) -> int:
        """定时清理过期 Checkpoint。

        扫描逻辑:
        - 超过 TTL 的过期快照 → 删除
        - 清理频率: 建议每 30 分钟调用一次

        返回: 清理的 Checkpoint 数

        示例:
            cleaned = await ckpt.clean_expired()
            if cleaned > 0:
                print(f"清理了 {cleaned} 个过期 checkpoint")
        """
        now = time.time()
        expired = []
        for key, data in self._store.items():
            created = self._created_at.get(key, 0)
            ttl = data.get("ttl", DEFAULT_TTL)
            if now - created > ttl:
                expired.append(key)

        for key in expired:
            self._store.pop(key, None)
            self._created_at.pop(key, None)

        if expired:
            logger.info(f"清理过期 Checkpoint: {len(expired)} 个")
        return len(expired)

    @property
    def count(self) -> int:
        return len(self._store)

    async def get_all_sessions(self) -> list[str]:
        """获取所有有 checkpoint 的 session 列表。"""
        sessions = set()
        for key in self._store:
            parts = key.split(":")
            if len(parts) >= 2:
                sessions.add(parts[1])
        return list(sessions)


# 全局单例
_checkpoint_mgr: CheckpointManager | None = None


def get_checkpoint_manager() -> CheckpointManager:
    global _checkpoint_mgr
    if _checkpoint_mgr is None:
        _checkpoint_mgr = CheckpointManager()
    return _checkpoint_mgr
