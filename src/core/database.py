"""
database.py — 数据库引擎 & 会话管理
===================================
Phase 2: SQLite (开发, 零安装)
Phase 3: PostgreSQL (生产, 改一行连接串)

切换方式: .env 中 DATABASE_URL 改为 postgresql+asyncpg://...
"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from src.core.config import get_settings
from src.core.logging import get_logger

logger = get_logger(__name__)

# 模型基类
class Base(DeclarativeBase):
    pass

# 导入所有模型到 Base.metadata
import src.data.models



# 引擎 & 会话工厂
_engine = None
_session_factory = None


def _get_database_url() -> str:
    """优先读 DATABASE_URL 环境变量，默认用 SQLite。"""
    import os
    url = os.environ.get("DATABASE_URL", "")
    if url:
        return url
    # 默认 SQLite (项目 data 目录下)
    from pathlib import Path
    db_path = Path(__file__).parent.parent.parent / "data" / "manufacturing.db"
    return f"sqlite+aiosqlite:///{db_path}"


def get_engine():
    global _engine
    if _engine is None:
        url = _get_database_url()
        logger.info(f"数据库引擎: {url[:50]}...")
        _engine = create_async_engine(
            url,
            echo=False,
            pool_size=20,           # 连接池大小
            max_overflow=10,        # 超出 pool_size 后的最大连接数
            pool_recycle=3600,      # 连接回收时间 (秒)
            pool_pre_ping=True,     # 每次使用前验证连接
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(get_engine(), class_=AsyncSession, expire_on_commit=False)
    return _session_factory


async def get_session() -> AsyncSession:
    """获取一个数据库会话 (FastAPI Depends 用)。

    用法:
        from src.core.database import get_session
        async with await get_session() as session:
            ...
    """
    factory = get_session_factory()
    return factory()


async def init_db():
    """启动时调用: 创建所有表。"""
    from sqlalchemy import create_engine, inspect
    import os
    url = os.environ.get("DATABASE_URL", "")
    if url:
        # PostgreSQL: async 创建
        engine = get_engine()
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    else:
        # SQLite: 同步创建 (异步 SQLite 不支持 create_all)
        from pathlib import Path
        import src.data.models as _models  # 注册所有 ORM 模型
        db_path = Path(__file__).parent.parent.parent / "data" / "manufacturing.db"
        sync_engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(sync_engine)
        sync_engine.dispose()
    logger.info("数据库表已创建/确认")


async def close_db():
    """关闭时调用: 释放引擎。"""
    global _engine
    if _engine:
        await _engine.dispose()
        _engine = None
        logger.info("数据库引擎已关闭")
