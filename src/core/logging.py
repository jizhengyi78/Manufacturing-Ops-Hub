"""
logging.py — 统一日志配置
=========================
角色：全系统日志的统一入口，基于 loguru 库。

输出目标:
- stdout: 开发时终端查看 (彩色格式)
- logs/app.log: 全量日志，50MB 轮转，保留7天
- logs/error.log: 仅 ERROR 级别，10MB 轮转，保留30天

日志级别: DEBUG < INFO < WARNING < ERROR
生产环境建议 INFO，开发调试可改 DEBUG。

使用方式:
    # 方式1: 直接用全局 logger
    from src.core.logging import logger
    logger.info("启动成功")

    # 方式2: 创建模块级 logger (推荐，能定位源码位置)
    from src.core.logging import get_logger
    logger = get_logger(__name__)
    logger.info("检索完成", extra={"query": "...", "latency_ms": 120})

注意事项:
- setup_logging() 必须在应用启动时调用一次 (在 app.py 的 lifespan 中)
- 不要在循环里打 DEBUG 日志，高并发下会刷满磁盘
- 审计相关的日志走 security/audit.py，不走这里
"""

import sys

from loguru import logger

from src.core.config import get_settings


def setup_logging() -> None:
    settings = get_settings()
    logger.remove()

    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stdout,
        format=fmt,
        level=settings.log_level,
        colorize=True,
    )

    logger.add(
        "logs/app.log",
        format=fmt,
        level="INFO",
        rotation="50 MB",
        retention="7 days",
        encoding="utf-8",
    )

    logger.add(
        "logs/error.log",
        format=fmt,
        level="ERROR",
        rotation="10 MB",
        retention="30 days",
        encoding="utf-8",
    )


def get_logger(name: str):
    """获取模块级 logger。"""
    return logger.bind(name=name)


# 默认导出一个 root logger 方便直接使用
__all__ = ["logger", "setup_logging", "get_logger"]
