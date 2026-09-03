"""
app.py — FastAPI 应用入口
=========================
制造业多 Agent 生产运维数字助手的主 API 服务。

启动方式:
  uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

应用生命周期:
  1. startup: 初始化日志、预热模型、启动后台任务
  2. running: 处理请求
  3. shutdown: 清理资源

模块加载顺序:
  1. setup_logging() — 日志系统最先初始化
  2. get_compiled_graph() — 预热 LangGraph (首次调用会加载 Embedding 模型)
  3. 启动 FastAPI — 开始监听请求

架构分层:
  请求 → Middleware (注入检测/鉴权/限流) → Route → Graph → Agent → LLM
                ↑                                    ↓
            [异常处理] ← ← ← ← ← ← ← ← ← ← ← ← [异常传播]

注意事项:
- 生产环境不要用 --reload (热重载有内存泄漏)
- 端口 8000 是开发默认值，生产通过环境变量 PORT 指定
- 前端静态文件通过 /app 路径 serve (Phase 2)
"""

from contextlib import asynccontextmanager
import asyncio

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.core.config import get_settings
from src.core.logging import setup_logging, logger
from src.core.exceptions import ManufacturingAgentError
from src.api.routes import conversation, health
from src.api.deps import get_compiled_graph


async def _warmup_ocr():
    """后台预热 OCR 模型，不阻塞启动流程。"""
    try:
        from src.integration.ocr import warmup
        await warmup()
    except Exception as e:
        logger.warning(f"OCR 预热失败: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理。

    Startup:
    1. 初始化日志
    2. 预热 LangGraph (加载 embedding 模型等)
    3. 日志启动信息

    Shutdown:
    1. 清理会话内存
    2. 日志关闭信息
    """
    # Startup
    setup_logging()
    settings = get_settings()
    logger.info(f"制造业多Agent生产运维助手 v0.1.0 启动中...")
    logger.info(f"LLM: 主={settings.default_model}, 备={settings.fallback_model}")
    logger.info(f"嵌入模型: {settings.embedding_model} on {settings.embedding_device}")

    # 安全检查
    if not settings.jwt_secret_key:
        import secrets
        settings.jwt_secret_key = secrets.token_hex(32)
        logger.warning("JWT_SECRET_KEY 未设置，已生成随机密钥（服务重启后会话失效）")
    if not settings.a2a_secret_key:
        import secrets
        settings.a2a_secret_key = secrets.token_hex(32)
        logger.warning("A2A_SECRET_KEY 未设置，已生成随机密钥")

    # 初始化数据库 (SQLite: data/manufacturing.db)
    from src.core.database import init_db
    await init_db()

    # Phase 2: 尝试连 Redis (不可用自动降级)
    from src.core.redis_client import get_redis as _get_redis, is_redis_available
    if await is_redis_available():
        redis = await _get_redis()
        from src.memory.session import get_session_memory
        get_session_memory().configure_redis(redis)
        logger.info("Redis 已连接，会话/缓存持久化已启用")
    else:
        logger.info("Redis 不可用，使用内存模式")

    # 预热: 确保 Graph 在第一个请求到达前编译完成
    try:
        from src.api.deps import get_compiled_graph as _get_graph
        _get_graph()
        logger.info("LangGraph 工作流已就绪")
    except Exception as e:
        logger.error(f"Graph 预热失败: {e}")

    # OCR 模型后台预热（不阻塞后续启动）
    try:
        asyncio.create_task(_warmup_ocr())
    except Exception as e:
        logger.warning(f"OCR 预热跳过: {e}")

    # 自动加载种子数据（BM25内存索引 + Milvus向量库）
    try:
        from src.api.routes.admin import load_seed
        await load_seed()
    except Exception as e:
        logger.warning(f"种子数据加载跳过: {e}")

    logger.info("应用启动完成")

    yield

    # Shutdown
    logger.info("应用关闭中...")
    from src.memory.session import get_session_memory
    memory = get_session_memory()
    expired = await memory.clear_inactive(ttl_seconds=0)  # 清理全部
    logger.info(f"清理了 {expired} 个会话")
    logger.info("应用已关闭")


app = FastAPI(
    title="制造业多Agent生产运维助手",
    description="基于多Agent架构的离散制造生产运维数字助手",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS (开发环境允许所有来源)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 路由
app.include_router(conversation.router, prefix="/api/v1")
app.include_router(health.router, prefix="/api/v1")
from src.api.routes import admin, knowledge as knowledge_route, auth as auth_route
app.include_router(admin.router, prefix="/api/v1")
app.include_router(knowledge_route.router, prefix="/api/v1")
app.include_router(auth_route.router, prefix="/api/v1")

# Rate Limit Middleware
from src.api.middleware.ratelimit import RateLimitMiddleware
app.add_middleware(RateLimitMiddleware)

# Prometheus Metrics
from src.observability.metrics import get_metrics_response
@app.get("/metrics")
async def metrics():
    from fastapi.responses import Response
    data, ct = get_metrics_response()
    return Response(content=data, media_type=ct)

# Static Uploads
from pathlib import Path as _Path
from fastapi.staticfiles import StaticFiles
uploads_dir = _Path(__file__).parent.parent.parent / "data" / "uploads"
uploads_dir.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(uploads_dir)), name="uploads")

# 全局异常处理
@app.exception_handler(ManufacturingAgentError)
async def manufacturing_error_handler(request: Request, exc: ManufacturingAgentError):
    """将自定义异常转换为统一错误响应。"""
    return JSONResponse(
        status_code=_http_status(exc.code),
        content={
            "code": int(exc.code[1:]),  # "E10001" → 10001
            "message": exc.message,
            "detail": exc.detail,
        },
    )


@app.exception_handler(Exception)
async def general_error_handler(request: Request, exc: Exception):
    logger.error(f"未捕获异常: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "code": 50000,
            "message": "系统内部错误",
            "detail": {"error": str(exc)},
        },
    )


def _http_status(error_code: str) -> int:
    """根据错误码返回对应的 HTTP 状态码。"""
    if error_code.startswith("E1"):
        return 403  # 安全相关 → 403
    elif error_code.startswith("E2"):
        return 503  # 模型不可用 → 503
    elif error_code.startswith("E3"):
        return 503  # 检索不可用 → 503
    elif error_code.startswith("E4"):
        return 500  # Agent 异常 → 500
    elif error_code.startswith("E5"):
        return 502  # 集成异常 → 502
    return 500


@app.get("/")
async def root():
    """根路径 — 返回服务信息。"""
    return {
        "name": "制造业多Agent生产运维助手",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
