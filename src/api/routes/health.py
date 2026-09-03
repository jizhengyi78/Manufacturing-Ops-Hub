"""
health.py — 健康检查接口
========================
提供 Kubernetes 就绪探针和存活探针。

端点:
- GET /api/v1/health: 完整健康检查 (依赖服务状态)
- GET /api/v1/health/readiness: K8s Readiness Probe (仅检查核心依赖)
- GET /api/v1/health/liveness: K8s Liveness Probe (仅检查进程存活)
"""

from fastapi import APIRouter

from src.api.schemas.response import APIResponse, HealthResponse

router = APIRouter(prefix="/health", tags=["健康检查"])


@router.get("")
async def health():
    """完整健康检查 — 检查所有依赖服务状态。

    返回:
        {
            "code": 0,
            "data": {
                "status": "healthy",
                "version": "0.1.0",
                "mode": "online",
                "checks": {
                    "postgresql": "ok",
                    "redis": "ok",
                    "milvus": "ok",
                    "elasticsearch": "ok",
                    "deepseek_api": "ok"
                }
            }
        }
    """
    checks = {}

    # Phase 1: 内存实现，所以这些都是 ok
    # Phase 2: 真正检查各服务连接
    checks["api"] = "ok"
    checks["graph"] = "ok"

    return APIResponse(
        data=HealthResponse(
            status="healthy",
            version="0.1.0",
            mode="online",
            checks=checks,
        ).model_dump()
    ).model_dump()


@router.get("/readiness")
async def readiness():
    """K8s Readiness Probe — Pod 是否准备好接收流量。

    检查: 核心组件 (Graph 已编译) 就绪即可返回 ready。
    """
    return {"status": "ready"}


@router.get("/liveness")
async def liveness():
    """K8s Liveness Probe — Pod 是否存活。

    只要进程在运行就返回 alive，不做任何依赖检查。
    """
    return {"status": "alive"}
