"""
auth.py — 登录接口
==================
用户通过账号密码登录，数据库 bcrypt 验证，返回 JWT Token。
"""

import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.api.middleware.auth import create_token
from src.security.rbac import DEMO_USERS, UserContext, UserRole
from src.api.schemas.response import APIResponse
from src.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["认证"])


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    token: str
    user_id: str
    display_name: str
    role: str
    workshop_id: str


@router.post("/login")
async def login(req: LoginRequest):
    """POST /api/v1/auth/login — 用户登录 (bcrypt 验证)。

    请求: {"username": "worker_zhang", "password": "由 DEMO_PASSWORD 环境变量决定"}
    返回: {"code":0, "data":{"token":"eyJ...", "user_id":"worker_zhang", ...}}
    """
    # 1. 查数据库 (user 表, bcrypt 验证)
    import bcrypt

    user_ctx = None
    try:
        from src.core.database import get_session_factory
        from src.data.models import User
        factory = get_session_factory()
        async with factory() as session:
            from sqlalchemy import select
            result = await session.execute(select(User).where(User.username == req.username))
            db_user = result.scalar_one_or_none()
            if db_user and db_user.password_hash:
                if bcrypt.checkpw(req.password.encode(), db_user.password_hash.encode()):
                    user_ctx = UserContext(
                        user_id=db_user.id, role=UserRole(db_user.role),
                        workshop_id=db_user.workshop_id, display_name=db_user.display_name,
                    )
    except Exception as e:
        logger.warning(f"数据库用户查询失败: {e}")

    # 2. 兜底: DEMO_USERS (开发环境兼容)
    if user_ctx is None:
        demo = DEMO_USERS.get(req.username)
        if demo and req.password == os.environ.get("DEMO_PASSWORD", ""):
            user_ctx = demo
        elif not os.environ.get("DEMO_PASSWORD"):
            raise HTTPException(401, "服务器未配置演示密码，请联系管理员")
        else:
            raise HTTPException(401, "用户名或密码错误")

    # 3. 生成 Token
    token = create_token(user_ctx)
    logger.info(f"用户登录: {user_ctx.display_name} ({user_ctx.role.value})")

    return APIResponse(data=LoginResponse(
        token=token, user_id=user_ctx.user_id,
        display_name=user_ctx.display_name, role=user_ctx.role.value,
        workshop_id=user_ctx.workshop_id,
    ).model_dump()).model_dump()


@router.get("/me")
async def list_users():
    """GET /api/v1/auth/me — 列出所有可用用户。"""
    users = [
        {"id": uid, "name": u.display_name, "role": u.role.value, "workshop": u.workshop_id}
        for uid, u in DEMO_USERS.items()
    ]
    demo_pw = os.environ.get("DEMO_PASSWORD", "")
    tip = f"演示密码: {demo_pw}" if demo_pw else "请联系管理员获取密码"
    return APIResponse(data={"users": users, "tip": tip}).model_dump()
