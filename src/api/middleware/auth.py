"""
auth.py — JWT 认证中间件 & 工具函数
===================================
创建和验证 JWT Token，从请求中提取用户身份。

Token Payload:
  {
    "sub": "user-uuid",
    "role": "worker",
    "workshop_id": "workshop-a",
    "display_name": "张工",
    "exp": 1718000000
  }

使用方式:
  # 生成 token
  token = create_token(user)

  # 从请求中获取用户
  user = await get_current_user(request)
"""

from datetime import datetime, timedelta

from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError

from src.core.config import get_settings
from src.security.rbac import UserContext, UserRole, DEMO_USERS
from src.core.logging import get_logger

logger = get_logger(__name__)

security = HTTPBearer(auto_error=False)


def create_token(user: UserContext) -> str:
    """生成 JWT Token。

    参数:
        user: 用户上下文

    返回: JWT 字符串

    示例:
        user = DEMO_USERS["worker_zhang"]
        token = create_token(user)  # "eyJhbGciOi..."
    """
    settings = get_settings()
    expire = datetime.utcnow() + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {
        "sub": user.user_id,
        "role": user.role.value,
        "workshop_id": user.workshop_id,
        "display_name": user.display_name,
        "exp": expire,
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def verify_token(token: str) -> dict | None:
    """验证 JWT Token，返回 payload 或 None。

    示例:
        payload = verify_token("eyJhbGci...")
        if payload:
            user_id = payload["sub"]
    """
    settings = get_settings()
    try:
        return jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
    except JWTError:
        return None


async def get_current_user(request: Request) -> UserContext | None:
    """从请求的 Authorization Header 中提取当前用户。

    返回 UserContext 或 None (未登录)。
    未登录时使用默认访客身份 (worker)。

    用法 (在路由中):
        user = await get_current_user(request)
        if user is None:
            raise HTTPException(401, "请先登录")
    """
    # 1. 尝试从 Authorization Header 获取 JWT
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        payload = verify_token(token)
        if payload:
            return UserContext(
                user_id=payload["sub"],
                role=UserRole(payload["role"]),
                workshop_id=payload.get("workshop_id", ""),
                display_name=payload.get("display_name", ""),
            )

    # 2. 开发环境: 从 X-User-Id Header 获取 (前端角色切换用)
    user_id = request.headers.get("X-User-Id", "")
    if user_id and user_id in DEMO_USERS:
        return DEMO_USERS[user_id]

    # 3. 未认证: 返回 None (对话接口应拒绝)
    return None
