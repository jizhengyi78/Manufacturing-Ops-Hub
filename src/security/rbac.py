"""
rbac.py — RBAC 权限引擎
========================
角色：全系统权限校验的唯一入口。Guard Node 调用这里的函数做三层校验。

三维权限模型:
1. 角色权限 (permission): 能否做这个操作 (读SOP/创建工单/查看成本等)
2. 车间隔离 (workshop): 产线工人只能看自己车间的数据
3. 密级控制 (classification): 不同角色能看到的文档密级上限

权限矩阵 (简化版):
工人     → knowledge:read, diagnosis:use, max=INTERNAL
维修工   → + inspection:use, scheduling:read, work_order:create
班组长   → + report:use, cost:read (本车间)
车间主任 → + scheduling:write, max=CONFIDENTIAL
工艺工程师 → max=SECRET, 可跨车间
厂长     → 全部权限, 可跨车间, max=SECRET

A2A ACL: 每个 Agent 定义了可以调用哪些其他 Agent
- diagnosis 可以调 knowledge/scheduling/quality
- scheduling 只能调 knowledge (不能调 cost/report)

使用方式:
    from src.security.rbac import check_permission, check_workshop_access, DEMO_USERS
    user = DEMO_USERS["worker_zhang"]
    check_permission(user, "knowledge:read")  # OK
    check_workshop_access(user, "workshop-b") # 抛异常

注意事项:
- DEMO_USERS 是 Phase 1 的 mock 数据，Phase 2 切 PostgreSQL
- 权限变更需更新 _ROLE_PERMISSIONS 和 _A2A_ACL
- 厂长 workshop_id 为空字符串，视为可访问所有车间
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.core.exceptions import (
    PermissionDeniedError,
    CrossWorkshopAccessError,
    ClassificationDeniedError,
)


class UserRole(str, Enum):
    WORKER = "worker"               # 产线工人
    MAINTAINER = "maintainer"       # 设备维修工
    SHIFT_LEAD = "shift_lead"      # 班组长
    WORKSHOP_DIRECTOR = "workshop_director"  # 车间主任
    PROCESS_ENGINEER = "process_engineer"     # 工艺工程师
    PLANT_MANAGER = "plant_manager"           # 厂长


class Classification(str, Enum):
    PUBLIC = "public"           # 公开: 全员可读
    INTERNAL = "internal"       # 内部: 本车间可读
    CONFIDENTIAL = "confidential"  # 机密: 车间主任+工艺工程师+厂长
    SECRET = "secret"           # 绝密: 厂长+工艺工程师


@dataclass
class UserContext:
    user_id: str
    role: UserRole
    workshop_id: str  # 厂长可为空串
    display_name: str = ""


# 角色权限矩阵
_ROLE_PERMISSIONS: dict[UserRole, dict[str, Any]] = {
    UserRole.WORKER: {
        "knowledge:read": True,
        "knowledge:write": False,
        "diagnosis:use": True,      # 仅查询，不可创建工单
        "inspection:use": False,
        "scheduling:read": False,
        "scheduling:write": False,
        "quality:use": False,
        "report:use": False,
        "cost:read": False,
        "work_order:create": False,
        "max_classification": Classification.INTERNAL,
    },
    UserRole.MAINTAINER: {
        "knowledge:read": True,
        "knowledge:write": False,
        "diagnosis:use": True,
        "inspection:use": True,
        "scheduling:read": True,    # 只读排程窗口
        "scheduling:write": False,
        "quality:use": False,
        "report:use": False,
        "cost:read": False,
        "work_order:create": True,
        "max_classification": Classification.INTERNAL,
    },
    UserRole.SHIFT_LEAD: {
        "knowledge:read": True,
        "knowledge:write": False,
        "diagnosis:use": True,
        "inspection:use": True,
        "scheduling:read": True,
        "scheduling:write": False,
        "quality:use": True,
        "report:use": True,          # 本车间报表
        "cost:read": True,           # 本车间成本
        "work_order:create": True,
        "max_classification": Classification.INTERNAL,
    },
    UserRole.WORKSHOP_DIRECTOR: {
        "knowledge:read": True,
        "knowledge:write": True,     # 可管理本车间文档
        "diagnosis:use": True,
        "inspection:use": True,
        "scheduling:read": True,
        "scheduling:write": True,    # 可调整本车间排程
        "quality:use": True,
        "report:use": True,
        "cost:read": True,
        "work_order:create": True,
        "max_classification": Classification.CONFIDENTIAL,
    },
    UserRole.PROCESS_ENGINEER: {
        "knowledge:read": True,
        "knowledge:write": True,     # 可管理工艺文档
        "diagnosis:use": True,
        "inspection:use": True,
        "scheduling:read": True,
        "scheduling:write": False,
        "quality:use": True,
        "report:use": True,
        "cost:read": False,
        "work_order:create": True,
        "max_classification": Classification.SECRET,
    },
    UserRole.PLANT_MANAGER: {
        "knowledge:read": True,
        "knowledge:write": True,
        "diagnosis:use": True,
        "inspection:use": True,
        "scheduling:read": True,
        "scheduling:write": True,
        "quality:use": True,
        "report:use": True,
        "cost:read": True,
        "work_order:create": True,
        "max_classification": Classification.SECRET,
        "cross_workshop": True,       # 可跨车间访问
    },
}

# A2A 跨 Agent 调用 ACL
_A2A_ACL: dict[str, set[str]] = {
    # caller_agent: {allowed_target_agents}
    "diagnosis":   {"knowledge", "scheduling", "quality"},
    "knowledge":   {"diagnosis"},
    "inspection":  {"diagnosis", "knowledge"},
    "scheduling":  {"knowledge"},
    "quality":     {"diagnosis", "knowledge"},
    "report":      {"knowledge"},
    "guard":       {"*"},   # Guard 可调用所有
    "router":      {"*"},
}


def check_permission(user: UserContext, permission: str) -> None:
    """校验权限，无权限抛异常。"""
    perms = _ROLE_PERMISSIONS.get(user.role, {})
    if not perms.get(permission, False):
        raise PermissionDeniedError(f"角色 [{user.role.value}] 无权限 [{permission}]")


def check_workshop_access(user: UserContext, target_workshop: str) -> None:
    """校验车间访问权限。"""
    if not target_workshop:
        return
    # 厂长和工艺工程师可跨车间
    perms = _ROLE_PERMISSIONS.get(user.role, {})
    if perms.get("cross_workshop", False):
        return
    if user.workshop_id != target_workshop:
        raise CrossWorkshopAccessError(
            f"用户 [{user.role.value}] 不能访问车间 [{target_workshop}]，"
            f"仅限车间 [{user.workshop_id}]"
        )


def check_classification_access(user: UserContext, classification: Classification) -> None:
    """校验文档密级访问权限。"""
    perms = _ROLE_PERMISSIONS.get(user.role, {})
    max_cls = perms.get("max_classification", Classification.INTERNAL)

    cls_order = {
        Classification.PUBLIC: 0,
        Classification.INTERNAL: 1,
        Classification.CONFIDENTIAL: 2,
        Classification.SECRET: 3,
    }
    if cls_order.get(classification, 0) > cls_order.get(max_cls, 1):
        raise ClassificationDeniedError(
            f"角色 [{user.role.value}] 密级上限 [{max_cls.value}]，"
            f"无法访问 [{classification.value}]"
        )


def check_a2a_acl(caller_agent: str, target_agent: str) -> None:
    """校验 A2A 跨 Agent 调用 ACL。"""
    allowed = _A2A_ACL.get(caller_agent, set())
    if "*" not in allowed and target_agent not in allowed:
        raise PermissionDeniedError(
            f"Agent [{caller_agent}] 不允许调用 Agent [{target_agent}]"
        )


# Simplified demo data for Phase 1 (no DB yet)
DEMO_USERS: dict[str, UserContext] = {
    "worker_zhang": UserContext("worker_zhang", UserRole.WORKER, "workshop-a", "张工(产线)"),
    "maintainer_li": UserContext("maintainer_li", UserRole.MAINTAINER, "workshop-a", "李工(维修)"),
    "shift_lead_wang": UserContext("shift_lead_wang", UserRole.SHIFT_LEAD, "workshop-a", "王班长"),
    "director_zhao": UserContext("director_zhao", UserRole.WORKSHOP_DIRECTOR, "workshop-a", "赵主任"),
    "engineer_chen": UserContext("engineer_chen", UserRole.PROCESS_ENGINEER, "workshop-a", "陈工(工艺)"),
    "manager_zhou": UserContext("manager_zhou", UserRole.PLANT_MANAGER, "", "周厂长"),

    # B车间用户
    "worker_sun": UserContext("worker_sun", UserRole.WORKER, "workshop-b", "孙工(产线-B)"),
    "maintainer_huang": UserContext("maintainer_huang", UserRole.MAINTAINER, "workshop-b", "黄工(维修-B)"),
    "shift_lead_liu": UserContext("shift_lead_liu", UserRole.SHIFT_LEAD, "workshop-b", "刘班长(B)"),
}
