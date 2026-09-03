"""
sanitize.py — 输出脱敏
======================
角色：在 Guard Node 输出阶段，对 LLM 响应做敏感数据脱敏。

脱敏内容:
- 无条件脱敏: 银行卡号、手机号、邮箱 (个人隐私)
- 条件脱敏: 成本数据、工资信息 (仅高权限角色可看)

使用方式:
    from src.security.sanitize import sanitize_output
    safe_text = sanitize_output(llm_response, user_can_see_cost=False)

注意事项:
- 正则脱敏有误杀风险 (如设备参数中恰好含数字串)
- user_can_see_cost 由 RBAC 中的 cost:read 权限决定
- 这是最后一道防线，不能替代 RBAC 层的权限控制
- 未来可替换为基于 NER 的脱敏模型
"""

import re
from typing import Set

# 敏感字段 (在输出中检测并脱敏)
_SENSITIVE_PATTERNS: list[tuple[str, str]] = [
    (r'\b\d{15,19}\b', '[银行卡号已隐藏]'),           # 银行卡号
    (r'\b1[3-9]\d{9}\b', '[手机号已隐藏]'),           # 手机号
    (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[邮箱已隐藏]'),  # 邮箱
    (r'(成本\s*[：:]\s*)\d+\.?\d*', r'\1[已脱敏]'),   # 成本数据 — 仅低权限用户
    (r'(工资|薪酬|薪资)\s*[：:]\s*\d+\.?\d*', r'\1[已脱敏]'),  # 工资
]

_LOW_COST_FIELDS = ["成本", "cost", "费用", "利润", "利润率"]


def sanitize_output(text: str, user_can_see_cost: bool = False) -> str:
    """输出脱敏: 基础个人信息 + 条件性成本脱敏。"""
    result = text
    for pattern, replacement in _SENSITIVE_PATTERNS:
        if not user_can_see_cost and "成本" in pattern:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        elif "工资" in pattern and not user_can_see_cost:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        elif "银行卡" in pattern or "手机号" in pattern or "邮箱" in pattern:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    return result


def contains_sensitive_data(text: str) -> bool:
    """快速检测响应是否包含未脱敏的敏感数据。"""
    for pattern, _ in _SENSITIVE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False
