"""
rule_engine.py — 规则匹配引擎
=============================
LLM 不可用时降级到规则库。报警码 → SOP 精确匹配，全程不调 LLM。

数据: data/seeds/alarm_rules.json (15条核心报警码)

用法:
    from src.model.rule_engine import match_rule
    result = match_rule("HT-E-0021")
    if result:
        print(result)  # "HT-E-0021: 料筒温度传感器异常。处理步骤: 1.检查温控器..."
"""

import json
import re
from pathlib import Path
from functools import lru_cache

from src.core.logging import get_logger

logger = get_logger(__name__)


@lru_cache
def _load_rules() -> list[dict]:
    path = Path(__file__).parent.parent.parent / "data" / "seeds" / "alarm_rules.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"规则库加载: {len(data['rules'])} 条")
    return data["rules"]


def match_rule(query: str) -> str | None:
    """从查询文本中匹配规则。

    优先级: 精确报警码匹配 > 报文码模糊匹配 > 无匹配

    返回: 格式化的SOP文本，或 None

    示例:
        >>> match_rule("HT-E-0021报警怎么处理")
        "HT-E-0021 (P1): 料筒温度传感器异常\n处理步骤:\n1. 检查温控器接线..."
    """
    rules = _load_rules()
    query_upper = query.upper()

    # 1. 精确报警码匹配
    for r in rules:
        if r["alarm_code"] in query_upper:
            return _format_rule(r)

    # 2. 模糊匹配: 提取查询中的报警码模式
    alarm_match = re.search(r'[A-Z]{2,}-[A-Z0-9]+-\d+', query_upper)
    if alarm_match:
        code = alarm_match.group(0)
        for r in rules:
            if r["alarm_code"] == code:
                return _format_rule(r)

    # 3. 设备型号+故障关键词模糊匹配
    for r in rules:
        if r["equipment"].upper() in query_upper and any(
            kw in query for kw in r.get("meaning", "").split()
        ):
            return _format_rule(r)

    return None


def _format_rule(rule: dict) -> str:
    """格式化规则为可读文本。"""
    parts = [
        f"【规则降级模式 - 无需LLM】",
        f"报警码: {rule['alarm_code']} ({rule['level']})",
        f"含义: {rule['meaning']}",
        f"设备: {rule['equipment']}",
        f"\n处理步骤:",
    ]
    for i, step in enumerate(rule["steps"], 1):
        parts.append(f"  {i}. {step}")
    if rule.get("safety"):
        parts.append(f"\n{rule['safety']}")
    return "\n".join(parts)


def get_rule_coverage() -> dict:
    """获取规则库覆盖统计。"""
    rules = _load_rules()
    codes = [r["alarm_code"] for r in rules]
    return {
        "total_rules": len(rules),
        "alarm_codes": codes,
        "equipment": list(set(r["equipment"] for r in rules)),
    }
