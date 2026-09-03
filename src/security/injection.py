"""
injection.py — Prompt Injection 防御
====================================
角色：三层防御体系中的检测引擎，负责识别和拦截注入攻击。

三层防御:
1. 用户输入层 (detect_input_injection)
   - 在 Guard Node 中调用，拦截用户直接输入的恶意指令
   - 匹配中英文常见的注入模式 (忽略指令/角色冒充/数据窃取)

2. 检索上下文层 (sanitize_retrieval_context)
   - 在 Memory Node 调用，检测被检索到的文档片段是否夹带恶意指令
   - 防御场景: 第三方供应商文档/外包维修记录中嵌入的注入语句

3. 文档入库层 (scan_document_for_injection)
   - 在文档上传时调用，批量导入前先扫描
   - 标记风险文档需要人工审核后才入库

使用示例:
    from src.security.injection import detect_input_injection
    blocked, reason = detect_input_injection(user_input)
    if blocked:
        raise InjectionDetectedError(f"匹配: {reason}")

注意事项:
- 当前是基于正则的轻量检测 (覆盖率 ~90%)，生产建议加上分类器模型
- 正则检测有漏报风险，所以系统提示词加固是必要的最后一道防线
- sanitize_retrieval_context 是在检索后、送入LLM前做的
- 不要让检测延迟超过 50ms，否则影响用户体验
"""

import re
from typing import Tuple

from src.core.logging import get_logger

logger = get_logger(__name__)

# 注入特征模式 (生产级需扩展 + 分类器模型)
_INJECTION_PATTERNS = [
    # 英文注入
    r"ignore\s+(all\s+)?(previous|above|prior)\s+instructions?",
    r"forget\s+(all\s+)?(your|previous|earlier)\s+(instructions?|rules?)",
    r"you\s+are\s+now\s+(a\s+)?\w+\s+(assistant|bot|model)",
    r"act\s+as\s+(if\s+you\s+are|a\s+)?\w+",
    r"disregard\s+(your\s+)?(instructions?|guidelines?|rules?)",
    r"override\s+(your\s+)?(system\s+)?(prompt|instructions?)",
    r"show\s+(me\s+)?(your\s+)?(system\s+)?(prompt|instructions?)",
    r"do\s+not\s+follow\s+(your\s+)?(instructions?|rules?)",
    r"you\s+must\s+(not\s+)?(follow|obey)",
    # 中文注入
    r"忽略(之前|上面|以上|所有)?(的)?(指令|规则|设定|提示)",
    r"忘记(你|之前|刚才)?(的)?(设定|规则|指令|身份)",
    r"你现在是.{1,20}(助手|模型|角色)",
    r"不要(按照|遵守|执行)(你的)?(规则|指令)",
    r"告诉(我|我一下)(你的)?(系统)?(提示词|prompt)",
    r"显示(你的)?(系统)?(提示词|prompt)",
    r"我是(厂长|车间主任|管理员|超级管理员)",
    r"以(厂长|管理员|超级管理员)(的)?(身份|角色|权限)",
    r"列出(所有|全部).{1,10}(数据|信息|记录|密码|密钥|工资)",
    # 数据窃取
    r"(extract|dump|leak|exfiltrate)\s+(the\s+)?(data|prompt|system|config)",
    r"what\s+(is|are)\s+your\s+(system\s+)?(prompt|instructions)",
]

# 检索污染特征 (文档中夹带的注入指令)
_RETRIEVAL_POLLUTION_PATTERNS = [
    r"忽略上述(规则|指令|内容|文档)",
    r"输出(系统配置|数据库|密码|密钥|所有数据)",
    r"bypass\s+(the\s+)?(security|filter|guard)",
    r"ignore\s+(the\s+)?(above|document|context|content)",
]

_compiled_input = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]
_compiled_pollution = [re.compile(p, re.IGNORECASE) for p in _RETRIEVAL_POLLUTION_PATTERNS]


def detect_input_injection(text: str) -> Tuple[bool, str]:
    """检测用户输入中的注入攻击。返回 (是否拦截, 匹配到的模式)。"""
    for i, pattern in enumerate(_compiled_input):
        match = pattern.search(text)
        if match:
            logger.warning(f"检测到注入: pattern#{i} matched '{match.group()}' in '{text[:100]}'")
            return True, f"matched_pattern_{i}"
    return False, ""


def sanitize_retrieval_context(chunks: list[str]) -> list[str]:
    """检索后清洗: 检测检索片段中的指令注入特征，命中则剔除该片段。"""
    clean = []
    removed = 0
    for chunk in chunks:
        is_clean = True
        for pattern in _compiled_pollution:
            if pattern.search(chunk):
                logger.warning(f"检索片段被清洗: '{chunk[:80]}...'")
                is_clean = False
                removed += 1
                break
        if is_clean:
            clean.append(chunk)
    if removed:
        logger.info(f"检索后清洗: 剔除 {removed}/{len(chunks)} 个片段")
    return clean


def scan_document_for_injection(content: str) -> Tuple[bool, list[str]]:
    """文档入库前置扫描: 检测文档内容是否含注入指令。返回 (是否有风险, 风险标签列表)。"""
    risks = []
    for i, pattern in enumerate(_compiled_pollution):
        if pattern.search(content):
            risks.append(f"pollution_pattern_{i}")
    for i, pattern in enumerate(_compiled_input):
        if pattern.search(content):
            risks.append(f"injection_pattern_{i}")
    return len(risks) > 0, risks
