"""
compressor.py — 上下文压缩
==========================
角色：当对话历史 + 检索结果的总 Token 超过模型窗口 70% 时，
      对旧消息做摘要压缩，保留关键信息丢弃闲聊。

为什么需要压缩:
- 多轮故障排查可能持续 20+ 轮 (设备报警→诊断→维修→验证)
- 加上检索到的文档片段，总 Token 很容易超过 8K/32K 窗口
- 不压缩的话: 要么截断丢失关键信息，要么超出窗口回答混乱

压缩策略:
1. 保留最近 4 轮对话 (完整保留)
2. 更早的对话用 LLM 做摘要 (Qwen-2.5-7B, 成本低)
3. 摘要保留: 设备号、报警码、操作步骤、维修结论
4. 摘要丢弃: 闲聊、"好的"、"收到"、"谢谢"等
5. 检索片段单独压缩: 长文档做关键信息提取

触发条件:
- 总 token > 模型窗口 × 70%
- 例如: 32K 窗口 × 70% = 22.4K → 超过就触发

压缩流程:
┌─────────────────┐
│ 1. 估算总 token  │ ← 按 1 token ≈ 2 中文字符 粗略估算
├─────────────────┤
│ 2. 判断是否超阈值│ ← 总 token > 窗口 × 0.7 → 触发
├─────────────────┤
│ 3. 分离待压缩部分 │ ← 保留最近 4 轮，更早的提取出来
├─────────────────┤
│ 4. LLM 摘要压缩  │ ← 调用 Qwen-2.5-7B (不是主模型，省钱)
├─────────────────┤
│ 5. 替换上下文     │ ← 旧消息 → 替换为摘要 system 消息
└─────────────────┘

使用示例:
    from src.memory.compressor import ContextCompressor

    compressor = ContextCompressor()
    compressed_msgs = await compressor.compress(
        messages=all_messages,
        max_window_tokens=32000,
        retain_recent=4,  # 保留最近 4 轮
    )
    # compressed_msgs = [summarized_system_msg, ...recent_4_msgs]

注意事项:
- 压缩用便宜模型 (Qwen-2.5-7B)，不用主推理模型
- 不要丢掉报警码和设备型号 (压缩 prompt 里要明确要求保留)
- 压缩频率不要太高 (不要每轮都压，只在超阈值时压)
- Phase 1 用简单截断模拟压缩 (不调 LLM)，Phase 2 接入真正的压缩
"""

import re
from typing import Optional

from src.core.config import get_settings
from src.core.logging import get_logger

logger = get_logger(__name__)

# 粗略估算: 1 token ≈ 2 中文字符
CHARS_PER_TOKEN = 2


class ContextCompressor:
    """上下文压缩器。

    Phase 1: 简单截断 + 结构保留 (不调 LLM)
    Phase 2: 接入 Qwen-2.5-7B 做真正的摘要压缩

    示例:
        compressor = ContextCompressor()
        msgs = [{"role": "user", "content": "..."}, ...]  # 30条消息
        compressed = await compressor.compress(msgs, max_window_tokens=32000)
        # compressed 保留了最近几轮完整 + 早期摘要
    """

    def __init__(self):
        self.compressor_model = get_settings().compressor_model

    def _estimate_tokens(self, text: str) -> int:
        """粗略估算 Token 数。

        中文: 1 token ≈ 2 字符
        英文: 1 token ≈ 4 字符
        公式: len(text) / CHARS_PER_TOKEN (中文为主简化)

        示例:
            >>> compressor._estimate_tokens("设备报警")
            2  # "设备报警" 4个字符 ÷ 2 = 2 tokens (实际约
        """
        return max(1, len(text) // CHARS_PER_TOKEN)

    def _estimate_messages_tokens(self, messages: list[dict]) -> int:
        """估算消息列表的总 Token 数。"""
        total = 0
        for m in messages:
            total += self._estimate_tokens(m.get("content", ""))
        return total

    def _extract_key_info(self, messages: list[dict]) -> str:
        """从旧消息中提取关键信息 (Phase 1: 规则提取，Phase 2: LLM 摘要)。

        保留: 设备型号、报警码、故障描述、维修步骤、最终结论
        丢弃: 闲聊、确认性回复

        示例:
            msgs = [
                {"role": "user", "content": "HA-003报警了"},
                {"role": "assistant", "content": "好的，什么报警？"},
                {"role": "user", "content": "料筒温度异常，报警码HT-E-0021"},
                {"role": "assistant", "content": "收到，根据SOP-HA-12，步骤1: 关闭加热..."},
            ]
            summary = compressor._extract_key_info(msgs)
            # "设备HA-003, 报警码HT-E-0021, 料筒温度异常, 维修参考SOP-HA-12"
        """
        key_parts = []
        # 设备型号
        model_pattern = re.compile(r'(HA-\w+|MA\d+|CNC-\w+|[A-Z]{2,}-\w{3,})', re.IGNORECASE)
        # 报警码
        alarm_pattern = re.compile(r'[A-Z]{2,}-\w{3,}')

        for m in messages:
            content = m.get("content", "")
            models = model_pattern.findall(content)
            alarms = alarm_pattern.findall(content)
            if models:
                key_parts.append(f"设备: {', '.join(set(models))}")
            if alarms:
                key_parts.append(f"报警码: {', '.join(set(alarms))}")

        # 最后一条 assistant 消息的关键结论 (取前 200 字)
        for m in reversed(messages):
            if m.get("role") == "assistant":
                content = m.get("content", "")
                if len(content) > 20:
                    key_parts.append(f"结论: {content[:200]}")
                break

        return " | ".join(set(key_parts)) if key_parts else "..."

    async def compress(
        self,
        messages: list[dict],
        max_window_tokens: int = 32000,
        retain_recent: int = 4,
    ) -> list[dict]:
        """压缩上下文。

        参数:
            messages: 完整消息列表 (从旧到新)
            max_window_tokens: 模型窗口大小 (token 数)
            retain_recent: 保留最近 N 条消息不压缩

        返回: 压缩后的消息列表

        压缩逻辑:
        1. 总 token < 窗口 × 70% → 不压缩，直接返回
        2. 总 token >= 窗口 × 70%:
           a. 保留最近 retain_recent 条消息 (完整)
           b. 更早的消息提取关键信息做成摘要
           c. 摘要放在最前面作为 system 消息

        示例:
            msgs = [m1, m2, m3, ..., m20]
            compressed = await compressor.compress(msgs, 32000, retain_recent=4)
            # compressed = [
            #   {"role": "system", "content": "[历史摘要] 设备HA-003, 报警码HT-E-0021..."},
            #   m17, m18, m19, m20
            # ]
        """
        if not messages:
            return messages

        total_tokens = self._estimate_messages_tokens(messages)
        threshold = int(max_window_tokens * get_settings().context_window_ratio_threshold)

        if total_tokens <= threshold:
            # 不需要压缩
            logger.debug(f"上下文无需压缩: {total_tokens}/{threshold} tokens")
            return messages

        # 需要压缩
        logger.info(f"上下文压缩触发: {total_tokens}/{threshold} tokens (>{max_window_tokens * 0.7:.0f})")

        if len(messages) <= retain_recent:
            # 消息太少，无法压缩，保留全部
            return messages

        # 分离: 保留最近 N 条，更早的压缩
        recent = messages[-retain_recent:]
        older = messages[:-retain_recent]

        # 提取关键信息
        summary = self._extract_key_info(older)

        # 构建压缩后的消息
        summary_msg = {
            "role": "system",
            "content": f"[历史对话摘要 - 由上下文压缩器生成]\n{summary}\n\n请基于此摘要和后续对话继续回答。",
        }

        compressed = [summary_msg] + recent
        compressed_tokens = self._estimate_messages_tokens(compressed)
        logger.info(f"上下文压缩完成: {total_tokens} → {compressed_tokens} tokens")

        return compressed

    async def compress_retrieval_context(
        self,
        documents: list[str],
        max_chars: int = 8000,
    ) -> list[str]:
        """压缩检索片段: 总长度超过阈值时做关键信息提取。

        参数:
            documents: 检索到的文档片段列表
            max_chars: 最大字符数

        返回: 压缩后的片段列表

        示例:
            docs = ["很长的文档1 (5000字)", "很长的文档2 (6000字)"]
            compressed = await compressor.compress_retrieval_context(docs, 8000)
        """
        total = sum(len(d) for d in documents)
        if total <= max_chars:
            return documents

        # 简单截断: 每个文档保留前 N 字符 (按比例分配)
        ratios = [len(d) / total for d in documents]
        result = []
        for doc, ratio in zip(documents, ratios):
            budget = int(max_chars * ratio)
            if len(doc) > budget:
                result.append(doc[:budget] + f"\n... [已截断, 原文{len(doc)}字符]")
            else:
                result.append(doc)
        return result


# 全局单例
_compressor: ContextCompressor | None = None


def get_compressor() -> ContextCompressor:
    global _compressor
    if _compressor is None:
        _compressor = ContextCompressor()
    return _compressor
