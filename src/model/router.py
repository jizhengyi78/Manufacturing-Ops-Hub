"""
router.py — 多模型路由
=======================
角色：根据任务复杂度将请求分派到不同模型，控制成本。
      简单 SOP 问答 → Qwen (便宜)
      复杂故障诊断 → DeepSeek (强推理)
      报表查询     → 直接走 SQL，不调 LLM

分派逻辑 (classify_complexity):
- 报表关键词 (OEE/良率/成本/统计等) → REPORT (不走 LLM)
- 故障关键词 (为什么/怎么修/诊断等) 或查询 > 50 字 → COMPLEX
- 其余 → SIMPLE

使用示例:
    from src.model.router import router
    result = await router.chat(messages=[{"role":"user","content":"怎么换模具"}])
    print(result.content)

注意事项:
- 模型选择和 配置绑定: 在 .env 中配置哪个模型做简单/复杂任务
- API Key: DeepSeek 和 Qwen 需要不同 API Key，缺一个也可以配置成同一个
- classify_complexity 是基于规则的快速分类，不需要 LLM 调用
- 生产建议替换为轻量分类器模型
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator

from openai import AsyncOpenAI

from src.core.config import get_settings
from src.core.logging import get_logger

logger = get_logger(__name__)


class TaskComplexity(str, Enum):
    SIMPLE = "simple"       # SOP 问答、简单查询
    COMPLEX = "complex"     # 故障诊断、多步推理
    REPORT = "report"       # 报表生成 (走 SQL, 不调 LLM)


@dataclass
class ModelResult:
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    fallback_used: bool = False


@dataclass
class ModelRouter:
    _clients: dict[str, AsyncOpenAI] = field(default_factory=dict)

    def _get_client(self, model: str) -> AsyncOpenAI:
        if model not in self._clients:
            settings = get_settings()
            if "deepseek" in model:
                client = AsyncOpenAI(
                    api_key=settings.deepseek_api_key,
                    base_url=settings.deepseek_base_url,
                )
            elif "qwen" in model:
                client = AsyncOpenAI(
                    api_key=settings.qwen_api_key,
                    base_url=settings.qwen_base_url,
                )
            else:
                client = AsyncOpenAI(
                    api_key=settings.deepseek_api_key,
                    base_url=settings.deepseek_base_url,
                )
            self._clients[model] = client
        return self._clients[model]

    def classify_complexity(self, query: str) -> TaskComplexity:
        """基于关键词 + 查询长度做快速分类，避免额外 LLM 调用。"""
        # 报表关键词
        report_kw = ["OEE", "良率", "产量", "成本", "报表", "统计", "对比", "趋势",
                      "日报", "月报", "本月", "今天", "昨日"]
        if any(kw in query for kw in report_kw):
            return TaskComplexity.REPORT

        # 复杂诊断关键词
        complex_kw = ["为什么", "原因", "根因", "怎么修", "怎么处理", "诊断",
                       "分析", "排查", "怎么办", "反复", "一直"]
        if any(kw in query for kw in complex_kw) or len(query) > 50:
            return TaskComplexity.COMPLEX

        return TaskComplexity.SIMPLE

    def select_model(self, complexity: TaskComplexity) -> str:
        settings = get_settings()
        if complexity == TaskComplexity.SIMPLE:
            return settings.simple_task_model
        elif complexity == TaskComplexity.COMPLEX:
            return settings.complex_task_model
        return settings.default_model

    async def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        stream: bool = False,
    ) -> ModelResult:
        """调用 LLM, 返回统一结果。"""
        model = model or get_settings().default_model
        client = self._get_client(model)

        t0 = time.time()
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        latency = (time.time() - t0) * 1000

        choice = response.choices[0]
        return ModelResult(
            content=choice.message.content or "",
            model=model,
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            latency_ms=latency,
        )

    async def chat_stream(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        """流式调用，逐 chunk yield 文本。"""
        model = model or get_settings().default_model
        client = self._get_client(model)

        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


# 全局单例
router = ModelRouter()
