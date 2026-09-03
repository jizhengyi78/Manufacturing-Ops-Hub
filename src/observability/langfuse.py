"""
langfuse.py — LLM 可观测性 (Langfuse 集成)
===========================================
Phase 4: 追踪每次 LLM 调用的 token 消耗、延迟、模型选择和结果。

无 API Key 时自动降级为空操作，不影响系统运行。

用法:
    from src.observability.langfuse import trace_llm_call
    with trace_llm_call(model="deepseek-chat", messages=[...]) as trace:
        result = await router.chat(messages=...)
        trace.set_result(result)
"""

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from src.core.logging import get_logger

logger = get_logger(__name__)

_langfuse_client = None


def init_langfuse():
    """初始化 Langfuse（需要 API Key 才能连接）。"""
    global _langfuse_client
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")
    if not public_key or not secret_key:
        logger.info("Langfuse 未配置 (无 API Key)，LLM 追踪降级为本地日志")
        return

    try:
        from langfuse import Langfuse
        _langfuse_client = Langfuse(
            public_key=public_key, secret_key=secret_key,
            host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
        logger.info("Langfuse 已连接")
    except Exception as e:
        logger.warning(f"Langfuse 初始化失败: {e}")


@dataclass
class LLMTrace:
    """LLM 调用追踪记录。有 Langfuse 时上报，否则本地日志。"""
    trace_name: str = "llm_call"
    model: str = ""
    user_id: str = ""
    session_id: str = ""
    start_time: float = field(default_factory=time.time)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0
    output_preview: str = ""
    error: str = ""

    def set_result(self, prompt_tokens: int, completion_tokens: int, output: str):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.latency_ms = (time.time() - self.start_time) * 1000
        self.output_preview = output[:200]

    def set_error(self, error: str):
        self.error = error
        self.latency_ms = (time.time() - self.start_time) * 1000

    def __enter__(self):
        return self

    def __exit__(self, *args):
        # 本地日志
        level = "ERROR" if self.error else "INFO"
        logger.bind(llm_trace=True).log(
            level,
            f"LLM[{self.model}]: {self.prompt_tokens}+{self.completion_tokens}tokens, "
            f"{self.latency_ms:.0f}ms, preview={self.output_preview[:50]}"
        )

        # Langfuse 上报 (如果已初始化)
        if _langfuse_client:
            try:
                trace = _langfuse_client.trace(
                    name=self.trace_name,
                    user_id=self.user_id,
                    session_id=self.session_id,
                    metadata={"model": self.model, "latency_ms": self.latency_ms},
                )
                trace.generation(
                    name=f"{self.model}_completion",
                    model=self.model,
                    usage={"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens},
                )
            except Exception:
                pass  # Langfuse 上报失败不影响业务


@contextmanager
def trace_llm_call(model: str = "", user_id: str = "", session_id: str = "", trace_name: str = "llm_call"):
    """创建 LLM 调用追踪上下文。

    用法:
        with trace_llm_call(model="deepseek-chat", user_id="worker_zhang") as trace:
            result = await router.chat(messages=...)
            trace.set_result(result.prompt_tokens, result.completion_tokens, result.content)
    """
    trace = LLMTrace(
        trace_name=trace_name, model=model,
        user_id=user_id, session_id=session_id,
    )
    try:
        yield trace
    except Exception as e:
        trace.set_error(str(e))
