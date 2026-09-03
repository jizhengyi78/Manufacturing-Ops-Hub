"""
metrics.py — Prometheus 指标
============================
Phase 4: 系统级指标收集和暴露。

指标:
  - http_requests_total (counter): 总请求数
  - http_request_duration_seconds (histogram): 请求延迟
  - llm_calls_total (counter): LLM 调用次数
  - llm_token_usage_total (counter): Token 消耗
  - retrieval_duration_seconds (histogram): 检索延迟
  - active_sessions (gauge): 活跃会话数

用法:
    在 FastAPI app 中 mount metrics 路由:
        from src.observability.metrics import setup_metrics
        setup_metrics(app)
    然后访问 GET /metrics 获取 Prometheus 格式数据。
"""

import time
from contextlib import contextmanager

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

from src.core.logging import get_logger

logger = get_logger(__name__)

# ── 指标定义 ──────────────────────────────────

http_requests = Counter(
    "http_requests_total", "Total HTTP requests",
    ["method", "endpoint", "status"],
)

http_duration = Histogram(
    "http_request_duration_seconds", "HTTP request latency",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

llm_calls = Counter(
    "llm_calls_total", "Total LLM API calls",
    ["model", "agent"],
)

llm_tokens = Counter(
    "llm_token_usage_total", "Total LLM token usage",
    ["model", "type"],  # type: prompt / completion
)

retrieval_duration = Histogram(
    "retrieval_duration_seconds", "Retrieval latency",
    ["source"],  # source: bm25 / dense / hybrid
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5],
)

active_sessions = Gauge(
    "active_sessions", "Number of active chat sessions",
)

knowledge_base_docs = Gauge(
    "knowledge_base_documents", "Number of indexed documents",
)

# ── 辅助函数 ──────────────────────────────────

def record_http(method: str, endpoint: str, status: int, duration_s: float):
    http_requests.labels(method=method, endpoint=endpoint, status=str(status)).inc()
    http_duration.labels(method=method, endpoint=endpoint).observe(duration_s)


def record_llm(model: str, agent: str, prompt_tokens: int, completion_tokens: int):
    llm_calls.labels(model=model, agent=agent).inc()
    llm_tokens.labels(model=model, type="prompt").inc(prompt_tokens)
    llm_tokens.labels(model=model, type="completion").inc(completion_tokens)


def record_retrieval(source: str, duration_s: float):
    retrieval_duration.labels(source=source).observe(duration_s)


def set_active_sessions(count: int):
    active_sessions.set(count)


def set_kb_docs(count: int):
    knowledge_base_docs.set(count)


@contextmanager
def timed_http(method: str, endpoint: str):
    """HTTP 请求计时上下文管理器。"""
    t0 = time.time()
    status = 200
    try:
        yield
    except Exception:
        status = 500
        raise
    finally:
        record_http(method, endpoint, status, time.time() - t0)


def get_metrics_response():
    """返回 Prometheus 格式的指标数据。"""
    return generate_latest(), CONTENT_TYPE_LATEST
