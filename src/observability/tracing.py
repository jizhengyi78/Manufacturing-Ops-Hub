"""
tracing.py — OpenTelemetry 全链路追踪
======================================
Phase 4: 分布式追踪 — API → Graph → Agent → LLM → 返回。

Trace 层级:
  HTTP Request (/api/v1/conversation/chat)
    └── Graph Execution (guard → router → knowledge → ...)
          ├── Hybrid Retrieval (BM25 + Dense)
          │     ├── Milvus Search
          │     └── ES/BM25 Search
          ├── Reranker
          └── LLM Call (DeepSeek/Qwen/Fallback)

用法:
    from src.observability.tracing import setup_tracing, get_tracer
    setup_tracing()
    tracer = get_tracer("knowledge")
    with tracer.start_as_current_span("retrieval") as span:
        span.set_attribute("query", query)
        results = await hybrid.search(...)
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource

from src.core.logging import get_logger

logger = get_logger(__name__)

_tracer_provider = None


def setup_tracing(service_name: str = "manufacturing-agent", console_export: bool = True):
    """初始化 OpenTelemetry 追踪。

    开发环境: ConsoleSpanExporter (终端输出)
    生产环境: OTLPSpanExporter (→ Jaeger/Zipkin/Grafana Tempo)
    """
    global _tracer_provider
    resource = Resource.create({"service.name": service_name, "service.version": "0.1.0"})
    _tracer_provider = TracerProvider(resource=resource)

    if console_export:
        _tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    # 生产环境: 接 OTLP Collector
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        import os
        otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
        if otlp_endpoint:
            _tracer_provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint))
            )
            logger.info(f"OTLP tracing enabled: {otlp_endpoint}")
    except ImportError:
        pass  # OTLP not installed, skip

    trace.set_tracer_provider(_tracer_provider)
    logger.info(f"OpenTelemetry tracing initialized (service={service_name})")


def get_tracer(name: str) -> trace.Tracer:
    """获取模块级 Tracer。

    示例:
        tracer = get_tracer("knowledge_node")
        with tracer.start_as_current_span("hybrid_search") as span:
            span.set_attribute("query.length", len(query))
            ...
    """
    return trace.get_tracer(name)


def shutdown_tracing():
    """关闭追踪，刷新剩余 span。"""
    global _tracer_provider
    if _tracer_provider:
        _tracer_provider.shutdown()
        logger.info("OpenTelemetry tracing shutdown")
