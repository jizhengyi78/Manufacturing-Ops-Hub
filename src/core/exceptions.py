"""
exceptions.py — 自定义异常体系
============================
角色：全系统的异常分类和错误码定义。每个异常类对应一个错误码，
      配合 FastAPI exception_handler 自动转换为 HTTP 响应。

异常分类 (按错误码前缀):
- E1xxxx: 安全相关 (注入/越权/密级)
- E2xxxx: 模型/LLM 相关 (超时/熔断/全挂)
- E3xxxx: 检索相关 (Milvus/ES/摄入/双写)
- E4xxxx: Agent/Graph 相关 (A2A超时/死循环/限流)
- E5xxxx: 集成相关 (MES/工单)

使用方式:
    from src.core.exceptions import InjectionDetectedError
    raise InjectionDetectedError("检测到恶意输入")

在 API 层会被全局 exception_handler 捕获，转换为:
    {"code": "E10001", "message": "检测到注入攻击", "detail": {...}}

注意事项:
- 不要在这里捕获异常，这层只定义异常类型
- 新增异常时确保错误码不重复
- detail 字段用于存放额外上下文 (如哪个模式被命中)
"""

# ── 基础异常 (E0xxxx) ──────────────────────────
class ManufacturingAgentError(Exception):
    """所有业务异常的基类。
    不要直接抛这个，抛它的子类。"""
    code: str = "E00000"
    message: str = "系统内部错误"

    def __init__(self, message: str | None = None, detail: dict | None = None):
        self.message = message or self.message
        self.detail = detail or {}
        super().__init__(self.message)


# ── Security (E1xxxx) ──────────────────────────
class SecurityError(ManufacturingAgentError):
    code = "E10000"

class InjectionDetectedError(SecurityError):
    code = "E10001"
    message = "检测到注入攻击，请求已拦截"

class PermissionDeniedError(SecurityError):
    code = "E10002"
    message = "权限不足"

class CrossWorkshopAccessError(SecurityError):
    code = "E10003"
    message = "跨车间访问被拒绝"

class ClassificationDeniedError(SecurityError):
    code = "E10004"
    message = "文档密级不足，无权访问"


# ── Model / LLM (E2xxxx) ───────────────────────
class ModelError(ManufacturingAgentError):
    code = "E20000"

class ModelTimeoutError(ModelError):
    code = "E20001"
    message = "LLM 调用超时"

class ModelCircuitOpenError(ModelError):
    code = "E20002"
    message = "模型已熔断，使用降级方案"

class AllModelsFailedError(ModelError):
    code = "E20003"
    message = "所有模型不可用，启用规则降级"


# ── Retrieval (E3xxxx) ─────────────────────────
class RetrievalError(ManufacturingAgentError):
    code = "E30000"

class MilvusUnavailableError(RetrievalError):
    code = "E30001"
    message = "Milvus 服务不可用"

class ESUnavailableError(RetrievalError):
    code = "E30002"
    message = "Elasticsearch 服务不可用"

class IngestionError(RetrievalError):
    code = "E30003"
    message = "文档摄入失败"

class DualWriteConsistencyError(RetrievalError):
    code = "E30004"
    message = "双写不一致，已触发补偿"


# ── Agent / Graph (E4xxxx) ─────────────────────
class AgentError(ManufacturingAgentError):
    code = "E40000"

class A2ATimeoutError(AgentError):
    code = "E40001"
    message = "A2A 调用超时"

class A2ALoopDetectedError(AgentError):
    code = "E40002"
    message = "检测到 A2A 死循环，已切断"

class A2ARateLimitedError(AgentError):
    code = "E40003"
    message = "A2A 并发已满，请求排队超时"

class GraphCheckpointError(AgentError):
    code = "E40004"
    message = "Graph Checkpoint 操作失败"


# ── Integration / MES (E5xxxx) ─────────────────
class IntegrationError(ManufacturingAgentError):
    code = "E50000"

class MESUnavailableError(IntegrationError):
    code = "E50001"
    message = "MES 服务不可用"

class MESSyncError(IntegrationError):
    code = "E50002"
    message = "MES 同步异常，需人工对账"
