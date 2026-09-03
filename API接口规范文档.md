# API 接口规范文档

## 制造业多 Agent 生产运维数字助手

---

## 一、通用规范

### 1.1 接口前缀

```
开发环境: http://localhost:8000/api/v1
生产环境: https://{factory-host}/api/v1
```

### 1.2 鉴权方式

所有接口（除健康检查）需在 Header 中携带 JWT Token：

```
Authorization: Bearer {jwt_token}
```

JWT Payload 结构：
```json
{
  "sub": "user-uuid",
  "role": "maintainer",
  "workshop_id": "workshop-a",
  "display_name": "张工",
  "exp": 1718000000
}
```

### 1.3 统一响应格式

成功响应：
```json
{
  "code": 0,
  "message": "success",
  "data": { ... }
}
```

列表响应：
```json
{
  "code": 0,
  "message": "success",
  "data": {
    "items": [ ... ],
    "total": 100,
    "page": 1,
    "page_size": 20
  }
}
```

错误响应：
```json
{
  "code": 40001,
  "message": "权限不足：无法访问B车间数据",
  "detail": null
}
```

### 1.4 全局错误码

| 错误码 | HTTP状态码 | 说明 |
|--------|-----------|------|
| 0 | 200 | 成功 |
| 40001 | 403 | RBAC 权限不足 |
| 40002 | 403 | 跨车间访问被拒绝 |
| 40003 | 403 | 文档密级不足 |
| 40010 | 400 | Prompt注入被拦截 |
| 40011 | 400 | 请求参数校验失败 |
| 40100 | 401 | 未登录或Token过期 |
| 40101 | 401 | A2A签名校验失败 |
| 40400 | 404 | 资源不存在 |
| 42900 | 429 | 请求频率超限 |
| 50001 | 500 | LLM调用失败（已降级） |
| 50002 | 500 | 检索服务不可用 |
| 50003 | 500 | MES服务不可用 |
| 50010 | 500 | 系统内部错误 |

---

## 二、对外 REST 接口

### 2.1 对话接口

#### POST /api/v1/conversation/chat

发送对话消息，返回流式 SSE 响应。

**Request:**
```json
{
  "session_id": "uuid-optional",         // 不传则新建会话
  "message": "注塑机HA-003料筒温度异常怎么处理",
  "context": {                           // 可选上下文
    "equipment_id": "HA-003",
    "fault_code": "HT-E-0021"
  }
}
```

**Response:** SSE 流式 (Content-Type: text/event-stream)
```
event: message
data: {"type":"text","content":"根据","step":1}

event: message
data: {"type":"text","content":"SOP-HA-12","step":2}

event: citation
data: {"doc_id":"sop-ha-12","title":"海天MA1200换模指导书","chunk_index":3}

event: done
data: {"session_id":"uuid-xxx","tokens_used":456,"model":"deepseek-v3","latency_ms":2800}
```

SSE 事件类型：
| event | 说明 |
|-------|------|
| message | 文本增量，逐字输出 |
| citation | 引用来源，可多个 |
| error | 错误信息 |
| done | 对话完成 |

#### GET /api/v1/conversation/history

获取历史对话列表。

**Query:** `?session_id=xxx&page=1&page_size=20`

**Response:**
```json
{
  "code": 0,
  "data": {
    "session_id": "uuid-xxx",
    "messages": [
      {
        "role": "user",
        "content": "料筒温度异常怎么处理",
        "created_at": "2026-07-17T14:30:00Z"
      },
      {
        "role": "assistant",
        "content": "根据SOP-HA-12，处理步骤如下...",
        "citations": [{ "doc_id": "...", "title": "..." }],
        "created_at": "2026-07-17T14:30:05Z"
      }
    ]
  }
}
```

#### DELETE /api/v1/conversation/{session_id}

删除指定会话及关联 Checkpoint。

---

### 2.2 知识库管理接口

#### POST /api/v1/knowledge/documents

上传文档。

**Request:** multipart/form-data
```
file: {binary}
doc_type: "equipment_manual"
workshop_id: "workshop-a"
equipment_model: "HA-MA1200"
classification: "internal"
```

**Response:**
```json
{
  "code": 0,
  "data": {
    "doc_id": "doc-uuid-xxx",
    "status": "ingesting",
    "chunk_count_estimated": 15
  }
}
```

#### GET /api/v1/knowledge/documents

文档列表。

**Query:** `?doc_type=equipment_manual&workshop_id=workshop-a&status=active&page=1&page_size=20`

#### GET /api/v1/knowledge/documents/{doc_id}

文档详情。

#### PUT /api/v1/knowledge/documents/{doc_id}

更新文档元数据（不重新索引）。

#### DELETE /api/v1/knowledge/documents/{doc_id}

删除文档（ES + Milvus 双索引同时删除，关联缓存失效）。

#### GET /api/v1/knowledge/search

管理后台检索测试接口。

**Query:**
```
?query=注塑机换模步骤&workshop_id=workshop-a&hybrid=true&top_k=10
```

**Response:**
```json
{
  "code": 0,
  "data": {
    "results": [
      {
        "rank": 1,
        "chunk_id": "...",
        "doc_title": "换模作业指导书",
        "content_snippet": "...",
        "bm25_score": 0.85,
        "dense_score": 0.92,
        "fusion_score": 0.91,
        "reranker_score": 0.95,
        "source": "sop"
      }
    ],
    "latency_ms": 120
  }
}
```

---

### 2.3 工单回调接口（MES Webhook）

#### POST /api/v1/webhook/mes/work-order-status

MES 工单状态回调（幂等）。

**Request Header:**
```
X-MES-Signature: HMAC-SHA256(body, shared_secret)
X-MES-Event-ID: evt-uuid-xxx
```

**Request:**
```json
{
  "mes_event_id": "evt-uuid-xxx",
  "work_order_id": "WO-2026-0042",
  "new_status": "completed",
  "operator": "张工",
  "timestamp": "2026-07-17T15:00:00Z",
  "remark": "已更换料筒传感器，温度恢复正常"
}
```

**Response:**
```json
{
  "code": 0,
  "message": "ok"
}
```

幂等处理：`mes_event_id` 首次处理返回 200，重复推送直接返回 200（查 Redis 去重）。

---

### 2.4 报表接口

#### GET /api/v1/report/oee

查询 OEE 数据。

**Query:** `?workshop_id=workshop-a&date_from=2026-07-01&date_to=2026-07-17`

**Response:**
```json
{
  "code": 0,
  "data": {
    "workshop_id": "workshop-a",
    "period": { "from": "2026-07-01", "to": "2026-07-17" },
    "oee": 87.3,
    "availability": 93.1,
    "performance": 91.5,
    "quality": 97.2,
    "daily_trend": [
      { "date": "2026-07-01", "oee": 88.1 },
      { "date": "2026-07-02", "oee": 86.5 }
    ]
  }
}
```

#### GET /api/v1/report/alarm-stats

告警统计。

**Query:** `?workshop_id=workshop-a&date_from=2026-07-01&date_to=2026-07-17`

#### GET /api/v1/report/token-usage

Token 消耗统计。

**Query:** `?workshop_id=workshop-a&date_from=2026-07-01&date_to=2026-07-17&group_by=agent`

---

### 2.5 健康检查

#### GET /api/v1/health

```json
{
  "code": 0,
  "data": {
    "status": "healthy",
    "version": "1.0.0",
    "mode": "online",
    "checks": {
      "postgresql": "ok",
      "redis": "ok",
      "milvus": "ok",
      "elasticsearch": "ok",
      "deepseek_api": "ok",
      "qwen_api": "ok"
    }
  }
}
```

#### GET /api/v1/health/readiness

K8s Readiness Probe，检查依赖服务。

---

## 三、WebSocket 流式接口

### 3.1 WS /api/v1/ws/chat

对话的 WebSocket 备选通道（除 SSE 外）。

**连接:**
```
ws://localhost:8000/api/v1/ws/chat?token={jwt_token}
```

**客户端→服务端 (发送消息):**
```json
{
  "type": "message",
  "session_id": "uuid-optional",
  "content": "注塑机怎么换模具",
  "context": { "equipment_id": "HA-003" }
}
```

**客户端→服务端 (停止生成):**
```json
{
  "type": "stop",
  "session_id": "uuid-xxx"
}
```

**服务端→客户端 (流式文本):**
```json
{
  "type": "text",
  "content": "根据SOP-HA-12",
  "step": 1
}
```

**服务端→客户端 (完成):**
```json
{
  "type": "done",
  "session_id": "uuid-xxx",
  "tokens_used": 456
}
```

### 3.2 心跳机制

```
客户端每30秒发送: {"type":"ping"}
服务端回复:       {"type":"pong"}
客户端60秒未收到pong → 触发重连（指数退避: 1s→2s→4s→8s, 最大30s）
```

---

## 四、MES 对接接口

### 4.1 本系统 → MES（工单创建）

#### POST {MES_BASE_URL}/api/work-orders

**Request:**
```json
{
  "idempotent_key": "mes_event_id-uuid",
  "source": "manufacturing-agent",
  "workshop_id": "workshop-a",
  "equipment_id": "HA-003",
  "equipment_model": "海天MA1200",
  "fault_code": "HT-E-0021",
  "fault_desc": "料筒温度传感器异常",
  "priority": "P1",
  "diagnosis": "传感器故障，建议更换",
  "created_by": "诊断Agent",
  "callback_url": "https://{agent-host}/api/v1/webhook/mes/work-order-status"
}
```

**Response (201):**
```json
{
  "work_order_id": "WO-2026-0042",
  "status": "created",
  "created_at": "2026-07-17T14:32:00Z"
}
```

### 4.2 MES 对接规范

| 配置项 | 值 |
|--------|-----|
| 鉴权方式 | HMAC-SHA256 签名 |
| 超时 | 连接 5s / 读取 30s |
| 重试 | 指数退避 3 次 |
| Webhook回调 | POST JSON |
| 签名Header | X-MES-Signature |

---

## 五、内部 A2A 接口

### 5.1 A2A 调用协议

Agent 间调用不经过 HTTP，而是通过内部事件总线（`core/events.py`），但对外暴露为可观测的调用记录。

**调用格式（内部消息对象）:**
```python
A2AMessage(
    message_id="uuid",
    caller_agent="diagnosis",
    caller_user="user-uuid",
    target_agent="scheduling",
    auth_token="jwt-with-role-and-scope",
    timestamp=datetime.now(),
    nonce="random-string",
    payload={
        "intent": "check_available_window",
        "equipment_id": "HA-003",
        "estimated_repair_minutes": 30
    },
    signature="HMAC-SHA256(message_id+caller_agent+target_agent+timestamp+nonce, secret)"
)
```

**返回格式:**
```python
{
    "message_id": "uuid",          # 对应请求的 message_id
    "target_agent": "scheduling",
    "status": "success",           # success / timeout / rejected / error
    "data": {
        "available_window": "2026-07-17 15:00-15:30",
        "impact": "预计影响产能5%"
    },
    "latency_ms": 450
}
```

### 5.2 A2A 错误码

| 错误码 | 说明 |
|--------|------|
| A2A_OK | 成功 |
| A2A_TIMEOUT | 调用超时(>30s) |
| A2A_REJECTED | 被调用方拒绝(权限/ACL) |
| A2A_CIRCUIT_OPEN | 被调用方已熔断 |
| A2A_LOOP_DETECTED | 检测到死循环 |
| A2A_RATE_LIMITED | 超出并发限制(排队超时) |
| A2A_ERROR | 内部错误 |

---

## 六、请求限制

| 限制项 | 值 |
|--------|-----|
| 单次对话最大Token | 8K (简单任务) / 32K (复杂诊断) |
| API 限流 (每用户) | 30 次/分钟 (对话), 10 次/分钟 (上传) |
| 单次上传文档大小 | 50MB |
| 单次检索最大返回 | Top-20 (Reranker后) |
| SSE 连接超时 | 120s |
| 单次A2A调用超时 | 30s |
| 全局A2A并发上限 | 20 |
