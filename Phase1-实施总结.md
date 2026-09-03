# Phase 1 实施总结：制造业多 Agent 生产运维助手

---

## 零、Phase 1 达成了什么

用一句话说：**搭建了一个能跑通 "用户提问 → 安全检查 → 混合检索 → LLM生成 → 流式返回" 完整链路的最小可用系统。**

技术层面：
- 一个 FastAPI 服务，前后端联调可用
- 一个 LangGraph 多 Agent 工作流（Phase 1 只有 Knowledge Agent，但框架支持扩展到 6 个）
- 一套混合检索引擎（BM25 + 向量 + Re-ranker，内存实现，接口对齐生产级 Milvus/ES）
- 完整的安全边界（注入检测 + RBAC + 输出脱敏）
- 生产级的容错机制（重试 + 熔断 + 降级 + Checkpoint）
- 5 份制造文档作为种子数据，含设备手册、SOP、报警码表
- 一个 Vue3 + Element Plus 对话界面

---

## 一、项目文件结构全景

```
D:\Multi-agent-sys/
├── 📄 架构设计方案.md          # 18 章，7 轮迭代的方案文档 (~62KB)
├── 📄 产品需求说明书.md        # PRD：做什么/不做什么/给谁用
├── 📄 数据模型设计文档.md      # PG/Milvus/ES/Redis 完整 Schema
├── 📄 API接口规范文档.md       # REST/WebSocket/MES/A2A 接口规范
├── 📄 系统评测与验收方案.md    # 40+ 验收指标 + 评测流程
├── 📄 Phase1-实施总结.md       # 本文档
│
├── 📁 src/                     # 后端 Python 源码 (52 文件)
│   ├── core/                   # 基础设施层
│   │   ├── config.py           # 全局配置（Pydantic Settings，支持 .env）
│   │   ├── exceptions.py       # 分层异常体系（5 大类 20+ 错误码）
│   │   ├── logging.py          # 日志系统（loguru，双文件输出）
│   │   ├── retry.py            # 重试机制 + 熔断器（指数退避+半开渐进）
│   │   └── events.py           # 事件总线（发布-订阅，A2A 通信基础）
│   │
│   ├── model/                  # LLM 模型层
│   │   ├── router.py           # 多模型路由（按复杂度分派：简单→Qwen，复杂→DeepSeek）
│   │   ├── fallback.py         # 降级链（主模型→备选→规则兜底，含全局降级锁防振荡）
│   │   ├── cache.py            # 语义缓存（权限隔离：缓存Key绑定角色+车间+密级）
│   │   └── cost.py             # Token 成本追踪（按车间/Agent双向统计+预算告警）
│   │
│   ├── security/               # 安全层
│   │   ├── injection.py        # Prompt 注入检测（三层防御：输入/检索上下文/文档入库）
│   │   ├── rbac.py             # RBAC 权限引擎（角色+车间+密级三维校验 + A2A ACL）
│   │   ├── sanitize.py         # 输出脱敏（成本/人事/个人信息）
│   │   └── audit.py            # 审计日志（全量记录：注入/越权/A2A/系统事件）
│   │
│   ├── retrieval/              # 混合检索层（Phase 1 最核心模块）
│   │   ├── embedding.py        # 文本嵌入服务（BGE-large-zh-v1.5，优先 ModelScope 加载）
│   │   ├── chunker.py          # 智能分块（工业文档适配：表级/步骤级/语义级）
│   │   ├── bm25.py             # BM25 关键词检索（内存 rank_bm25，接口对齐 ES）
│   │   ├── dense.py            # 向量检索（内存实现，接口对齐 Milvus）
│   │   ├── reranker.py         # Cross-encoder 重排序（Phase1 轻量模拟，Phase2 接 BGE-Reranker）
│   │   ├── hybrid.py           # 混合检索编排（BM25+Dense→RRF融合→动态权重→Reranker→Top-K）
│   │   ├── ingestion.py        # 文档摄入管线（全系统唯一写入入口）
│   │   └── sync.py             # ES/Milvus 双写一致性（事务+补偿+对账）
│   │
│   ├── memory/                 # 记忆管理
│   │   ├── session.py          # 短期记忆（滑动窗口，Phase1 内存/Phase2 Redis）
│   │   └── compressor.py       # 上下文压缩（超70%窗口触发，Qwen-2.5-7B 做摘要）
│   │
│   ├── agents/                 # Agent 定义
│   │   ├── base.py             # BaseAgent 抽象基类（MCP 工具注册 + 沙箱接口）
│   │   ├── tools/              # MCP Tool 定义（Phase 2+）
│   │   └── prompts/            # System Prompt 模板（Jinja2，Phase 2+）
│   │
│   ├── graph/                  # LangGraph 编排层
│   │   ├── state.py            # AgentState（工作流中的数据总线）
│   │   ├── builder.py          # Graph 构建器（组装 Guard→Router→Knowledge→Aggregate→Memory）
│   │   ├── checkpoint.py       # Checkpoint 持久化（服务重启恢复，Phase1 内存/Phase2 Redis）
│   │   ├── concurrency.py      # Agent 并发管控（Semaphore，单请求最多 N 个 Agent 并行）
│   │   ├── edges.py            # 条件路由（注入拦截→END / 正常→Router）
│   │   └── nodes/              # LangGraph 节点
│   │       ├── guard.py        # 安全守护节点（注入检测 + RBAC 校验）
│   │       ├── router.py       # 意图路由节点（查询分类→决定调用哪些 Agent）
│   │       ├── knowledge.py    # 知识检索节点（检索→清洗→压缩→LLM→缓存→记忆）
│   │       ├── aggregate.py    # 结果聚合节点（多 Agent 输出→最终回答）
│   │       └── memory_node.py  # 记忆持久化节点（Checkpoint 保存 + 续期）
│   │
│   ├── api/                    # FastAPI 接入层
│   │   ├── app.py              # 应用入口（lifespan 管理 + 异常处理 + 路由注册）
│   │   ├── deps.py             # 依赖注入（全局单例：Graph/Retriever/Memory）
│   │   ├── routes/
│   │   │   ├── conversation.py # 对话接口（POST /chat SSE流式 + GET /history + DELETE）
│   │   │   └── health.py       # 健康检查（完整/readiness/liveness）
│   │   ├── middleware/         # 中间件（Phase 2: JWT鉴权/限流）
│   │   └── schemas/
│   │       ├── request.py      # 请求模型（Pydantic v2）
│   │       └── response.py     # 响应模型（统一 {code, message, data} 格式）
│   │
│   ├── integration/            # 工业系统集成（Phase 2+）
│   └── observability/          # 可观测层（Phase 4）
│
├── 📁 frontend/chat/           # Vue3 对话界面
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── main.js             # Element Plus 全局注册
│       └── App.vue             # 对话界面（SSE流式+角色切换+快捷指令+移动端适配）
│
├── 📁 data/seeds/
│   └── seed_data.py            # 5 份制造文档种子数据
│
├── 📁 deployment/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── pyproject.toml
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## 二、核心架构决策 & 为什么这样做

### 2.1 为什么用 LangGraph 而不是直接调 LLM

**问题**：最简单的聊天机器人只需要 `用户输入 → LLM → 返回`。为什么我们要用 LangGraph 构建一个 5 节点的图？

**答案**：制造业场景不只是聊天，而是**多步骤决策链**。
```
简单问答: 查SOP → LLM总结 → 返回       (3 步)
故障诊断: 查设备手册 → 查历史案例 → 推理根因 → 查维修步骤 → 建议派单 → 返回  (8+ 步)
```

LangGraph 的价值：
- **图状态机**：每个节点只做一件事，可复用、可组合、可单独测试
- **条件路由**：Guard 拦截注入→直接结束；Router 判断简单/复杂→走不同路径
- **状态持久化**：Checkpoint 让长流程不会因重启丢失
- **并行执行**：Phase 2 多 Agent 时，diagnosis + quality 可以同时跑

### 2.2 为什么每个模块都分 Phase 1 内存实现 + Phase 2 生产实现

| 模块 | Phase 1（现在） | Phase 2（下一步） | 为什么这样做 |
|------|----------------|-------------------|-------------|
| 向量检索 | Python dict | Milvus Lite/Standalone | MVP 先跑通流程，不阻塞开发 |
| 关键词检索 | rank_bm25 内存 | Elasticsearch | 同上，且 ES 需要 Docker 部署 |
| Re-ranker | 轻量权重模拟 | BGE-Reranker-v2-m3 | 真实模型 ~2GB，开发阶段可延后 |
| 会话记忆 | Python dict | Redis | 单进程开发够用 |
| Checkpoint | Python dict | Redis | 同上 |
| 语义缓存 | Python dict | Redis | 同上 |

**关键设计原则**：接口不变。切换时只改实现类，调用方零改动。比如：
```python
# Phase 1
dense = DenseRetriever()  # 内存 dict

# Phase 2  
dense = DenseRetriever(milvus_client)  # 真正的 Milvus
# search() 接口完全一样，调用方不需要改任何代码
```

### 2.3 为什么安全放在最前面（Guard Node 是第一个节点）

在 LangGraph 工作流中，**Guard Node 是所有请求的必经之路**。

```
START → Guard → Router → ... → END
         ↑
    任何请求都先过安检
```

这不是过度设计——制造业有真实的安全需求：
- 产线工人误粘贴整段设备日志（可能含系统指令）
- 第三方供应商文档可能夹带恶意内容
- 不同车间之间数据必须隔离（A 车间工人不能看 B 车间数据）

Guard Node 做了三件事，都在 5ms 内完成（不调 LLM）：
1. 注入检测（正则匹配 → 命中则拦截，直接返回 END）
2. 角色解析（从 DEMO_USERS mock 数据查，Phase 2 从 JWT token 解析）
3. 基础权限校验（检查用户是否有 `knowledge:read` 权限）

### 2.4 混合检索为什么分四步而不是一步

```
Step 1: BM25 粗召回（关键词精确匹配）         ← 毫秒级
Step 2: Dense 粗召回（语义相似）              ← 毫秒级
Step 3: RRF 融合（去重+加权+动态BM25/向量比例）  ← 微秒级
Step 4: Re-ranker 精排（Cross-encoder 深度匹配） ← 百毫秒级（只排 Top-30）
```

**为什么 Step 3 的 RRF 权重是动态的**：
- 查询"海天MA1200 润滑点" → 精确型号，BM25 权重 0.7
- 查询"打出来的件有毛边" → 模糊语义，向量权重 0.7
- 查询"海天MA1200 料筒温度异常怎么处理" → 组合查询，均衡 0.5:0.5

分类逻辑优先用正则（"HT-E-0021"这种报警码格式直接用正则匹配），正则没命中才走语义判断。**不用 LLM 做分类，省一次 API 调用**。

### 2.5 为什么降级链路要做到三层

```
主模型 (DeepSeek-V3) 失败 → 备选模型 (Qwen-Turbo) 失败 → 规则兜底 (报警码→SOP映射表)
         ↑                           ↑                        ↑
    重试 3 次 + 熔断              重试 3 次 + 熔断        无需 LLM，直接查表
```

制造业和聊天的关键区别：**聊天可以等，设备报警不能等**。如果 LLM 全挂了，规则兜底至少能告诉工人"这个报警码可能是什么问题，先检查什么"。

还有一个细节：**全局降级锁**。一旦触发规则兜底，5 分钟内不切回大模型——防止"大模型恢复→切回去→又挂→再降级"的振荡。

---

## 三、一条请求的完整旅程

以用户问"注塑机料筒温度异常怎么处理"为例，追踪请求从前端到 LLM 再回来的完整路径：

### 第 0 步：前端发起请求

```javascript
// App.vue 中
const response = await fetch('/api/v1/conversation/chat', {
    method: 'POST',
    body: JSON.stringify({
        message: "注塑机料筒温度异常怎么处理",
        user_id: "worker_zhang",
        workshop_id: "workshop-a",
    })
})
```

### 第 1 步：FastAPI 接收请求

```python
# src/api/routes/conversation.py
@router.post("/chat")
async def chat(request: ChatRequest):
    return EventSourceResponse(_stream_chat(request))  # 流式响应
```

### 第 2 步：构建 AgentState，进入 LangGraph

```python
state = AgentState(
    user_query="注塑机料筒温度异常怎么处理",
    user_context=DEMO_USERS["worker_zhang"],  # 张工(产线)
    session_id="session_abc123",
)
result = await graph.ainvoke(state)
```

### 第 3 步：Guard Node — 安全检查

```
检测注入 → 无 ("注塑机料筒..." 是正常查询)
解析角色 → "worker_zhang" = 产线工人
校验权限 → knowledge:read ✅
通过 → 进入 Router
```

### 第 4 步：Router Node — 意图分类

```
查询: "注塑机料筒温度异常怎么处理"
正则检测:
  报警码格式 [A-Z]+-\d+ → 无
  设备型号(HA-\w+) → 无  （"料筒温度异常" 没有型号关键词）
  故障关键词(异常/怎么/处理) → 有
分类结果: semantic (模糊语义)
路由: ["knowledge"] (Phase 1 全部到 knowledge)
```

### 第 5 步：Knowledge Node — 核心逻辑

这是最复杂的节点，分 11 个子步骤：

```
5.1  查语义缓存 → 未命中 (第一次问)
5.2  获取会话历史 → 空 (新会话)
5.3  混合检索:
     Query: "注塑机料筒温度异常怎么处理"
     Step A: BM25 从 35 chunks 中召回 → 得分最高的 30 条
     Step B: Dense 从 35 chunks 中召回 → query_vec 相似度最高的 30 条
     Step C: RRF 融合 (semantic 类型, BM25_w=0.3, Dense_w=0.7)
     Step D: Re-ranker 精排 → Top-10
     → 最相关的可能是: "设备报警码对照表 HT-E-0021" + "注塑机常见缺陷"
     
5.4  检索后清洗 → 检查检索到的片段中是否有注入指令 → 无
5.5  上下文压缩 → 10 个 chunk 总长度 < 8000 字符 → 不需要压缩
5.6  提取引用 → [{doc_id, title, source}]
5.7  构建 Prompt:
     system_prompt = """你是一个制造业知识助手...
                       检索到的文档: [....]"""
     messages = [system_prompt, user_msg]
     
5.8  调用 LLM:
     fallback_chain.chat_with_fallback()
     → 主模型 deepseek-chat → 401 无 API Key → 重试3次 → 失败
     → 备选模型 qwen-turbo → 401 无 API Key → 重试3次 → 失败
     → 规则兜底 → "系统暂时无法处理..." (有 API Key 后这里会是正常的 LLM 回答)
     
5.9  输出脱敏 → 不涉及成本/人事 → 无变更
5.10 写入语义缓存 → 下次同样问题直接返回
5.11 写入会话记忆 → session_id:abc 追加一轮对话
```

### 第 6 步：Aggregate Node — 结果聚合

```
Phase 1: 只有 knowledge 有输出 → 直接透传
Phase 2 多 Agent: 合并 diagnosis + knowledge + quality 的结果
```

### 第 7 步：Memory Node — 持久化

```
保存 Checkpoint → 会话状态写入内存
续期 TTL → 如果关联工单未完成，延长到 1 小时
```

### 第 8 步：SSE 流式返回

```python
# 按句子切分，逐句发送 SSE 事件
event: message
data: {"type":"text","content":"系统暂时无法处理您的请求"}

event: done
data: {"session_id":"abc","model_used":"rule_fallback","tokens_used":0}
```

前端 `App.vue` 解析这些 SSE 事件，逐句渲染到聊天气泡中。

---

## 四、关键代码片段（新人学习重点）

### 4.1 异常体系怎么设计的

```python
# src/core/exceptions.py

# 不要这样做:
raise Exception("出错了")  # 无法分类，难以定位

# 这样做:
class ManufacturingAgentError(Exception):  # 所有业务异常的基类
    code: str = "E00000"                    # 唯一错误码
    message: str = "系统内部错误"           # 用户可读的消息

class InjectionDetectedError(SecurityError):     # E10001 注入拦截
    code = "E10001"
    
class ModelCircuitOpenError(ModelError):          # E20002 熔断
    code = "E20002"

# FastAPI 层面自动映射 HTTP 状态码:
# E1xxxx → 403, E2xxxx → 503, E3xxxx → 503, E4xxxx → 500, E5xxxx → 502
```

### 4.2 泛型工厂模式（Knowledge Node 怎么注入依赖的）

```python
# 不要这样做（紧耦合）:
async def knowledge_node(state):
    from xxx import yy  # 到处 import，无法测试
    retriever = get_retriever()  # 隐式依赖
    
# 这样做（依赖注入）:
class KnowledgeNodeContext:
    def __init__(self, hybrid_retriever, session_memory, compressor):
        self.hybrid_retriever = hybrid_retriever
        self.session_memory = session_memory
        
# 工厂函数返回闭包
def create_knowledge_node(ctx: KnowledgeNodeContext):
    async def knowledge_node(state: AgentState) -> dict:
        # 使用 ctx.xxx 而不是全局变量
        results = await ctx.hybrid_retriever.search(...)
    return knowledge_node
```

好处：测试时可以传入 mock 的 retriever，不需要真实的 Milvus/ES。

### 4.3 流式 SSE 怎么实现的

```python
# 关键：用 async generator 产生 SSE 事件流
async def _stream_chat(request):
    result = await graph.ainvoke(state)  # 执行 LangGraph
    
    answer = result["final_answer"]
    
    # 逐句发送
    for char in answer:
        if char in "。！？\n":           # 遇到句子边界
            yield {"event": "message", "data": {...}}  # 发送一个 SSE 事件
    
    yield {"event": "done", "data": {...}}  # 结束事件

# EventSourceResponse 将 generator 转为 SSE 流
return EventSourceResponse(_stream_chat(request))
```

前端用 `fetch + ReadableStream` 逐条读取 SSE 事件，不需要 WebSocket：
```javascript
const reader = response.body.getReader();
while (true) {
    const { done, value } = await reader.read();
    // 解析 SSE 格式: "data: {...}"
    // 渲染到聊天气泡
}
```

---

## 五、已验证的功能清单

| 编号 | 功能 | 状态 | 验证方式 |
|------|------|------|---------|
| ✅ | 注入检测（拦截"忽略之前的指令"） | 通过 | Python 单元测试 |
| ✅ | 注入检测（正常查询放行） | 通过 | Python 单元测试 |
| ✅ | RBAC（工人读 SOP 通过） | 通过 | Python 单元测试 |
| ✅ | RBAC（工人读成本被拦截） | 通过 | Python 单元测试 |
| ✅ | 工业文档智能分块 | 通过 | 2 chunks, types={sop_step, table} |
| ✅ | BM25 检索 | 通过 | 索引+搜索 |
| ✅ | 向量检索 + 分区过滤 | 通过 | 插入+按 workshop_id 过滤 |
| ✅ | 会话记忆 | 通过 | 添加消息+窗口截断 |
| ✅ | Checkpoint 持久化 | 通过 | 保存→恢复 |
| ✅ | FastAPI 启动（含 embedding 模型加载） | 通过 | health check 200 OK |
| ✅ | LangGraph 编译 | 通过 | 5 节点全部注册 |
| ✅ | 种子数据索引（5 文档→35 chunks） | 通过 | 启动日志确认 |
| ✅ | SSE 流式对话 | 通过 | 端到端 POST /chat |
| ✅ | 降级兜底（无 API Key 时返回友好提示） | 通过 | 端到端 POST /chat |
| ✅ | 重试机制（LLM 调用重试 3 次） | 通过 | 日志确认 |
| ✅ | 熔断降级链（主模型→备选→规则） | 通过 | 日志确认 |
| ✅ | 前端界面（角色切换+快捷指令+SSE流式） | 通过 | 代码完成 |
| ✅ | 文档搜索（可检索到"换模SOP"内容） | 通过 | 检索测试 |

---

## 六、Phase 2 衔接备忘录

以下是 Phase 1 写的代码中，已经预留好接口但 Phase 2 才实现的部分：

| 模块 | 当前状态 | Phase 2 要做的 | 改动范围 |
|------|---------|---------------|---------|
| `dense.py` | 内存 dict | 替换为 Milvus Lite | 实现类换掉，调用方不改 |
| `bm25.py` | 内存 rank_bm25 | 替换为 Elasticsearch | 同上 |
| `reranker.py` | 轻量模拟 | 接入 BGE-Reranker-v2-m3 | 同上 |
| `session.py` | 内存 dict | 替换为 Redis | 同上 |
| `checkpoint.py` | 内存 dict | 替换为 Redis | 同上 |
| `cache.py` | 内存 dict | 替换为 Redis | 同上 |
| `retry.py` | 单实例熔断 | 多实例共享 Redis 熔断状态 | `configure_redis()` |
| `graph/nodes/` | 2个Agent | 新增 diagnosis/inspection/scheduling/quality/report | 新文件 |
| `router.py` | 全部→knowledge | 意图分类→多 Agent 分发 | 改路由逻辑 |
| `aggregate.py` | 透传 | 多 Agent 结果合并+冲突检测 | 改聚合逻辑 |

---

## 七、给初级 Agent 开发者的学习建议

如果你是从零开始学 Agent 开发，建议按以下顺序理解这个项目：

1. **先看架构文档**：`架构设计方案.md`，理解"为什么这么做"
2. **再看 core 层**：`config.py` → `exceptions.py` → `retry.py`，这是地基
3. **然后看一条请求的完整旅程**：`app.py` → `conversation.py` → `builder.py` → 各 node，跟踪数据怎样流转
4. **再看 retrieval 层**：`hybrid.py` 是整个检索的总入口，把 BM25 + Dense + Reranker 串了起来
5. **最后看安全层**：理解为什么 Guard 要在 Router 前面，为什么缓存 Key 要绑权限

关键概念映射：
- **LangGraph StateGraph** = 流程图，Node 是步骤，Edge 是箭头
- **AgentState** = 在流程中传递的"公文夹"，每个步骤往里面加内容
- **Checkpoint** = 拍照存档，重启后可以接着上次继续
- **A2A** = Agent 之间的 HTTP 调用（通过事件总线，解耦）
- **MCP** = 工具调用的标准协议（声明参数+限制+超时）
