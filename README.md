# 制造业多 Agent 生产运维数字助手

基于 LangGraph 多 Agent 架构的离散制造生产运维 AI 助手，覆盖注塑/冲压/CNC/装配等通用制造场景。

## 业务闭环

```
设备报警 → 故障诊断 Agent（根因分析）→ 知识库 Agent（维修 SOP 匹配）
→ 排程 Agent（产能影响评估）→ 工单推送 → 维修执行
→ 质检 Agent（首检确认）→ 报表 Agent（OEE/良率更新）→ 知识沉淀（写入经验库）
```

## 技术栈

| 层级 | 技术 |
|------|------|
| Agent 框架 | LangGraph（图状态机 + Checkpoint） |
| API | FastAPI + WebSocket（SSE 流式） |
| 向量检索 | Milvus Lite + BM25（rank-bm25）混合检索 + BGE-Reranker |
| 嵌入模型 | BAAI/bge-large-zh-v1.5 / bge-small-zh-v1.5 |
| 数据库 | SQLite（开发）/ PostgreSQL（生产） |
| 缓存 / 会话 | Redis |
| LLM 主模型 | DeepSeek-chat（可配置） |
| LLM 降级 | 多模型路由 + 熔断器 + 规则兜底 |
| OCR | EasyOCR（中英文） |
| 前端 | Vue3 + Element Plus + Vite |
| 可观测 | OpenTelemetry + Prometheus + Grafana |
| 部署 | Docker Compose / systemd + nginx |

## 目录结构

```
manufacturing-agent/
├── src/
│   ├── api/                    # FastAPI 接入层
│   │   ├── app.py              # 应用入口 & 生命周期
│   │   ├── deps.py             # 依赖注入
│   │   ├── routes/             # 路由
│   │   │   ├── conversation.py # 对话（SSE 流式 + 图片聊天）
│   │   │   ├── knowledge.py    # 知识库管理（上传/搜索/删除）
│   │   │   ├── admin.py        # 管理（种子数据/种子用户）
│   │   │   ├── auth.py         # JWT 认证
│   │   │   └── health.py       # 健康检查
│   │   ├── middleware/         # 中间件
│   │   │   ├── auth.py         # JWT 鉴权
│   │   │   └── ratelimit.py    # API 限流
│   │   └── schemas/            # Pydantic 请求/响应模型
│   ├── graph/                  # LangGraph 编排
│   │   ├── state.py            # AgentState 定义
│   │   ├── builder.py          # 图构建
│   │   ├── checkpoint.py       # 检查点持久化
│   │   ├── nodes/              # 工作流节点
│   │   │   ├── guard.py        # 安全守护（注入检测 + RBAC）
│   │   │   ├── router.py       # 意图路由
│   │   │   ├── knowledge.py    # 知识检索
│   │   │   ├── diagnosis.py    # 故障诊断
│   │   │   ├── inspection.py   # 巡检预警
│   │   │   ├── scheduling.py   # 排程优化
│   │   │   ├── quality.py      # 质量分析
│   │   │   ├── report.py       # 数据报表
│   │   │   ├── aggregate.py    # 多 Agent 结果聚合
│   │   │   └── memory_node.py  # 记忆持久化
│   │   └── edges.py            # 条件边/路由逻辑
│   ├── agents/                 # Agent 定义
│   │   ├── base.py             # BaseAgent + ToolDefinition
│   │   ├── tools/              # MCP 工具注册
│   │   └── prompts/            # System Prompt 模板（Jinja2）
│   ├── retrieval/              # 混合检索
│   │   ├── embedding.py        # 嵌入服务（BGE 模型）
│   │   ├── bm25.py             # BM25 关键词检索
│   │   ├── dense.py            # 向量检索（Milvus Lite / 内存）
│   │   ├── reranker.py         # Cross-encoder 重排序
│   │   ├── hybrid.py           # 混合检索融合（RRF）
│   │   ├── chunker.py          # 语义分块（表级/SOP步骤/标题）
│   │   ├── ingestion.py        # 文档摄入管线（PDF/Word/Excel）
│   │   └── sync.py             # 双写一致性
│   ├── memory/                 # 记忆管理
│   │   ├── session.py          # 短期记忆（Redis / 内存）
│   │   ├── persistent.py       # 长期记忆（Milvus）
│   │   └── compressor.py       # LLM 上下文压缩
│   ├── model/                  # 模型层
│   │   ├── router.py           # 多模型路由（复杂度分类）
│   │   ├── fallback.py         # 降级策略（主→备→规则兜底）
│   │   ├── cache.py            # 语义缓存
│   │   ├── cost.py             # Token 成本追踪
│   │   ├── concurrency.py      # LLM 调用排队
│   │   └── rule_engine.py      # 规则引擎兜底
│   ├── security/               # 安全
│   │   ├── injection.py        # Prompt Injection 检测
│   │   ├── sanitize.py         # 输出脱敏
│   │   ├── rbac.py             # 基于角色的权限控制
│   │   └── audit.py            # 审计日志
│   ├── observability/          # 可观测
│   │   ├── tracing.py          # OpenTelemetry 追踪
│   │   ├── langfuse.py         # LLM 专属观测
│   │   └── metrics.py          # Prometheus 指标
│   ├── integration/            # 外部集成
│   │   └── ocr.py              # EasyOCR 图片文字提取
│   └── core/                   # 基础设施
│       ├── config.py           # 全局配置（Pydantic Settings）
│       ├── database.py         # 数据库引擎（SQLAlchemy async）
│       ├── redis_client.py     # Redis 客户端
│       ├── retry.py            # 重试 + 熔断器（指数退避）
│       ├── events.py           # 事件总线
│       ├── exceptions.py       # 自定义异常
│       └── logging.py          # 日志配置
├── frontend/
│   ├── chat/                   # 智能助手前端（Vue3）
│   └── admin/                  # 知识库管理后台（Vue3）
├── data/
│   ├── seeds/                  # 种子数据（设备手册/SOP/故障案例）
│   ├── documents/              # 文档存储
│   ├── uploads/                # 上传文件
│   └── finetune/               # 微调数据集
├── evaluation/                 # 评测体系
│   ├── runner.py               # 评测执行器
│   ├── report.json             # 评测报告
│   └── benchmark/              # 测试集（故障诊断/SOP问答/注入测试）
├── deployment/
│   ├── docker-compose.yml      # 一键部署
│   ├── Dockerfile              # 后端镜像
│   ├── Dockerfile.frontend     # 前端镜像
│   ├── nginx.conf              # Nginx 配置
│   └── grafana_dashboard.json  # Grafana 监控面板
├── scripts/
│   ├── start.sh / start.bat    # 开发启动
│   ├── deploy.sh               # 服务器部署
│   └── finetune/               # LoRA 微调脚本
└── tests/                      # 单元测试 & 集成测试
```

## 快速开始

### 环境要求

- Python >= 3.10
- Node.js >= 18
- Redis（可选，不可用时自动降级为内存模式）

### 1. 安装依赖

```bash
# 后端
pip install -e .

# 前端
cd frontend/chat && npm install
cd frontend/admin && npm install
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，至少填写：
#   DEEPSEEK_API_KEY=sk-xxxxx
#   DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
```

### 3. 启动开发服务

```bash
# 后端（端口 8000）
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# 智能助手前端（端口 5173）
cd frontend/chat && npm run dev

# 管理后台前端（端口 5174）
cd frontend/admin && npm run dev
```

### 4. Docker Compose 部署

```bash
cd deployment
docker-compose up -d
```

## 生产部署

服务器部署脚本：`scripts/deploy.sh`

```bash
# 在服务器上执行
bash scripts/deploy.sh
```

部署后访问：
- 智能助手：`http://<服务器IP>`
- 管理后台：`http://<服务器IP>/admin`
- API 文档：`http://<服务器IP>/docs`
- 健康检查：`http://<服务器IP>/api/v1/health`
- Prometheus 指标：`http://<服务器IP>/metrics`

## 默认用户

演示环境预置 9 个用户，密码通过环境变量 `DEMO_PASSWORD` 配置。

| 用户名 | 角色 | 车间 |
|--------|------|------|
| worker_zhang | 产线工人 | A |
| maintainer_li | 维修工 | A |
| shift_lead_wang | 班组长 | A |
| director_zhao | 车间主任 | A |
| engineer_chen | 工艺工程师 | A |
| manager_zhou | 厂长 | 全部 |
| worker_sun | 产线工人 | B |
| maintainer_huang | 维修工 | B |
| shift_lead_liu | 班组长 | B |

## 功能特性

### 多 Agent 协作

工作流：Guard（安全检测）→ Router（意图分类）→ Specialist Agents（并行）→ Aggregate（结果聚合）→ Memory（持久化）

| Agent | 功能 |
|-------|------|
| Knowledge | 混合检索（BM25 + 向量 + Re-ranker），从知识库召回相关文档 |
| Diagnosis | 故障根因分析，结合检索结果和设备报警码推理 |
| Inspection | 巡检预警，分析设备巡检数据给出建议 |
| Scheduling | 排程优化，评估设备故障对产能的影响 |
| Quality | 质量分析，关联缺陷数据和工艺参数 |
| Report | 数据报表，生成 OEE/良率等统计报告 |

### 混合检索

```
Query → 查询分类 → [BM25 + Dense(Milvus)] → RRF 融合 → BGE-Reranker → Top-K
```

- **精确匹配**（型号/报警码）：BM25 权重 0.7
- **混合查询**（型号 + 描述）：BM25:Dense = 0.5:0.5
- **语义查询**（故障描述）：Dense 权重 0.7
- 支持 PDF / Word / Excel / TXT / MD / CSV 上传

### 记忆系统

| 类型 | 存储 | 用途 |
|------|------|------|
| 短期记忆 | Redis List（滑动窗口） | 当前会话上下文 |
| 长期记忆 | Milvus（向量） | 跨会话故障经验召回 |
| 上下文压缩 | LLM 摘要 | Token 超阈值自动触发 |
| 知识沉淀 | 自动评估 → 写入经验库 | 高质量对话自动归档 |

### 安全防护

- **Prompt Injection 检测**：正则 + 关键词 + 模式匹配
- **RBAC**：6 种角色（工人/维修工/班组长/车间主任/工艺工程师/厂长），车间级隔离
- **JWT 认证**：bcrypt 密码哈希，可配置过期时间
- **输出脱敏**：成本数据/人员信息自动脱敏
- **审计日志**：全操作记录
- **API 限流**：基于 IP + 用户的双重限流

### 模型路由 & 降级

```
简单任务 → Qwen-turbo（便宜）
复杂推理 → DeepSeek-chat（强推理）
报表生成 → 规则引擎（不走 LLM）

降级链：主模型 → 备选模型 → 规则引擎兜底
熔断器：连续失败 5 次 → 熔断 30s → 半开探测 → 恢复/重熔
```

### 可观测

- **OpenTelemetry**：全链路追踪（请求 → 路由 → 检索 → LLM → 响应）
- **Prometheus**：请求 QPS、延迟、Token 消耗、检索延迟
- **Grafana**：预配置监控面板
- **Langfuse**：LLM 专属观测（可选）

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/auth/login` | 用户登录 |
| GET | `/api/v1/auth/me` | 当前用户信息 |
| POST | `/api/v1/conversation/chat` | SSE 流式对话 |
| POST | `/api/v1/conversation/image-chat` | 图片对话（OCR） |
| GET | `/api/v1/conversation/sessions` | 会话列表 |
| GET | `/api/v1/conversation/session/{id}/restore` | 恢复会话 |
| DELETE | `/api/v1/conversation/{id}` | 删除会话 |
| POST | `/api/v1/knowledge/upload` | 上传知识文档 |
| GET | `/api/v1/knowledge/documents` | 文档列表 |
| POST | `/api/v1/knowledge/search` | 知识检索测试 |
| DELETE | `/api/v1/knowledge/documents/{title}` | 删除文档 |
| POST | `/api/v1/admin/seed` | 加载种子数据 |
| GET | `/api/v1/health` | 健康检查 |
| GET | `/metrics` | Prometheus 指标 |

## 评测体系

```
evaluation/benchmark/
├── fault_cases.json      # 故障诊断评测
├── sop_qa.json           # SOP 问答评测
├── alarm_cases.json      # 报警场景评测
├── safety_eval.json      # 安全防护评测
├── injection_test.json   # 注入检测评测
└── rbac_test.json        # 权限控制评测
```

运行评测：
```bash
python evaluation/runner.py
```

## LoRA 微调

```bash
# 数据清洗
python scripts/finetune/clean_data.py

# LoRA 训练
python scripts/finetune/train_lora.py --base_model qwen-2.5-14b-int4 --epochs 3
```

## 待实现（Roadmap）

- [ ] MES/ERP 系统对接
- [ ] OPC UA / Modbus 工业协议接入
- [ ] 设备实时数据采集（MQTT/Redis Streams）
- [ ] 预测性维护（时序异常检测）
- [ ] 实时告警推送（WebSocket）
- [ ] K8s 部署配置
- [ ] 工单系统联动
- [ ] 移动端 PWA 支持

## License

MIT
