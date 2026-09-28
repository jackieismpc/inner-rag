# 可观测性：追踪、日志与指标

对应 `docs/DEVELOPMENT_PLAN.md` 的 **Phase 5**。目标是：一次问答的每一步都能被**计时、归因、回放**，
线上排障从「翻日志猜」变成「看 trace 定位」。

现状：loguru 同时写 stdout 与 `LOG_DIR/app.log`，检索日志与 Prompt 统计落在
`services/retrieval_log.py`（`/api/system/stats` 可查）；没有 request_id、没有结构化字段、没有跨步骤 trace。

## 1. 配置项

### 1.1 已有（不变）

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `LOG_DIR` | `./logs` | 日志目录（`app.log`） |
| `LOG_LEVEL` | `INFO` | 全局级别 |
| `LOG_RETRIEVAL` | `true` | 是否记录检索明细 |
| `LOG_PROMPT` | `true` | 是否记录 Prompt 内容；**生产建议 `false`** |
| `APP_VERSION` | `0.3.0` | 写进 trace 与日志，便于按版本对比指标 |

### 1.2 新增（Phase 5）

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `LANGSMITH_TRACING` | `false` | 总开关。**默认关闭**，保证测试与 CI 零网络、零费用 |
| `LANGSMITH_API_KEY` | `""` | LangSmith Key；为空时即使 `TRACING=true` 也只降级告警 |
| `LANGSMITH_PROJECT` | `inner-rag` | 项目名（实验分组） |
| `LANGSMITH_ENDPOINT` | 官方默认 | 自建 / 代理部署时改这里 |
| `LANGSMITH_WORKSPACE_ID` | `""` | 多工作区时指定 |
| `LOG_FORMAT` | `text` | `text`（人读）或 `json`（采集用，一行一个 JSON） |
| `LOG_SAMPLE_RATE` | `1.0` | 成功请求的追踪采样率（错误请求 100% 记录） |
| `METRICS_BACKEND` | `none` | `none`（只提供 `/api/system/metrics`）或 `prometheus` |
| `METRICS_TOKEN` | `""` | 非空时 `/api/system/metrics` 需要 `X-Metrics-Token` |

## 2. 追踪模型

### 2.1 trace 树

一次 `POST /api/chat/send` 或 `/api/chat/stream` 产生：

```
rag.request                         # 顶层：请求级（含 request_id、会话、kb）
├── retrieve                        # 检索阶段
│   ├── cache.query                 # 查询缓存（命中则跳过后面两步）
│   ├── embed.query                 # 查询向量化
│   └── vector.search               # 向量检索（含 strategy、k、阈值、空召回）
├── prompt.build                    # Prompt 组装（分块数、总字符数、来源清单）
└── llm.generate                    # 模型调用（token 用量、首 token 延迟、结束原因）
```

文档入库链路（后台任务）单独一棵树：

```
ingest.document
├── parse                           # 文本抽取（含 OCR 后端与页数）
├── chunk                           # 分块（chunk 数、平均长度）
├── embed.documents                 # 批量嵌入（batch 数、缓存命中率）
└── vector.write                    # 写入向量库（batch 数、耗时）
```

### 2.2 metadata 与 tags 约定

所有 run 统一注入（缺值就不带该键，不要填 `None` 字符串）：

| 键 | 取值 | 用途 |
| --- | --- | --- |
| `request_id` | 透传或新生成的 UUID | 与日志串联的唯一钥匙 |
| `user_id` | 整数（字符串化） | 「谁问的」；与日志的 `user=` 对齐（Phase 3 起可用） |
| `session_id` | `conv_id` | LangSmith 的 thread，把多轮问答串成一个会话 |
| `kb_id` / `doc_id` | 整数（字符串化） | 按知识库/文档筛 trace |
| `provider` / `model` | 如 `deepseek` / `deepseek-flash` | 按后端对比延迟与成本 |
| `embedding_identity` | `provider:model` | 向量空间一致性；评测必须记录 |
| `top_k` / `rerank_top_k` / `strategy` / `score_threshold` | 检索参数 | 参数实验可对比 |
| `cache_hit` | `true` / `false` | 量化缓存收益 |
| `app_version` / `env` | 版本与 `dev|staging|prod` | 指标分版本看 |
| `chunk_count` / `context_chars` | 整数 | Prompt 膨胀诊断 |
| `prompt_tokens` / `completion_tokens` | 整数 | 成本核算 |

tags：`["env:<env>", "provider:<provider>", "kind:chat|ingest|eval"]`。

### 2.3 采样、脱敏与降级

- **采样**：`LOG_SAMPLE_RATE` 只作用于成功请求；**失败请求 100% 记录**（失败样本才是排障依据）。
- **脱敏**：trace 默认只记**元数据与统计**（长度、条数、耗时、token），不记 Prompt 与回答正文；
  需要正文时由 `LOG_PROMPT=true` 显式开启，且 trace 上打 `contains_prompt:true` 标记。
- **降级链**：`LANGSMITH_TRACING=false` → 只本地计时日志（默认行为，零网络）；
  `TRACING=true` 但 Key 缺失或 `Client` 初始化失败 → 一行 `WARNING` 说明原因后同样降级为本地；
  运行中上报失败 → 丢弃该 span 并计数 `tracing_errors_total`，**绝不抛给请求**。
- **密钥**：只从 `core/config.py` 读取；打印配置时一律掩码。

## 3. 运行日志

### 3.1 request_id 中间件

> Phase 3 已有 `IdentityContextMiddleware`（纯 ASGI）解析 Token、把 `user_id` 写进 `contextvars`，
> 日志格式已统一带 `user=`（未登录 / Token 无效时为 `-`）；权限判定本身不在此层（见 `docs/architecture.md` 3.8）。
> 下面是 Phase 5 在此基础上补的 request_id。

`RequestIdMiddleware`：

- 读 `X-Request-ID`（存在则透传，便于网关串联），否则生成 UUID4；
- 写入 `contextvars`，日志与 trace 自动带上，不需要每个函数手动传参；
- 响应头回写 `X-Request-ID`（前端展示、工单引用都用它）。

访问日志字段（每个请求一行，`text` 与 `json` 两种格式内容一致）：

```json
 {"ts":"2026-09-28T09:20:01.123Z","level":"INFO","event":"http_access","request_id":"…",
 "method":"POST","path":"/api/chat/send","status":200,"duration_ms":1284.5,
 "user":3,"kb_id":1,"client":"127.0.0.1"}
```

### 3.2 业务日志

保留现有 `[VECTOR_STORE]` / `[EMBED]` / `[FILTER]` 等前缀（人读友好），在 JSON 模式下映射为结构化字段：

```json
{"event":"retrieve","request_id":"…","kb_id":1,"strategy":"similarity",
 "k":8,"hits":5,"filtered_out":3,"top_score":0.72,"duration_ms":83.1,"cache_hit":false}
```

规则：

- 日志消息里的数值（耗时、条数、分数）在 JSON 模式下必须是**数字类型**，不要塞进字符串；
- 异常日志必须带 `request_id`、`path`、异常类型与堆栈；`ProviderError` 额外带 `provider`/`model`；
- 每条日志都要能回答「谁（user / request_id / kb）在什么阶段（event）花了多久（duration_ms）」；
- 不许打印密钥、完整 Prompt（除非 `LOG_PROMPT=true`）与用户隐私字段。

## 4. 指标

`GET /api/system/metrics`（Prometheus 文本或 JSON，取决于 `METRICS_BACKEND`）：

| 指标 | 类型 | 说明 |
| --- | --- | --- |
| `rag_requests_total{endpoint,status}` | counter | 请求数 |
| `rag_request_duration_ms{endpoint}` | histogram | 端到端延迟（p50/p95/p99） |
| `rag_retrieve_duration_ms{strategy}` | histogram | 检索耗时 |
| `rag_llm_ttfb_ms{provider}` | histogram | 首 token 延迟（流式体验的关键指标） |
| `rag_retrieve_empty_total` | counter | 空召回次数（阈值过高的信号） |
| `rag_retrieve_filtered_total` | counter | 被阈值滤掉的条数 |
| `rag_cache_hits_total{namespace}` | counter | query / embedding 缓存命中 |
| `rag_llm_tokens_total{provider,type}` | counter | prompt / completion token |
| `rag_llm_cost_usd_total{provider,model}` | counter | 估算成本（价格表进配置） |
| `rag_ingest_documents_total{status}` | counter | 入库成功/失败 |
| `tracing_errors_total` | counter | 追踪上报失败次数 |

约束：

- 指标是**进程内累计**（单 worker 语义）；多 worker 部署时标注该限制，Phase 10 视情况接 Prometheus；
- `/api/system/metrics` 也要遵守「不泄露密钥、不带正文」；
- 空召回率与 `rag_llm_ttfb_ms` 是 Phase 6 评测与 Phase 8 调优的日常观测重点。

## 5. 排障 playbook

| 症状 | 先看什么 | 常见原因 |
| --- | --- | --- |
| 回答变慢 | `rag_request_duration_ms` 分解到 `retrieve` / `llm.generate` | 检索慢 → 向量库变大；生成慢 → provider 侧拥堵或 `LLM_MAX_TOKENS` 过大 |
| 首 token 卡住 | `rag_llm_ttfb_ms` + trace 中 `llm.generate` 起始时间 | 流式未开启、provider 冷启动、`LLM_TIMEOUT` 太小导致重试 |
| 答「没有找到相关内容」 | `rag_retrieve_empty_total` + trace 的 `vector.search` | 阈值过高、embedding 截断（512 token 模型）、分块过粗 |
| 引用了错误文档 | trace 的 `sources` 列表 + `filtered_out` | `TOP_K` 过大、阈值过低、混合检索权重不合理 |
| 建库慢 / 花费高 | `ingest.document` 子 span 耗时分布 | 嵌入 batch 太小、缓存未命中、免费额度限流 |
| 配置错误 | `/api/system/health` 的 `llm` / `embedding` 字段 | 见 `docs/architecture.md` 第 6 节的错误契约 |
| trace 看不到 | 启动日志中的 tracing 状态行 | `LANGSMITH_TRACING=false`、Key 缺失、网络不通（均只告警） |

排障统一入口：**拿 `request_id` 串日志 → 拿 `session_id` 找 LangSmith thread → 在 trace 里定位最慢的 span**。

## 6. Phase 5 完成定义（DoD）

1. `/api/chat/send` 在 LangSmith 能看到完整 trace：`retrieve` 与 `llm.generate` 两个子 run，
   含耗时、token 用量、metadata；
2. 同一次请求的 stdout 日志能用 `request_id` 全部串起来，`LOG_FORMAT=json` 时每行都能 `json.loads`；
3. `LANGSMITH_TRACING=false`（默认）时**零网络调用**，且有离线测试断言；
4. 追踪上报失败不影响接口成功率（有测试：注入一个必然失败的 tracer，断言请求仍 200）；
5. `.env.example`、README「可观测性」小节、本文件三者一致。
