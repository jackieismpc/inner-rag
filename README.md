# inner-rag

> 多 Provider、可插拔、好部署的企业内部知识库问答系统 —— FastAPI + LangChain 1.x + zvec + SQLite / PostgreSQL
> （内置登录与知识库级 ACL：账号由运维脚本发放，成员按知识库授权只读 / 可写）

`inner-rag` 把企业里散落的文档（PDF / Word / Excel / 纯文本，扫描件与图片走 OCR）解析、分块、向量化
入库，再基于「向量检索 + 引用溯源 + 流式问答」回答问题。Chat 模型与 Embedding 模型是两个**互相独立的
可插拔后端**：同一条链路既能跑本地 Ollama（全离线），也能直接接 OpenRouter / DeepSeek / 任意 OpenAI
兼容网关，改两个环境变量即可切换，业务代码、接口与数据库都不用动。

![主界面](images/main.png)

## 特性

- **文档与检索**：PDF / Word / Excel / 文本类解析（图片与扫描页走可插拔 OCR）；每个知识库一个独立
  collection，统一 cosine 空间；`similarity` / `mmr` / `hybrid` 三种策略，返回**真实**相关度
  （`1 - 余弦距离`），MMR 召回项如实标注「无分数」而不是伪造 1.0
- **流式问答与会话**：SSE 逐 token 推送（先来源后答案），多轮对话取**最近** N 条历史，
  会话、消息与引用来源全部持久化
- **身份与访问控制**：本地账号 + JWT（HS256）登录，密码只存 argon2id 哈希（带随机盐），停用账号立即失效；
  知识库级 ACL 分 `read`（看库 / 提问）/ `write`（+ 增删文档）/ `owner`（+ 改设置 / 删库 / 授权成员）三级，
  列表按「我拥有或被授权」过滤
- **模型后端可插拔**：Chat 与 Embedding 各自独立选型（Ollama / OpenRouter / DeepSeek / OpenAI 兼容 /
  离线 mock），只改 `.env`；provider 名写错或漏填 Key 时得到「该去 .env 改哪个变量」的明确提示（503）。
  五个插件点（provider / 向量库 / 缓存 / 队列 / 关系库）统一走 `plugins/` 注册表，第三方包可用
  entry point 注册实现而**不改本项目源码**；`GET /api/system/plugins` 可查当前后端与全部可选项
- **性能与成本控制**：Embedding 缓存（按 `provider:model` 隔离）+ 检索缓存（LRU + TTL，按库精确失效）、
  批量嵌入 + 信号量限流、模型实例在工厂内复用、文档入库走有界并发的后台队列（带退避重试）
- **可观测与工程化**：每个请求一个 `request_id`（贯穿响应头、日志与 trace）；检索 / 问答 / 入库全链路
  span 计时，可选上报 LangSmith（默认关闭，零网络零费用）；`LOG_FORMAT=json` 一行一 JSON；
  `GET /api/system/metrics` 输出延迟分位、空召回、缓存命中与 token 用量；uv 锁依赖、Alembic 迁移、
  生产环境拒绝用默认 / 过短的 JWT 密钥启动、ruff + mypy、234 个离线 pytest 用例（另 8 个联网验收）、
  Dockerfile + docker compose

## 架构

```mermaid
flowchart LR
    FE["Vue 3 前端<br/>登录态 + SSE 流式渲染"] --> API
    subgraph API["FastAPI 后端（除登录与 /health 外均需登录）"]
        AUTHAPI["认证 API<br/>登录 / 当前用户"]
        GUARD["鉴权依赖 get_current_user<br/>+ ensure_kb_access"]
        KB["知识库 API"]
        DOC["文档 API"]
        CHAT["对话 API / SSE"]
        SYS["系统状态 API"]
    end
    GUARD --> ACL["core/access.py<br/>read / write / owner"]
    ACL --> DB
    CHAT --> RAG["RAG Service<br/>检索 + Prompt + LLM"]
    DOC --> DS["Document Service<br/>解析 → 分块 → 入库"]
    DS --> PARSE["Parser<br/>PDF/Word/Excel/Image"]
    PARSE --> OCR["OCR Backend<br/>可插拔"]
    DS --> VS["Vector Store Service"]
    RAG --> VS
    VS --> CACHE["QueryCache / EmbeddingCache"]
    VS --> ZVEC[("zvec（Alibaba 开源）")]
    VS --> EMB["Embedding Provider<br/>（providers 工厂）"]
    RAG --> LLM["LLM Provider<br/>（providers 工厂）"]
    KB --> DB[("SQLite（开发默认）<br/>PostgreSQL（部署）")]
    DOC --> DB
    CHAT --> DB
```

检索链路：查询 → 检索缓存 → 向量库（cosine 距离换算为相关度）→ 阈值过滤 → 组装 Prompt
（含最近几轮对话历史）→ LLM（流式 / 非流式）→ 落库并返回引用来源。

`providers/` 内部分层：`specs`（provider 元数据与校验，含 base_url / 模型名 / key 环境变量名）、
`chat`（Chat 模型构造 + 离线 mock 模型）、`embeddings`（向量模型构造 + 截断包装 + 离线 mock 向量）、
`factory`（按配置构造并缓存实例、探活、模型发现）。

`plugins/registry.py` 是所有插件点的名单来源：`Registry[T]` 把「有哪些实现」变成运行时可枚举的数据，
内置实现在各自模块注册，第三方包用 entry point（`inner_rag.chat_providers` / `inner_rag.embedding_providers` /
`inner_rag.vector_stores` / `inner_rag.cache_backends` / `inner_rag.task_queues`）追加。
关系库访问统一走 `repositories/`（`build_repositories(db)` 返回聚合仓储，与请求共享同一个会话）。

## 技术栈

| 层次 | 选型 |
| --- | --- |
| 语言 / 包管理 | Python 3.13（uv 管理）、`uv.lock` 锁定依赖 |
| Web 框架 | FastAPI 0.141+、Uvicorn 0.54+、SSE 流式响应 |
| LLM 编排 | LangChain 1.x（`langchain-core` 1.6+、`langchain-text-splitters`）+ `langchain-ollama`（本地）/ `langchain-openai`（OpenAI 兼容云端）/ `langchain-deepseek`（DeepSeek 官方集成） |
| 向量库 | **zvec 0.7.0**（Alibaba 开源、嵌入式、HNSW + cosine，默认后端，锁版本）；ChromaDB 1.5+ / `langchain-chroma` 保留为兼容后端（`VECTOR_STORE=chroma`）；`memory` 零依赖进程内实现（测试 / CI / 替换演练用） |
| 关系库 | SQLite（开发默认）+ PostgreSQL 16（部署可选）+ SQLAlchemy 2.1 + Alembic 1.20 |
| 认证与权限 | JWT（PyJWT，HS256）+ argon2id 口令哈希（argon2-cffi）+ 知识库级 ACL（owner / member） |
| 文档解析 | pypdf、PyMuPDF、python-docx、docx2txt、openpyxl、xlrd、Pillow、chardet |
| 前端 | Vue 3 + Vite + Pinia + Tailwind CSS 3 |
| 质量 | ruff、pytest（+ pytest-asyncio）、mypy |

> **向量库选型**：本项目选用 **zvec**（[Alibaba 开源](https://github.com/alibaba/zvec)的嵌入式向量库，
> Apache-2.0，定位「向量库里的 SQLite」：进程内嵌入、无需独立服务、HNSW + cosine、WAL 持久化），
> 理由是「零运维」，与 SQLite 单文件开发模型一致。Phase 4 起 zvec 已是默认后端（`VECTOR_STORE=zvec`），
> ChromaDB 保留为兼容实现；两者受同一个 `VectorStore` 契约约束并跑同一套契约测试（见 `docs/architecture.md` 3.2）。
> Phase 7 又加了 `memory`（零依赖进程内实现，`VECTOR_STORE=memory`）：它是「换后端不改业务代码」的实证——
> 新增它只加了一个实现类 + 一次注册，同一套契约用例直接全绿；代价是数据只在内存、重启即丢，仅供测试与演练。
> 注意内嵌 zvec 按 collection 目录独占写锁，**必须单进程部署**（不要 `uvicorn --workers`）。

## 快速开始

### 0. 前置条件

- Linux / macOS（Windows 建议 WSL2）+ [uv](https://docs.astral.sh/uv/)（会自行安装 Python 3.13）
- 模型三选一：云端 API（需 OpenRouter / DeepSeek 的 key，推荐）/ 本地 [Ollama](https://ollama.com/)
  （全离线）/ `mock`（零依赖，仅演示与验收链路）
- 数据库：开发默认 SQLite 单文件，**无需任何安装**；只有部署阶段才需要 Docker 或 PostgreSQL

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 1. 安装依赖

```bash
git clone https://github.com/jackieismpc/inner-rag.git
cd inner-rag
uv sync                 # 创建 .venv 并按 uv.lock 安装依赖（含 Python 3.13）
```

需要本地 OCR（PaddleOCR）时追加：

```bash
uv sync --extra ocr-paddle
```

### 2. 配置环境变量与密钥

```bash
cp .env.example .env      # .env 已被 .gitignore 忽略，不会进仓库
# 需要改的通常只有：模型 provider 与 key、PORT、CORS_ORIGINS
```

数据库默认就是 SQLite（`sqlite:///./data/inner_rag.db`，单文件、零配置），**开发阶段不需要任何
数据库服务**。密钥统一写在 `.env` 里（`OPENROUTER_API_KEY` / `DEEPSEEK_API_KEY` / `OPENAI_API_KEY`），
代码只从 `.env` 与运行环境变量读取：不要把 key 写进源码，也不要提交 `.env`。

> 部署阶段换 PostgreSQL 时，把 `DATABASE_URL` 改成 `postgresql+psycopg://rag:rag@localhost:5432/rag_db`
> （可 `docker compose up -d postgres` 一键起库），再执行一次 `uv run alembic upgrade head`，代码无需改动。

### 3. 建表并启动后端

表结构由 Alembic 管理（`AUTO_CREATE_TABLES=false`），首次启动前必须先迁移（SQLite 也会自动建出
`data/inner_rag.db`）：

```bash
uv run alembic upgrade head
uv run uvicorn inner_rag.main:app --reload --port 8010
# 或者一步到位（自动 uv sync + 迁移 + 热重载）：
./scripts/start.sh
```

打开 <http://localhost:8010/docs> 查看交互式 API 文档，`/api/system/health` 检查后端与模型连通性：

```bash
curl -s http://localhost:8010/api/system/health
# {"status":"healthy","version":"0.3.0",
#  "llm":{"provider":"deepseek","model":"deepseek-flash","ok":true,"model_available":true,"error":null,"warning":null},
#  "embedding":{"provider":"openrouter","model":"liquid/lfm-2.5-embedding-350m:free","ok":true,...}}
```

> `status: degraded` 时看 `llm.error` / `embedding.error`：文案会直接指出该去 `.env` 改哪个变量
> （例如漏填 `OPENROUTER_API_KEY`）或哪个服务连不上。检索与问答会因此失败，但知识库、文档等
> 管理接口仍可用。
>
> `llm.warning` 是「能用但值得注意」的提示，不影响 `status`：例如某家 provider 的 `/models`
> 不完整（DeepSeek 只列主推模型，旧别名仍可调用），此时不会误报为不可用。

### 4. 创建第一个账号

系统**不提供注册接口**：企业内部账号由管理员发放。用运维脚本建号（密码交互式输入、不回显）：

```bash
uv run scripts/create_user.py admin          # 建号；密码交互式输入（≥8 位仅告警不阻断）
uv run scripts/create_user.py --list         # 看现有账号（含启用状态）
```

所有业务接口都要求登录（`Authorization: Bearer <token>`），只有 `POST /api/auth/login` 与
`GET /api/system/health` 免鉴权。前端访问 <http://localhost:3000> 时会直接跳到登录页。

### 5. 选择模型 Provider

Chat 与 Embedding 是**两个独立开关**，改 `.env` 即可，代码无需改动：

| Provider | 可用于 | 说明 |
| --- | --- | --- |
| `ollama` | chat + embedding | `http://localhost:11434`，本地、无需 key |
| `openrouter` | chat + embedding | `https://openrouter.ai/api/v1`，OpenAI 兼容聚合网关，一个 key 用数百个模型 |
| `deepseek` | **仅 chat** | `https://api.deepseek.com/v1`，官方集成（`langchain-deepseek`）；模型名以[官方文档](https://api-docs.deepseek.com)为准（默认 `deepseek-flash`，另有 `deepseek-v4-pro`）；官方**没有 embeddings 接口**，写成 `EMBEDDING_PROVIDER=deepseek` 会得到明确报错 |
| `openai` | chat + embedding | `https://api.openai.com/v1`；也可指向任何 OpenAI 兼容的自建网关 |
| `mock` | chat + embedding | 不联网、不需要 key、输出确定性；用于本地演示 / CI / 降级验收（只有词面相似度，不能用来评估检索效果） |

`GET /api/system/providers` 会列出全部可选项、当前选择与 key 是否已配置。

云端 API 路线（推荐，最快）。chat 与 embedding 分属不同厂商也完全可以：

```bash
# .env —— Chat 走 DeepSeek，Embedding 走 OpenRouter
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY={{DEEPSEEK_API_KEY}}
DEEPSEEK_CHAT_MODEL=deepseek-flash     # 也可用 deepseek-v4-pro
# LLM_REASONING_EFFORT=high            # 可选：minimal/low/medium/high（仅 DeepSeek 生效）

EMBEDDING_PROVIDER=openrouter
OPENROUTER_API_KEY={{OPENROUTER_API_KEY}}
EMBEDDING_MAX_INPUT_CHARS=400    # 见下方「小上下文模型」说明
```

> 免费的 `liquid/lfm-2.5-embedding-350m:free`（1024 维）只有 **512 token** 上下文，建议同时设
> `EMBEDDING_MAX_INPUT_CHARS=400`；免费路由的数据可能被上游留存，有合规要求时换付费模型。

本地 Ollama 路线：

```bash
ollama pull qwen3:14b            # 对话模型
ollama pull qwen3-embedding:8b   # 向量模型（必须与建库时的 embedding 保持一致）
```

完全离线路线：`LLM_PROVIDER=mock` + `EMBEDDING_PROVIDER=mock`，不需要任何模型服务，也能跑通
「上传 → 检索 → 带引用回答」的完整链路。

### 6. 启动前端

```bash
cd frontend
npm install
npm run dev        # http://localhost:3000，通过 Vite 代理访问后端 8010
```

打开后先用第 4 步创建的账号登录；未登录的请求会被后端拒绝（401），前端会自动回到登录页。

## 使用指南

先用账号换一个 token（12 小时有效），后续请求都带上它：

```bash
TOKEN=$(curl -s -X POST localhost:8010/api/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"admin"}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"]["access_token"])')
```

1. **建知识库**：前端「新建知识库」，或
   `curl -X POST localhost:8010/api/kb -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' -d '{"name":"产品手册"}'`
   （建库者即 `owner`；知识库列表只返回自己拥有或被授权的库）
2. **上传文档**：前端拖拽上传，或

   ```bash
   curl -X POST localhost:8010/api/doc/upload \
     -H "Authorization: Bearer $TOKEN" \
     -F kb_id=1 -F files=@./手册.pdf
   ```

   上传接口立即返回 `doc_ids`，解析与向量化在后台任务里进行；用
   `GET /api/doc?kb_id=1` 查看 `status`（`pending/processing/completed/failed`）与 `error_msg`。
3. **提问**：前端对话页会以 SSE 流式渲染答案与引用来源；等价的命令行调用：

   ```bash
   curl -N -X POST localhost:8010/api/chat/stream \
     -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
     -d '{"kb_id":1,"question":"这款产品的保修政策是什么？"}'
   ```

   SSE 事件顺序：`conv_id`（会话 ID）→ `sources`（引用来源，含相关度）→ `token`（增量文本，多条）
   → `done`（完整答案）。每个事件形如 `data: {"type":"sources","data":[...]}`。
4. **重新处理/重建索引**：单文档失败可 `POST /api/doc/{doc_id}/reprocess`；换了 embedding 模型则
   必须重建索引（见下方脚本），否则查询向量与库内向量不在同一空间。
5. **成员权限**：知识库详情页的「成员管理」（仅 `owner` 可见）可按用户名授权：
   `read` 能看文档、问问题，`write` 还能上传 / 删除文档，改库设置与删库仅 `owner`。权限每次请求
   实时判定，改权限或移除成员后立即生效，无需重新登录。

## 配置说明

全部配置项见 `.env.example`（按 应用 / 认证 / 服务端 / 数据库 / 密钥 / 模型 / 向量库 / 上传 / OCR / 检索 /
缓存 / CORS / 日志 分组）；本地实际生效的值写在 `.env`（已被 gitignore，不入库）。
几个容易踩坑的关键项：

| 配置 | 默认值 | 说明 |
| --- | --- | --- |
| `PORT` | `8010` | 后端端口（`8000` 在共享机器上常被占用），需与 `frontend/vite.config.js` 代理一致 |
| `DATABASE_URL` | `sqlite:///./data/inner_rag.db` | 开发默认 SQLite；部署切 `postgresql+psycopg://...` 后重跑 `alembic upgrade head` |
| `LLM_PROVIDER` / `EMBEDDING_PROVIDER` | `ollama` | 两个**独立**开关；向量侧没有 `deepseek`（官方无 embeddings 接口） |
| `OPENROUTER_API_KEY` / `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` | 空 | 云端 API 密钥，只写在 `.env`，不要提交 |
| `EMBEDDING_MAX_INPUT_CHARS` | `0` | 单条输入的字符上限（0 = 不截断）；小上下文模型建议设 `400` |
| `VECTOR_STORE` | `zvec` | 向量库后端：`zvec`（嵌入式，默认）/ `chroma`（兼容旧数据）/ `memory`（零依赖进程内，重启即丢）；切换后用 `scripts/reindex_kb.py <kb_id>` 重建 |
| `ZVEC_PATH` | `./data/zvec_db` | zvec 数据目录，每知识库一个 `kb_<id>/` collection；写锁目录独占，**单进程部署** |
| `CACHE_BACKEND` | `memory` | 缓存后端：`memory`（每 namespace 一份有界 LRU）；换 Redis 只需加实现 + 注册 |
| `QUERY_CACHE_MAX_SIZE` / `QUERY_CACHE_TTL` | `500` / `300` | 检索缓存容量与有效期（秒，0 = 不过期）；入库后仍会按 `kb_id` 精确失效 |
| `EMBEDDING_CACHE_MAX_SIZE` | `2000` | 嵌入缓存容量；按 `provider:model` 隔离，跨模型不会命中 |
| `TASK_QUEUE_BACKEND` | `inprocess` | 后台任务队列：`inprocess`（协程池）/ `inline`（同步执行，测试用） |
| `TASK_QUEUE_CONCURRENCY` | `2` | 后台并发上限；免费 embedding 路由限流严，别调大 |
| `TASK_QUEUE_MAX_RETRIES` / `TASK_QUEUE_RETRY_BACKOFF` | `1` / `5.0` | 失败重试次数与退避系数（`backoff × attempt` 秒） |
| `TASK_QUEUE_HISTORY` | `200` | 任务历史环形缓冲大小（`/api/system/stats` 展示） |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | 分块参数，影响召回粒度；改动后建议重建索引并跑基准 |
| `RETRIEVAL_SCORE_THRESHOLD` | `0.3` | 相关度阈值（`1 - 余弦距离`）；过高会导致空召回 |
| `AUTO_CREATE_TABLES` | `false` | 表结构交给 Alembic；仅测试/一次性库设为 `true` |
| `AUTH_SECRET_KEY` | `dev-only-insecure-...` | JWT 签名密钥（HS256）；`DEBUG=false` 时用默认值或短于 32 字节会**拒绝启动** |
| `AUTH_TOKEN_TTL_MINUTES` | `720` | 令牌有效期（分钟）；JWT 无状态、无法单独撤销，短 TTL 是泄漏后的唯一收敛手段 |
| `ENABLE_DOCS` | `true` | 是否开放 `/docs` `/redoc` `/openapi.json`；生产环境建议关闭 |
| `OCR_BACKEND` | `none` | `none` 或 `paddle`；关闭时图片/扫描件会明确报错而不是写入占位文本 |
| `ALLOW_LOCAL_IMPORT` | `false` | 是否允许 `import-path`（服务端文件系统读取能力），配合 `LOCAL_IMPORT_ROOT` 限定目录 |
| `CORS_ORIGINS` | `http://localhost:3000,...` | 前端来源白名单 |
| `LOG_FORMAT` | `text` | `text`（人读）/ `json`（一行一个 JSON，供采集端解析） |
| `LANGSMITH_TRACING` | `false` | LangSmith 追踪总开关，**默认关闭**；关闭时 span 照常计时并落本地日志 |
| `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT` | 空 / `inner-rag` | 追踪上报的目标；Key 为空时只降级告警，不影响请求 |
| `METRICS_BACKEND` / `METRICS_TOKEN` | `none` / 空 | `/api/system/metrics` 的输出格式与可选门禁（非空时需 `X-Metrics-Token`） |

其余项（各 provider 模型名、top_k、缓存大小、日志开关等）都在 `.env.example` 里有逐项注释；
其中「换 embedding 模型 = 换向量空间」需要重建索引。

## 可观测性（追踪 / 日志 / 指标）

排障统一入口：**拿 `request_id` 串日志 → 拿 `session_id`（即会话 ID）找 LangSmith thread →
在 trace 里定位最慢的 span**。

- **request_id**：每个请求生成或透传 `X-Request-ID`，同时写进响应头、每条日志与每个 span；
  文本日志里是 `rid=`，`LOG_FORMAT=json` 时每行一个 JSON 对象。
- **span 树**：一次问答是
  `rag.request → retrieve → (cache.query | vector.search → embed.query) → prompt.build → llm.generate`；
  一次文档入库是 `ingest.document → parse / vector.ingest`（`embed.documents` 嵌在 `vector.ingest` 里）。
- **追踪后端**：`LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY` 后上报到 LangSmith（项目 `LANGSMITH_PROJECT`）。
  默认关闭；Key 缺失、初始化失败或上报失败都只降级为本地计时日志并计入 `tracing_errors_total`，
  **绝不影响请求成功率**。
- **指标**：`GET /api/system/metrics`（JSON 或 Prometheus 文本）覆盖请求数与延迟分位、检索耗时、
  空召回与阈值过滤数、缓存命中、token 用量、首 token 延迟、入库成功 / 失败数。

开启追踪并自检（**密钥只走环境变量，不要写进 `.env`**）：

```bash
export LANGSMITH_API_KEY=...        # 环境变量优先级高于 .env，无需改代码
export LANGSMITH_TRACING=true
uv run scripts/check_langsmith.py --dataset   # 建 trace → 服务端读回 → 同步 dataset → 回写 feedback
uv run pytest -m live -q                      # 真实联网用例（默认 deselect）
```

`check_langsmith.py` 的判定是「**读回来**」而不是「没报错」——上报走后台队列，
`post()` 成功只代表进了队列，读不到就是没通。

两个必须知道的边界：

1. 指标是**进程内累计**（单 worker 语义）：多副本部署时每个副本各记一份，看板要按实例聚合；
2. **成本指标（`rag_llm_cost_usd_total`）尚未提供**——token 单价属于计费域，等 Phase 10 与成本看板
   一起定，不在代码里写没有来源的价格表。

细节见 `docs/observability.md`。

## 目录结构

```
inner-rag/
├── pyproject.toml            # uv 项目定义、依赖、ruff/pytest 配置
├── uv.lock                   # 依赖锁定
├── .python-version           # 3.13
├── alembic.ini
├── migrations/               # Alembic 迁移（env.py + versions/）
├── src/inner_rag/
│   ├── main.py               # FastAPI 应用入口（CORS、异常处理、路由注册）
│   ├── core/                 # 配置、数据库引擎与会话、安全原语（argon2id / JWT）、ACL 判定、身份与请求上下文、日志格式（text/json）、追踪门面、指标注册表
│   ├── models/               # SQLAlchemy 2.0 ORM 模型（含 user / kb_member）
│   ├── schemas/              # Pydantic 请求/响应模型
│   ├── api/                  # 路由：auth / kb / document / chat / system + 鉴权依赖（deps.py）
│   ├── repositories/         # 关系库访问的唯一入口：base 契约（Protocol）+ sqlalchemy 实现 + build_repositories
│   ├── plugins/              # 插件注册表：Registry[T] + chat/embedding/向量库/缓存/队列五个插件点 + plugin_status()
│   ├── providers/            # 模型后端抽象：specs / chat / embeddings / factory（多 provider）
│   └── services/             # parser、ocr、embedding、vector_store/（base + zvec/chroma/memory 适配 + 注册）、rag、cache、task_queue、retrieval_log
├── scripts/                  # 运维与排查脚本（含建号 create_user.py）+ start.sh
├── tests/                    # 离线 pytest 用例 + 可选的真实 API 联网验收（-m live）
├── benchmark/                # 基准脚本：指标、评测集运行、结果落盘、README 基准表维护
├── docs/                     # 开发计划、架构契约、评测与测试策略、龙族评测集、阶段记录
├── frontend/                 # Vue 3 + Vite 前端
├── docker-compose.yml        # 部署用：PostgreSQL 16（默认）/ 后端容器（profile=app）
├── Dockerfile
├── .env.example              # 配置模板（真实的 .env 不入库）
└── images/                   # 截图
```

## API 一览

除 `POST /api/auth/login`、`GET /api/system/health` 外，**所有接口都要求**
`Authorization: Bearer <token>`；未登录返回 401（带 `WWW-Authenticate: Bearer`），
已登录但无权访问他人知识库返回 403。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/auth/login` | 用户名 + 密码换 JWT（失败统一返回 401 且文案一致，不泄露账号是否存在） |
| GET | `/api/auth/me` | 当前登录用户（刷新页面后恢复登录态用；无 logout 接口，客户端丢弃 token 即可） |
| GET | `/api/kb` | 知识库列表（只含我拥有或被授权的；分页、关键词） |
| POST | `/api/kb` | 新建知识库（自动写入当前 embedding 标识） |
| GET | `/api/kb/{kb_id}` | 详情（含 collection 向量数） |
| PUT / DELETE | `/api/kb/{kb_id}` | 更新 / 删除（仅 `owner`；连带删除向量与上传文件） |
| GET | `/api/kb/{kb_id}/members` | 成员列表（仅 `owner`） |
| POST | `/api/kb/{kb_id}/members` | 按用户名授权 / 改权限（仅 `owner`；重复授权即改权限） |
| DELETE | `/api/kb/{kb_id}/members/{user_id}` | 移除成员授权（仅 `owner`，立即生效） |
| GET | `/api/doc` | 文档列表（分页、`status`、`keyword`） |
| POST | `/api/doc/upload` | 多文件上传（流式落盘 + 大小限制 + 后台解析入库） |
| POST | `/api/doc/import-path` | 从服务端路径导入（默认关闭，见 `ALLOW_LOCAL_IMPORT`） |
| GET / DELETE | `/api/doc/{doc_id}` | 文档详情 / 删除（同时清理向量与检索缓存） |
| POST | `/api/doc/{doc_id}/reprocess` | 重新处理文档 |
| POST | `/api/chat/stream` | SSE 流式问答（`conv_id` / `sources` / `token` / `done` / `error`） |
| POST | `/api/chat/send` | 非流式问答 |
| GET | `/api/chat/conversations` | 会话列表 |
| GET | `/api/chat/conversations/{id}/messages` | 会话消息（含引用来源） |
| DELETE | `/api/chat/conversations/{id}` | 删除会话 |
| GET | `/api/system/health` | 健康检查（后端 + Chat/Embedding provider 连通性与错误原因 + 各插件点当前后端） |
| GET | `/api/system/providers` | 全部可用 provider、当前选择、key 是否已配置（不返回密钥） |
| GET | `/api/system/plugins` | 五个插件点（chat / embedding / 向量库 / 缓存 / 队列）的当前实现、可选实现与第三方实现 |
| GET | `/api/system/stats` | 检索统计 + 缓存状态 + 后台队列状态 |
| GET | `/api/system/metrics` | 进程内指标（延迟分位、检索 / 缓存 / token / 入库）+ 追踪状态；**免登录**，配 `METRICS_TOKEN` 时需 `X-Metrics-Token` |
| GET | `/api/system/config` | 前端可用的非敏感运行时配置 |
| GET | `/api/system/models` | 当前 provider 的可用模型列表（Ollama / 云端 `/models`） |
| POST | `/api/system/cache/clear?kb_id=` | 手动清理缓存（指定知识库或全清） |

## 运维与排查脚本

```bash
uv run scripts/create_user.py <username>          # 建号（密码交互式输入、不回显）
uv run scripts/create_user.py <username> --display-name "张三"         # 建号时带显示名
uv run scripts/create_user.py <username> --reset-password             # 重置密码（忘记密码时用）
uv run scripts/create_user.py <username> --disable / --enable         # 停用 / 启用账号
uv run scripts/create_user.py --list                                  # 列出账号
uv run scripts/reindex_kb.py <kb_id>              # 重建整个知识库索引（换 embedding 模型后必做）
uv run scripts/reindex_kb.py <kb_id> --keep-vectors  # 保留向量，仅重新解析文档
uv run scripts/reindex_doc.py <doc_id>            # 重建单个文档索引
uv run scripts/check_vectors.py [kb_id]           # 向量体检：向量数、各文件分块数、与数据库是否一致
uv run scripts/query_probe.py <kb_id> "查询词" --strategy hybrid --k 8
                                                  # 检索探针：看命中哪些分块、相关度多少、被过滤多少
```

`query_probe.py` 是排查「答非所问」的第一手段：能直接区分「没召回」「被阈值过滤」和
「Prompt 组装问题」三种情况。

## 开发

```bash
uv run ruff check .            # 静态检查
uv run ruff format .           # 代码格式化
uv run pytest                  # 全量测试（离线；live 用例默认 deselect）
uv run pytest -m live -q       # 真实 provider 联网验收（需 OPENROUTER_API_KEY / DEEPSEEK_API_KEY，会产生少量费用）
uv run pytest -q tests/test_auth.py              # 只看登录 / 鉴权 / ACL 用例
uv run pytest -q tests/test_api.py::test_chat_stream_events_and_persistence
uv run mypy                    # 类型检查（配置见 pyproject.toml 的 [tool.mypy]）
uv run python -m benchmark.run_bench --mode fixtures   # 基准脚本离线自检（评测集校验 + 指标算法）
./scripts/gates.sh g0          # 门禁 G0（每次提交）：ruff + 评测离线自检
./scripts/gates.sh g1          # 门禁 G1（push 前）：G0 + mypy + 离线全量 + 迁移自检 + 冒烟 + changelog / 密钥检查
```

测试说明：`tests/conftest.py` 在导入应用之前就把环境切到临时 SQLite、临时目录与固定的 `mock`
provider，并用确定性的假 embedding / 假 LLM 替换真实 Provider，因此默认测试**完全离线**、可复现，
也不会产生模型调用费用，且不会受开发者本机 `.env` 的影响。`pytest -m live` 才会真实调用云端 API
（无 key 时自动 skip）。除 `client` 外还提供 `other_client`（另一个用户），用来验证跨库 403 隔离；
账号在测试里直接建（与 `create_user.py` 同一条路径），因为系统没有注册接口。

数据库迁移：

```bash
uv run alembic revision --autogenerate -m "add xxx"   # 修改 ORM 模型后生成迁移
uv run alembic upgrade head                          # 应用
uv run alembic check                                 # 校验模型与迁移是否漂移
uv run alembic downgrade -1                           # 回退一步
```

## 基准测试（Benchmark）

改检索、改分块、改 Prompt 之后必须能回答一个问题：**到底变好了没有**。`benchmark/` 下的脚本把评测集
（`docs/datasets/dragon_king/eval_v1.jsonl`，9 题含 1 条负样本）跑一遍，输出 RAG 指标与延迟，
并把结果写成下面这段表格的一行；指标定义、门禁阈值与判读方式见 `docs/evaluation.md`。

```bash
# 离线自检：mock provider + 短片段 fixture，不联网、不花钱（提交前跑这个）
uv run python -m benchmark.run_bench --mode fixtures

# 真实知识库：检索指标 + 延迟，写入 README 基准表
uv run python -m benchmark.run_bench --mode kb --kb-id 3 --update-readme

# 再加回答指标（要点命中率 / 引用精度 / 拒答正确率，会调用 LLM 产生费用）
uv run python -m benchmark.run_bench --mode kb --kb-id 3 --answer --update-readme

# 建评测库（从 data/uploads/龙族.pdf 按页窗口构建，产出 manifest）
uv run scripts/build_eval_kb.py --profile small --name dragon_king_small --owner admin

# 回答侧评测：judge 正确性 / 忠实度 / token / 失败归因 → docs/reports/eval-<日期>-<label>.md
uv run scripts/eval_answer.py --from-result benchmark/results/<上面的结果 json>
```

- `fixtures` 模式只验证脚本与指标算法（mock embedding 没有语义能力），**不写表也不落盘**，避免把自检数字当成成绩；
- `kb` 模式用真实知识库并绕过 QueryCache 直接查向量库，召回与延迟都是真值，`--answer` 按题计费。

<!-- BEGIN BENCHMARK -->
| 日期 | 配置 | 题数 | Recall@k | MRR | 页命中率 | 要点命中率 | 引用精度 | 拒答正确率 | 检索 p50 | 检索 p95 | 端到端 p50 | 结果文件 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-09-30 | kb1/openrouter:liquid/lfm-2.5-embedding-350m:free/hybrid/k=8+answer | 9 | 75.0% | 0.688 | 62.5% | 75.0% | 27.5% | 100.0% | 1238.6 ms | 2176.1 ms | 1442.6 ms | `benchmark/results/2026-09-30-kb-kb1-openrouter-liquid-lfm-2-5-embedding-350m-free-hybrid-k-8-answer.json` |
<!-- END BENCHMARK -->

表格由 `--update-readme` 写入，**不要手工编辑标记之间的区域**；kb 模式每次还在 `benchmark/results/`
留一份含逐题明细的 JSON，便于回溯。指标口径与注意事项见 `benchmark/README.md`。

回答侧的 judge 正确性 / 忠实度 / 失败归因在 `docs/reports/eval-*.md`，建库的页窗口与配置快照在
`docs/reports/eval-kb-*.json`。**引用里的页码是源 PDF 的物理页号**（不是子 PDF 的局部页号），
建库脚本不做任何重编号——这一点踩过坑：重编号会让引用翻不到原文、评测锚点全部对不上。

## 部署

开发阶段用 SQLite + 云端 API 就能跑通全链路（见 `docs/DEVELOPMENT_PLAN.md` 的 Phase 10）。

只跑数据库（后端仍跑在宿主机上，改代码无需重建镜像）：

```bash
docker compose up -d postgres
```

连后端一起跑（镜像内自动 `alembic upgrade head`）：

```bash
docker compose --profile app up -d --build
docker compose logs -f api
```

后端容器对外暴露 `8010`，以非 root 用户（uid 10001）运行；数据（上传文件、zvec 向量库、日志）
落在 `app_data` 卷的 `/data` 下（若改为绑定宿主机目录，注意该目录需允许 uid 10001 写入）。
镜像里的 `OLLAMA_BASE_URL` 默认指向 `host.docker.internal:11434`（Compose 已加 `extra_hosts`），
如需指向云端 API，直接在 `.env` 或 Compose 环境变量里覆盖。

> **单进程部署**：zvec 内嵌模式的写锁按 collection 目录独占、跨进程互斥，因此**不要**用
> `uvicorn --workers N`，也不要让多个副本共享同一份 `/data`（第二个进程会打不开向量库）。需要横向扩容
> 时应改为远程向量服务（Phase 10 视情况）。从 Chroma 切到 zvec、改分块参数或换 embedding 模型后，
> 用 `uv run scripts/reindex_kb.py <kb_id>` 重建索引，不做原地格式转换。

镜像构建状态：Dockerfile 的依赖安装与启动链路已在本机用 podman（无 sudo）实测通过——
按 `uv.lock` 冻结安装 129 个包，容器内 `alembic upgrade head` 与 `/api/system/health`（200）均正常。
本机缺少 rootless 必需的 `newuidmap`（setuid root，需管理员安装），因此 apt 沙箱降权、
`useradd --uid 10001 app`、`chown -R app:app` 这三行与非 root 运行只能在真 Docker 或 Phase 10 的 CI 中验证；
详见 `CHANGELOG.md` 对应条目与 `docs/DEVELOPMENT_PLAN.md` 风险登记簿。

前端生产构建：

```bash
cd frontend && npm run build          # 产物在 frontend/dist，可用任意静态服务器托管
```

## 常见问题

**Q：`/api/system/health` 返回 `degraded`？**
A：看响应里的 `llm.error` / `embedding.error`，它能直接定位原因：漏填 API Key（会指名该写哪个变量）、
provider 名写错（会列出可选值）、服务连不上（会带上 URL）或模型未拉取。

**Q：上传成功但文档 `status=failed`？**
A：看 `error_msg`。常见原因：扫描件/图片未启用 OCR（`OCR_BACKEND=none`）、密码保护的 PDF、
`.doc` 老格式（建议先转 `.docx`）、模型服务不可用。

**Q：提问总是「未找到相关信息」？**
A：先跑 `uv run scripts/query_probe.py <kb_id> "问题"`。若命中了分块但相关度低于
`RETRIEVAL_SCORE_THRESHOLD`，说明阈值偏高；若完全没命中，检查文档是否 `completed` 且向量数 > 0，
以及当前 embedding 模型是否与建库时一致。

**Q：换了 embedding 模型后检索结果全乱？**
A：向量空间变了，旧向量全部失效。执行 `uv run scripts/reindex_kb.py <kb_id>` 重建索引。

**Q：接口返回 401？**
A：未登录或 token 已过期（默认 12 小时），重新登录即可；前端遇到 401 会自动回登录页。
若是脚本调用，检查是否带了 `Authorization: Bearer <token>`（SSE 流式接口也一样）。

**Q：接口返回 403「无权访问该知识库」？**
A：登录没问题，但当前账号对该知识库没有权限。知识库只对「拥有者 + 被授权成员」可见：
请拥有者在知识库详情页的「成员管理」里授权（`read` / `write`）。被移除授权或降低权限后立即生效。

**Q：忘记密码 / 要新增账号？**
A：没有注册与找回入口（企业内部账号由管理员发放）：`uv run scripts/create_user.py <username>`
建号，`--reset-password` 重置，`--disable` 停用（停用后已签发的 token 立即失效）。

## 路线图

- **已完成**：多 Provider 抽象层（Chat / Embedding 独立选型、OpenRouter / DeepSeek 官方集成、`mock`
  降级路径、`/api/system/providers`）、SQLite 优先与密钥外置、Alembic 迁移、Docker 资产；
  身份与访问控制（Phase 3）——本地账号 + JWT 登录、argon2id 口令哈希、知识库级 ACL（owner / 只读 /
  可写）、前端登录页与按权限渲染、`scripts/create_user.py` 建号；
  向量库统一到 zvec（Phase 4）——`VectorStore` 契约 + zvec 默认后端 + Chroma 兼容实现，
  同一套契约测试参数化跑两个后端；
  可观测性（Phase 5）——`request_id` 贯穿、日志 text/json、指标注册表、LangSmith 追踪（默认关闭）；
  评测体系（Phase 6）——龙族真实语料评测库 + `benchmark/` 检索指标 + `scripts/eval_answer.py`
  （judge 正确性 / 忠实度 / 成本）与基线数字落档；
  可插拔深化（Phase 7）——五个插件点统一走 `plugins/` 注册表（含第三方 entry point）、
  关系库收敛到 `repositories/`、缓存与后台队列抽象化、`memory` 向量库作为替换演练实证、
  `GET /api/system/plugins` 暴露插件状态
- **进行中**：Phase 8——用评测集驱动检索与回答质量提升（分块策略 / rerank / 查询改写 / 阈值定标）
- **Phase 9–10**：OCR / VLM 文档面扩展 → 交付（Docker / PostgreSQL / CI）

每个阶段的交付物、完成定义、测试门禁（G0–G3）与里程碑见 `docs/DEVELOPMENT_PLAN.md`。

## License

MIT
