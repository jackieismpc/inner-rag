# inner-rag

> 多 Provider、可插拔、好部署的企业内部知识库问答系统 —— FastAPI + LangChain 1.x + ChromaDB + SQLite / PostgreSQL

`inner-rag` 把企业里散落的文档（PDF / Word / Excel / 纯文本，扫描件与图片走 OCR）解析、分块、向量化
入库，再基于「向量检索 + 引用溯源 + 流式问答」回答问题。Chat 模型与 Embedding 模型是两个**互相独立的
可插拔后端**：同一条链路既能跑本地 Ollama（全离线），也能直接接 OpenRouter / DeepSeek / 任意 OpenAI
兼容网关，改两个环境变量即可切换，业务代码、接口与数据库都不用动。

![主界面](images/main.png)

## 设计目标

1. **多 Provider**：Chat 覆盖 `ollama` / `openrouter` / `deepseek` / `openai`（兼容自建网关）/
   `mock`，Embedding 覆盖 `ollama` / `openrouter` / `openai` / `mock`；两者可自由组合，例如
   「Chat 走 DeepSeek + Embedding 走 OpenRouter」，或「Chat 走云端 + Embedding 走本地」。
2. **可插拔**：所有模型后端实现收敛在 `src/inner_rag/providers/` 一层，服务层、API 层、缓存层只依赖
   统一接口；新增一个 provider 只需补一条 provider 元数据与一个构造分支。
3. **好部署**：`uv sync` + 一次 `alembic upgrade head` 就能跑起来；开发默认 SQLite 单文件、零外部依赖，
   部署时只改 `DATABASE_URL` 即可切到 PostgreSQL，另附 Dockerfile 与 docker compose。
4. **工程化可用**：Alembic 迁移、CORS 白名单、文件名与导入路径校验、密钥只从 `.env` 读取、
   检索质量可观测（真实相关度、缓存命中率、空召回率、平均延迟）、68 个离线用例 + 可选的真实 API 联网验收。

## 特性

- **多格式文档解析**：PDF（逐页）、Word（docx / doc）、Excel（xlsx / xls）、文本类
  （txt / md / csv / json / xml / html）；图片与扫描页走可插拔 OCR 后端
- **向量化与检索**：ChromaDB 持久化，每个知识库一个独立 collection，统一 cosine 空间
- **多策略检索**：`similarity`（余弦相似度）/ `mmr`（多样性去重）/ `hybrid`（两者融合），
  返回**真实**相关性分数（1 - 余弦距离），MMR 召回项如实标注为「无分数」而不是伪造 1.0
- **流式问答**：SSE 逐 token 推送，先推引用来源再推答案，前端实时渲染并展示相关度
- **会话管理**：多轮对话（取**最近** N 条历史）、会话列表、消息与引用来源持久化
- **模型后端可插拔**：Chat 与 Embedding 各自独立选型（Ollama / OpenRouter / DeepSeek /
  OpenAI 兼容 / 离线 mock），只改 `.env` 里的两个变量；provider 名称写错或漏填 Key 时，得到的是
  「该去 .env 改哪个变量」的明确提示（503），而不是一个 500 或看不懂的 401
- **性能与成本控制**：Embedding 缓存（按 `provider:model` 身份隔离）+ 检索结果缓存（LRU + TTL，
  按知识库精确失效）、批量嵌入 + 信号量限流、模型实例在工厂内复用
- **可观测性**：检索日志、Prompt 日志、缓存命中率 / 空召回率 / 平均延迟统计
- **工程化**：uv 锁依赖、Alembic 迁移、CORS 白名单、文件名与导入路径安全校验、
  ruff + mypy 检查、68 个离线 pytest 用例（另有可选的真实 API 联网验收）、Dockerfile + docker compose

## 架构

```mermaid
flowchart LR
    FE["Vue 3 前端<br/>SSE 流式渲染"] --> API
    subgraph API["FastAPI 后端"]
        KB["知识库 API"]
        DOC["文档 API"]
        CHAT["对话 API / SSE"]
        SYS["系统状态 API"]
    end
    CHAT --> RAG["RAG Service<br/>检索 + Prompt + LLM"]
    DOC --> DS["Document Service<br/>解析 → 分块 → 入库"]
    DS --> PARSE["Parser<br/>PDF/Word/Excel/Image"]
    PARSE --> OCR["OCR Backend<br/>可插拔"]
    DS --> VS["Vector Store Service"]
    RAG --> VS
    VS --> CACHE["QueryCache / EmbeddingCache"]
    VS --> CHROMA[("ChromaDB")]
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

## 技术栈

| 层次 | 选型 |
| --- | --- |
| 语言 / 包管理 | Python 3.13（uv 管理）、`uv.lock` 锁定依赖 |
| Web 框架 | FastAPI 0.141+、Uvicorn 0.54+、SSE 流式响应 |
| LLM 编排 | LangChain 1.x（`langchain-core` 1.6+、`langchain-text-splitters`）+ `langchain-ollama`（本地）/ `langchain-openai`（OpenAI 兼容云端） |
| 向量库 | ChromaDB 1.5+（persistent client）/ `langchain-chroma` |
| 关系库 | SQLite（开发默认）+ PostgreSQL 16（部署可选）+ SQLAlchemy 2.1 + Alembic 1.20 |
| 文档解析 | pypdf、PyMuPDF、python-docx、docx2txt、openpyxl、xlrd、Pillow、chardet |
| 前端 | Vue 3 + Vite + Pinia + Tailwind CSS 3 |
| 质量 | ruff、pytest（+ pytest-asyncio）、mypy |

## 快速开始

### 0. 前置条件

- Linux（当前主力开发环境）/ macOS（Windows 建议 WSL2）
- [uv](https://docs.astral.sh/uv/)（不需要本地预装 Python，uv 会按 `.python-version` 自行安装 3.13）
- 模型侧三选一：**云端 API（推荐，最快）** 需要 OpenRouter / DeepSeek 的 key；或本地
  [Ollama](https://ollama.com/)（全离线）；或 `mock` provider（零依赖，仅演示与验收链路）
- 数据库：开发默认 SQLite，**无需任何安装**；只有部署阶段才需要 Docker 或一个 PostgreSQL 实例

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
#  "llm":{"provider":"deepseek","model":"deepseek-chat","ok":true,...},
#  "embedding":{"provider":"openrouter","model":"liquid/lfm-2.5-embedding-350m:free","ok":true,...}}
```

> `status: degraded` 时看 `llm.error` / `embedding.error`：文案会直接指出该去 `.env` 改哪个变量
> （例如漏填 `OPENROUTER_API_KEY`）或哪个服务连不上。检索与问答会因此失败，但知识库、文档等
> 管理接口仍可用。

### 4. 选择模型 Provider

Chat 与 Embedding 是**两个独立开关**，改 `.env` 即可，代码无需改动：

| Provider | 可用于 | 说明 |
| --- | --- | --- |
| `ollama` | chat + embedding | `http://localhost:11434`，本地、无需 key |
| `openrouter` | chat + embedding | `https://openrouter.ai/api/v1`，OpenAI 兼容聚合网关，一个 key 用数百个模型 |
| `deepseek` | **仅 chat** | `https://api.deepseek.com/v1`；官方**没有 embeddings 接口**，写成 `EMBEDDING_PROVIDER=deepseek` 会得到明确报错 |
| `openai` | chat + embedding | `https://api.openai.com/v1`；也可指向任何 OpenAI 兼容的自建网关 |
| `mock` | chat + embedding | 不联网、不需要 key、输出确定性；用于本地演示 / CI / 降级验收（只有词面相似度，不能用来评估检索效果） |

`GET /api/system/providers` 会列出全部可选项、当前选择与 key 是否已配置。

云端 API 路线（推荐，最快）。chat 与 embedding 分属不同厂商也完全可以：

```bash
# .env —— Chat 走 DeepSeek，Embedding 走 OpenRouter
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY={{DEEPSEEK_API_KEY}}

EMBEDDING_PROVIDER=openrouter
OPENROUTER_API_KEY={{OPENROUTER_API_KEY}}
EMBEDDING_MAX_INPUT_CHARS=400    # 见下方「小上下文模型」说明
```

> 默认云端向量模型是 `liquid/lfm-2.5-embedding-350m:free`（免费、1024 维），但它的输入上下文只有
> **512 token**，而分块按字符数切（`CHUNK_SIZE=1000`），所以建议同时设 `EMBEDDING_MAX_INPUT_CHARS=400`
> 让截断行为可预期。另：免费路由的数据可能被上游留存用于训练，有合规要求时请换付费模型
> （如 `openai/text-embedding-3-small`）。

本地 Ollama 路线：

```bash
ollama pull qwen3:14b            # 对话模型
ollama pull qwen3-embedding:8b   # 向量模型（必须与建库时的 embedding 保持一致）
```

完全离线路线：`LLM_PROVIDER=mock` + `EMBEDDING_PROVIDER=mock`，不需要任何模型服务，也能跑通
「上传 → 检索 → 带引用回答」的完整链路。

### 5. 启动前端

```bash
cd frontend
npm install
npm run dev        # http://localhost:3000，通过 Vite 代理访问后端 8010
```

## 使用指南

1. **建知识库**：前端「新建知识库」或 `curl -X POST localhost:8010/api/kb -H 'Content-Type: application/json' -d '{"name":"产品手册"}'`
2. **上传文档**：前端拖拽上传，或

   ```bash
   curl -X POST localhost:8010/api/doc/upload \
     -F kb_id=1 -F files=@./手册.pdf
   ```

   上传接口立即返回 `doc_ids`，解析与向量化在后台任务里进行；用
   `GET /api/doc?kb_id=1` 查看 `status`（`pending/processing/completed/failed`）与 `error_msg`。
3. **提问**：前端对话页会以 SSE 流式渲染答案与引用来源；等价的命令行调用：

   ```bash
   curl -N -X POST localhost:8010/api/chat/stream \
     -H 'Content-Type: application/json' \
     -d '{"kb_id":1,"question":"这款产品的保修政策是什么？"}'
   ```

   SSE 事件顺序：`conv_id`（会话 ID）→ `sources`（引用来源，含相关度）→ `token`（增量文本，多条）
   → `done`（完整答案）。每个事件形如 `data: {"type":"sources","data":[...]}`。
4. **重新处理/重建索引**：单文档失败可 `POST /api/doc/{doc_id}/reprocess`；换了 embedding 模型则
   必须重建索引（见下方脚本），否则查询向量与库内向量不在同一空间。

## 配置说明

全部配置项见 `.env.example`（按 应用 / 服务端 / 数据库 / 密钥 / 模型 / 向量库 / 上传 / OCR / 检索 /
缓存 / CORS / 日志 分组）；本地实际生效的值写在 `.env`（已被 gitignore，不入库）。
几个容易踩坑的关键项：

| 配置 | 默认值 | 说明 |
| --- | --- | --- |
| `PORT` | `8010` | 后端端口（`8000` 在共享机器上常被占用），需与 `frontend/vite.config.js` 代理一致 |
| `DATABASE_URL` | `sqlite:///./data/inner_rag.db` | 开发默认 SQLite；部署切 `postgresql+psycopg://...` 后重跑 `alembic upgrade head` |
| `SQLITE_TIMEOUT` | `30` | SQLite 等锁超时（秒）；同时影响 `busy_timeout` |
| `LLM_PROVIDER` | `ollama` | chat 后端：`ollama` / `openrouter` / `deepseek` / `openai` / `mock` |
| `EMBEDDING_PROVIDER` | `ollama` | 向量后端：`ollama` / `openrouter` / `openai` / `mock`（DeepSeek 没有 embedding 接口） |
| `OPENROUTER_API_KEY` / `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` | 空 | 云端 API 密钥，只写在 `.env`，不要提交 |
| `*_CHAT_MODEL` / `*_EMBEDDING_MODEL` | 见 `.env.example` | 各 provider 的模型名；换 embedding 模型等于换向量空间，需要重建索引 |
| `EMBEDDING_MAX_INPUT_CHARS` | `0` | 单条输入的字符上限（0 = 不截断）；小上下文模型（如 512 token 的免费 embedding）建议设 `400` |
| `LLM_TEMPERATURE` / `LLM_MAX_TOKENS` / `LLM_TIMEOUT` / `LLM_MAX_RETRIES` | `0.3` / `2048` / `60` / `2` | 云端调用参数（Ollama 也复用 temperature 与输出长度） |
| `AUTO_CREATE_TABLES` | `false` | 表结构交给 Alembic；仅测试/一次性库设为 `true` |
| `OLLAMA_LLM_MODEL` / `OLLAMA_EMBEDDING_MODEL` | `qwen3:14b` / `qwen3-embedding:8b` | 本地 Ollama 的模型名 |
| `embedding_key`（建库时写入） | `provider:model` | 知识库会锁定建库时的 embedding 身份，换模型后会被校验拦下并提示重建索引 |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | 分块参数，影响召回粒度 |
| `TOP_K` / `RERANK_TOP_K` | `8` / `5` | 向量召回数 / 进入 Prompt 的条数 |
| `RETRIEVAL_SCORE_THRESHOLD` | `0.3` | 相关度阈值（cosine 语义，`1 - 距离`）；过高会导致空召回 |
| `HISTORY_MAX_MESSAGES` | `20` | 送入模型的历史消息条数（取最近 N 条） |
| `OCR_BACKEND` | `none` | `none` 或 `paddle`；关闭时图片/扫描件会明确报错而不是写入占位文本 |
| `ALLOW_LOCAL_IMPORT` | `false` | 是否允许 `import-path`（服务端文件系统读取能力），开启时配合 `LOCAL_IMPORT_ROOT` 限定目录 |
| `CORS_ORIGINS` | `http://localhost:3000,...` | 前端来源白名单 |

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
│   ├── core/                 # 配置（pydantic-settings）、数据库引擎与会话
│   ├── models/               # SQLAlchemy 2.0 ORM 模型
│   ├── schemas/              # Pydantic 请求/响应模型
│   ├── api/                  # 路由：kb / document / chat / system
│   ├── providers/            # 模型后端抽象：specs / chat / embeddings / factory（多 provider）
│   └── services/             # parser、ocr、embedding、vector_store、rag、cache、retrieval_log
├── scripts/                  # 运维与排查脚本 + start.sh
├── tests/                    # 离线 pytest 用例 + 可选的真实 API 联网验收（-m live）
├── docs/                     # 执行路线、测试门禁与阶段记录
├── frontend/                 # Vue 3 + Vite 前端
├── docker-compose.yml        # 部署用：PostgreSQL 16（默认）/ 后端容器（profile=app）
├── Dockerfile
├── .env.example              # 配置模板（真实的 .env 不入库）
└── images/                   # 截图
```

## API 一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/kb` | 知识库列表（分页、关键词） |
| POST | `/api/kb` | 新建知识库（自动写入当前 embedding 标识） |
| GET | `/api/kb/{kb_id}` | 详情（含 collection 向量数） |
| PUT / DELETE | `/api/kb/{kb_id}` | 更新 / 删除（连带删除向量与上传文件） |
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
| GET | `/api/system/health` | 健康检查（后端 + Chat/Embedding provider 连通性与错误原因） |
| GET | `/api/system/providers` | 全部可用 provider、当前选择、key 是否已配置（不返回密钥） |
| GET | `/api/system/stats` | 检索统计 + 缓存状态 |
| GET | `/api/system/config` | 前端可用的非敏感运行时配置 |
| GET | `/api/system/models` | 当前 provider 的可用模型列表（Ollama / 云端 `/models`） |
| POST | `/api/system/cache/clear?kb_id=` | 手动清理缓存（指定知识库或全清） |

## 运维与排查脚本

```bash
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
uv run pytest -m live -q       # 真实 provider 联网验收（需 OPENROUTER_API_KEY，会产生少量费用）
uv run pytest -q tests/test_api.py::test_chat_stream_events_and_persistence
uv run mypy                    # 类型检查（配置见 pyproject.toml 的 [tool.mypy]）
```

测试说明：`tests/conftest.py` 在导入应用之前就把环境切到临时 SQLite、临时目录与固定的 `mock`
provider，并用确定性的假 embedding / 假 LLM 替换真实 Provider，因此默认测试**完全离线**、可复现，
也不会产生模型调用费用，且不会受开发者本机 `.env` 的影响。`pytest -m live` 才会真实调用云端 API
（无 key 时自动 skip）。

数据库迁移：

```bash
uv run alembic revision --autogenerate -m "add xxx"   # 修改 ORM 模型后生成迁移
uv run alembic upgrade head                          # 应用
uv run alembic check                                 # 校验模型与迁移是否漂移
uv run alembic downgrade -1                           # 回退一步
```

## 部署

开发阶段用 SQLite + 云端 API 就能跑通全链路（见 `docs/roadmap.md` 的 Phase 5 收尾计划）。

只跑数据库（后端仍跑在宿主机上，改代码无需重建镜像）：

```bash
docker compose up -d postgres
```

连后端一起跑（镜像内自动 `alembic upgrade head`）：

```bash
docker compose --profile app up -d --build
docker compose logs -f api
```

后端容器对外暴露 `8010`，以非 root 用户（uid 10001）运行；数据（上传文件、Chroma、日志）
落在 `app_data` 卷的 `/data` 下（若改为绑定宿主机目录，注意该目录需允许 uid 10001 写入）。
镜像里的 `OLLAMA_BASE_URL` 默认指向 `host.docker.internal:11434`（Compose 已加 `extra_hosts`），
如需指向云端 API，直接在 `.env` 或 Compose 环境变量里覆盖。

前端生产构建：

```bash
cd frontend && npm run build          # 产物在 frontend/dist，可用任意静态服务器托管
```

## 常见问题

**Q：`/api/system/health` 返回 `degraded`？**
A：看响应里的 `llm.error` / `embedding.error`，它能直接定位原因：漏填 API Key（会指名该写哪个变量）、
provider 名写错（会列出可选值）、服务连不上（会带上 URL）或模型未拉取。也可用 `GET /api/system/providers`
看当前选择与 key 状态。

**Q：`EMBEDDING_PROVIDER=deepseek` 报错？**
A：DeepSeek 官方只有 chat completion，没有 embeddings 接口。chat 用 DeepSeek、embedding 用
OpenRouter / OpenAI / Ollama 是常见组合，两个变量本来就是独立的。

**Q：云端 embedding 报输入过长 / 结果很怪？**
A：小上下文模型（如免费的 512 token embedding）需要配 `EMBEDDING_MAX_INPUT_CHARS`（字符数）
把每个分块截到上下文以内；截断是真截断，会损失分块尾部信息，必要时同时调小 `CHUNK_SIZE`。

**Q：上传成功但文档 `status=failed`？**
A：看 `error_msg`。常见原因：扫描件/图片未启用 OCR（`OCR_BACKEND=none`）、密码保护的 PDF、
`.doc` 老格式（建议先转 `.docx`）、模型服务不可用。

**Q：提问总是「未找到相关信息」？**
A：先跑 `uv run scripts/query_probe.py <kb_id> "问题"`。若命中了分块但相关度低于
`RETRIEVAL_SCORE_THRESHOLD`，说明阈值偏高；若完全没命中，检查文档是否 `completed` 且向量数 > 0，
以及当前 embedding 模型是否与建库时一致。

**Q：换了 embedding 模型后检索结果全乱？**
A：向量空间变了，旧向量全部失效。执行 `uv run scripts/reindex_kb.py <kb_id>` 重建索引。

**Q：`init_db` 启动报错提示缺表？**
A：表结构由 Alembic 管理，先执行 `uv run alembic upgrade head`（或设置 `AUTO_CREATE_TABLES=true`
用于临时库）。

**Q：端口冲突？**
A：改 `.env` 的 `PORT`，同时改 `frontend/vite.config.js` 的代理目标（或设置 `VITE_API_TARGET`）。

**Q：SQLite 报 `database is locked`？**
A：连接建立时已开启 WAL 与 `busy_timeout`；若仍偶发，把 `.env` 的 `SQLITE_TIMEOUT` 调大，或避免
同时上传多个大文件（SQLite 同时只允许一个写事务）。真要并发写入就切 PostgreSQL。

**Q：什么时候需要切 PostgreSQL？**
A：多进程/多实例部署、并发写入较多、或需要主从备份时。改 `DATABASE_URL` 后重跑
`uv run alembic upgrade head` 即可，两者共用同一套迁移（`docs/roadmap.md` Phase 5）。

**Q：API key 放哪里？**
A：只放 `.env`（已被 `.gitignore` 忽略），`.env.example` 里只留空占位。密钥一旦泄漏，
请立即在对应平台吊销并更换。

## 路线图

- **已完成**：多 Provider 抽象层（Chat / Embedding 独立选型、`mock` 降级路径、`/api/system/providers`）、
  SQLite 优先与密钥外置、Alembic 迁移、Docker 资产
- **下一步**：OCR 迁移到视觉大模型（扫描件与图片型 PDF）
- **之后**：检索质量评测集与 Rerank、任务队列与结构化日志、CI 与 PostgreSQL 部署验证

每一阶段的交付物、测试门禁（G0/G1/G2）与进度记录见 `docs/roadmap.md`。

## License

MIT
