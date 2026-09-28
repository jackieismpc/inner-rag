# inner-rag

> 多 Provider 的企业知识库问答系统 —— FastAPI + LangChain 1.x + ChromaDB + PostgreSQL

`inner-rag` 是一个完整可跑的 RAG（检索增强生成）知识库系统：把散落的文档（PDF / Word / Excel /
纯文本，扫描件与图片走 OCR）解析、分块、向量化入库，然后基于「向量检索 + 引用溯源 + 流式问答」
回答问题。LLM 与 Embedding 都按 Provider 解耦配置，既能在本地用 Ollama 跑全离线链路，也能直接
接 OpenRouter / DeepSeek 等云端 API。

![主界面](images/main.png)

## 背景与目标

这个项目来自一次真实的重构。最初的 `rag` 项目能跑通「上传 → 检索 → 问答」的主流程，但工程上
积累了不少问题：LangChain 已升级到 1.x 而代码还在用被移除的旧 API（`langchain.text_splitter`
等），依赖没有版本锁定、环境靠手写 venv + pip，只能绑定 Ollama 一种模型后端，数据库用 MySQL
且建表靠运行时 `create_all`，检索相关度算错、会话历史取错、后台任务用已关闭的 Session 等逻辑
缺陷也散落在各处。

因此有了 `inner-rag`：保留原项目的前后端形态作为起点，按阶段做一次彻底的重构，目标是让它成为
一个「能拿得出手、也能真正部署」的工程化项目。四个核心目标：

1. **升级到最新的 LangChain 生态**（LangChain 1.x / core 1.6+）与匹配的 Python 3.13；
2. **用 `uv` 管理** Python 版本、依赖与虚拟环境（含 `uv.lock` 锁定）；
3. **模型后端可插拔**：不只支持 Ollama，也能直接用 OpenRouter / DeepSeek / 任意 OpenAI 兼容端点；
4. **全面工程化重构**：PostgreSQL 16 + Alembic 迁移、可插拔 OCR、检索质量可观测、测试与 CI 友好的结构。

阶段划分（详见文末「阶段进展」）：

| 阶段 | 目标 | 状态 |
| --- | --- | --- |
| Phase 0 | 建立独立的 uv 项目与 git 基线（src 布局、可按包安装） | ✅ |
| Phase 1 | 全量升级陈旧 API + 修复必修逻辑缺陷，行为保持稳定 | ✅ |
| Phase 2 | Provider 抽象层（Ollama / OpenRouter / DeepSeek / OpenAI 兼容） | 待开始 |
| Phase 3 | OCR 迁移到视觉大模型（VLM），替代本地 PaddleOCR | 待开始 |
| Phase 4 | 检索质量评测、Rerank、Redis 缓存、任务队列、CI | 待开始 |

## 特性

- **多格式文档解析**：PDF（逐页）、Word（docx / doc）、Excel（xlsx / xls）、文本类
  （txt / md / csv / json / xml / html）；图片与扫描页走可插拔 OCR 后端
- **向量化与检索**：ChromaDB 持久化，每个知识库一个独立 collection，统一 cosine 空间
- **多策略检索**：`similarity`（余弦相似度）/ `mmr`（多样性去重）/ `hybrid`（两者融合），
  返回**真实**相关性分数（1 - 余弦距离），MMR 召回项如实标注为「无分数」而不是伪造 1.0
- **流式问答**：SSE 逐 token 推送，先推引用来源再推答案，前端实时渲染并展示相关度
- **会话管理**：多轮对话（取**最近** N 条历史）、会话列表、消息与引用来源持久化
- **性能与成本控制**：Embedding 缓存（按模型隔离 key）+ 检索结果缓存（LRU + TTL，按知识库精确失效）、
  批量嵌入 + 信号量限流
- **可观测性**：检索日志、Prompt 日志、缓存命中率 / 空召回率 / 平均延迟统计
- **工程化**：uv 锁依赖、Alembic 迁移、CORS 白名单、文件名与导入路径安全校验、
  ruff 静态检查、44 个离线 pytest 用例、Dockerfile + docker compose

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
    VS --> EMB["Embedding Provider"]
    RAG --> LLM["LLM Provider"]
    KB --> DB[("PostgreSQL 16")]
    DOC --> DB
    CHAT --> DB
```

检索链路：查询 → 检索缓存 → 向量库（cosine 距离换算为相关度）→ 阈值过滤 → 组装 Prompt
（含最近几轮对话历史）→ LLM（流式 / 非流式）→ 落库并返回引用来源。

## 技术栈

| 层次 | 选型 |
| --- | --- |
| 语言 / 包管理 | Python 3.13（uv 管理）、`uv.lock` 锁定依赖 |
| Web 框架 | FastAPI 0.141+、Uvicorn 0.54+、SSE 流式响应 |
| LLM 编排 | LangChain 1.x（`langchain-core` 1.6+、`langchain-text-splitters`、`langchain-ollama`） |
| 向量库 | ChromaDB 1.5+（persistent client）/ `langchain-chroma` |
| 关系库 | PostgreSQL 16 + SQLAlchemy 2.1 + Alembic 1.20 + psycopg 3 |
| 文档解析 | pypdf、PyMuPDF、python-docx、docx2txt、openpyxl、xlrd、Pillow、chardet |
| 前端 | Vue 3 + Vite + Pinia + Tailwind CSS 3 |
| 质量 | ruff、pytest（+ pytest-asyncio）、mypy |

## 快速开始

### 0. 前置条件

- Linux / macOS（Windows 建议 WSL2）
- [uv](https://docs.astral.sh/uv/)（不需要本地预装 Python，uv 会按 `.python-version` 自行安装 3.13）
- Docker（用于一键拉起 PostgreSQL；也可以指向已有实例）
- 可选的本地模型服务 [Ollama](https://ollama.com/)（不用 Ollama 时按 Phase 2 配置云端 API）

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

### 2. 启动数据库（PostgreSQL 16）

```bash
docker compose up -d postgres
```

不使用 Docker 时，把 `.env` 里的 `DATABASE_URL` 指向任意 PostgreSQL 16 实例即可。
本地临时体验也可以直接用 SQLite：

```bash
DATABASE_URL='sqlite:///./data/local.db' AUTO_CREATE_TABLES=true ...
```

### 3. 配置环境变量

```bash
cp .env.example .env
# 按需修改：DATABASE_URL、OLLAMA_BASE_URL / 模型名、PORT、CORS_ORIGINS 等
```

### 4. 建表并启动后端

表结构由 Alembic 管理（`AUTO_CREATE_TABLES=false`），首次启动前必须先迁移：

```bash
uv run alembic upgrade head
uv run uvicorn inner_rag.main:app --reload --port 8010
# 或者一步到位（自动 uv sync + 迁移 + 热重载）：
./scripts/start.sh
```

打开 <http://localhost:8010/docs> 查看交互式 API 文档，`/api/system/health` 检查后端与模型连通性：

```bash
curl -s http://localhost:8010/api/system/health
# {"status":"healthy","ollama":true,"llm_model":"qwen3:14b",...}
```

> `ollama: false` / `status: degraded` 说明 Ollama 未启动或模型名不对；检索与问答会因此失败，
> 但知识库、文档等管理接口仍可用。

### 5. 准备本地模型（Ollama 路线）

```bash
ollama pull qwen3:14b            # 对话模型
ollama pull qwen3-embedding:8b   # 向量模型（必须与建库时的 embedding 保持一致）
```

### 6. 启动前端

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

全部配置项见 `.env.example`（按 应用 / 服务端 / 数据库 / 模型 / 向量库 / 上传 / OCR / 检索 / 缓存 /
CORS / 日志 分组）。几个容易踩坑的关键项：

| 配置 | 默认值 | 说明 |
| --- | --- | --- |
| `PORT` | `8010` | 后端端口（`8000` 在共享机器上常被占用），需与 `frontend/vite.config.js` 代理一致 |
| `DATABASE_URL` | `postgresql+psycopg://rag:rag@localhost:5432/rag_db` | 支持 `sqlite:///...` 用于临时体验 |
| `AUTO_CREATE_TABLES` | `false` | 表结构交给 Alembic；仅测试/一次性库设为 `true` |
| `OLLAMA_LLM_MODEL` / `OLLAMA_EMBEDDING_MODEL` | `qwen3:14b` / `qwen3-embedding:8b` | embedding 标识会写入知识库并做一致性校验 |
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
│   └── services/             # parser、ocr、embedding、vector_store、rag、cache、retrieval_log
├── scripts/                  # 运维与排查脚本 + start.sh
├── tests/                    # 离线 pytest 用例（假 embedding / 假 LLM）
├── frontend/                 # Vue 3 + Vite 前端
├── docker-compose.yml        # PostgreSQL 16（默认）/ 后端容器（profile=app）
├── Dockerfile
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
| GET | `/api/system/health` | 健康检查（后端 + 模型连通性） |
| GET | `/api/system/stats` | 检索统计 + 缓存状态 |
| GET | `/api/system/config` | 前端可用的非敏感运行时配置 |
| GET | `/api/system/models` | Ollama 可用模型列表 |
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
uv run pytest                  # 全量测试（离线，不依赖 Ollama / PostgreSQL）
uv run pytest -q tests/test_api.py::test_chat_stream_events_and_persistence
uv run mypy                    # 类型检查（配置见 pyproject.toml 的 [tool.mypy]）
```

测试说明：`tests/conftest.py` 在导入应用之前就把环境切到临时 SQLite 与临时目录，并用确定性的
假 embedding / 假 LLM 替换真实 Provider，因此测试**完全离线**、可复现，也不会产生模型调用费用。

数据库迁移：

```bash
uv run alembic revision --autogenerate -m "add xxx"   # 修改 ORM 模型后生成迁移
uv run alembic upgrade head                          # 应用
uv run alembic check                                 # 校验模型与迁移是否漂移
uv run alembic downgrade -1                           # 回退一步
```

## Docker

只跑数据库（本地开发最常见）：

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

## 阶段进展与重构说明

### Phase 0：uv 项目与 git 基线 ✅

- 新仓库 `inner-rag`，`git init` 全新历史；原 `rag` 项目作为起点原样复制后提交基线
- 采用 `src/inner_rag/` 布局，`pyproject.toml`（hatchling）声明包，`uv.lock` 锁定 100+ 依赖
- Python 由 uv 管理（`.python-version` = 3.13），不再依赖本地已装的解释器

### Phase 1：全量升级陈旧 API + 修复必修缺陷 ✅

依赖与 API 升级（保留原行为）：

- `langchain.text_splitter` 在 LangChain 1.x 已移除 → 改用 `langchain-text-splitters`
- `ChatOllama(streaming=...)` 等旧参数移除 → 流式由调用方 `astream` 决定
- Pydantic v1 风格 `class Config` → `model_config = ConfigDict(from_attributes=True)`
- `datetime.utcnow()` → `datetime.now(UTC)`；`asyncio.get_event_loop()` → `asyncio.to_thread()`
- SQLAlchemy `declarative_base()` → `DeclarativeBase` / `Mapped` / `mapped_column`
- 关系库从 MySQL 切到 PostgreSQL 16 + psycopg 3，建表从运行时 `create_all` 改为 Alembic 迁移

必修缺陷（有意的行为变更）：

| 问题 | 原行为 | 现行为 |
| --- | --- | --- |
| 相关度算错 | `similarity_search_with_relevance_scores` 返回的其实是原始距离；MMR 项被虚构为 0.8/1.0 | 统一用 `similarity_search_with_score` + `1 - 距离` 换算并 clamp 到 `[0,1]`；MMR 项如实返回 `null` |
| 会话历史取错 | 取**最早** 20 条消息 | 取**最近** `HISTORY_MAX_MESSAGES` 条 |
| 后台任务用坏 Session | 把请求级 `Session` 传进 `BackgroundTasks`（响应返回时已关闭） | 后台任务自行创建/关闭 Session |
| 上传无大小前置校验 | 先整体读进内存再判断大小 | 流式落盘 + 边写边校验，超限立即中断并清理 |
| 文件名未净化 | 直接用上传的文件名拼路径（可路径穿越） | 只取 basename + 随机前缀，落地在该知识库目录内 |
| 检索缓存粒度粗 / 不失效 | 按 query 全局缓存，入库/删除后仍返回旧结果 | key 带 `kb_id`，入库、删除、重建后精确失效该知识库 |
| 静默吞异常 | 检索异常被吞掉后返回空结果 | 异常向上抛出，由 API 层给出明确错误 |
| 解析失败无提示 | 扫描件/图片写入占位文本污染索引 | 显式抛错，提示启用 OCR |
| CORS 不安全 | `allow_origins=["*"]` + `allow_credentials=True`（浏览器实际会拒绝） | 白名单化，由 `CORS_ORIGINS` 配置 |
| embedding 换模型无校验 | 静默检索到不同向量空间的向量 | 知识库记录 embedding 标识，写入/检索时校验并给出重建指引 |
| 前端代理失效 | `baseURL` 写成 `' http://localhost:8000/api'`（含前导空格，且绕过 Vite 代理） | 改为相对路径 `/api`，代理目标统一为后端 `8010` |

### Phase 2 ~ Phase 4：规划中

- **Phase 2 Provider 抽象**：`ChatProvider` / `EmbeddingProvider` 工厂，支持 Ollama / OpenRouter /
  DeepSeek / OpenAI 兼容端点，Chat 与 Embedding 独立选型，`/api/system/providers` 健康检查
- **Phase 3 OCR 升级**：用视觉大模型（OpenRouter / HuggingFace VLM）替代本地 PaddleOCR，
  处理扫描件与图片型 PDF
- **Phase 4 检索质量与交付**：构建评测集（召回率 / 引用准确率 / 延迟）、引入 Rerank 精排、
  Redis 缓存与任务队列、CI（ruff + pytest + 镜像构建）

## 常见问题

**Q：`/api/system/health` 返回 `degraded`？**
A：后端正常，但连不上 Ollama。检查 `OLLAMA_BASE_URL`、`ollama serve` 是否运行、模型名是否已 `pull`。

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

## License

MIT
