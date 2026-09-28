# 架构与可插拔契约

本文回答三件事：**代码怎么分层**、**每个插件点的契约是什么**、**怎么加一个新后端而不动业务代码**。
阶段路线见 `docs/DEVELOPMENT_PLAN.md`。

## 1. 分层与依赖方向

```
src/inner_rag/
├── main.py            # FastAPI 装配：lifespan（配置校验 + 生产密钥校验）、中间件顺序（CORS 最外层）、
│                      #   异常处理器、路由挂载、docs 开关（ENABLE_DOCS）
├── api/               # HTTP 边界：只做参数校验与序列化，不做业务
│   ├── auth.py        #   登录 / 当前用户
│   ├── deps.py        #   鉴权依赖：Token → user；ensure_kb_access / ensure_doc_access（把 ACL 映射成 HTTP 语义）
│   ├── chat.py        #   对话（含 SSE 流式）
│   ├── document.py    #   上传 / 列表 / 删除 / 重新向量化
│   ├── kb.py          #   知识库 CRUD + 成员授权
│   └── system.py      #   health / providers / stats / metrics
├── services/          # 业务逻辑（与具体后端解耦的层）
│   ├── parser.py      #   文本抽取（PDF / DOCX / TXT / MD …）
│   ├── ocr.py         #   扫描件 / 图片文字识别（可插拔后端）
│   ├── document.py    #   解析 → 分块 → 入库的编排与状态机
│   ├── embedding.py   #   embedding 门面 + 缓存 + identity
│   ├── vector_store.py#   向量写入与检索（Chroma 适配）
│   ├── cache.py       #   查询缓存 / 嵌入缓存（LRU）
│   ├── rag.py         #   检索 → Prompt 组装 → LLM 生成（含流式）
│   └── retrieval_log.py # 检索与 Prompt 统计
├── providers/         # 模型后端插件层
│   ├── specs.py       #   ProviderSpec 元数据 + 解析与校验（ProviderError）
│   ├── chat.py        #   Chat 实例构造（每 provider 一个分支）
│   ├── embeddings.py  #   Embedding 实例构造
│   └── factory.py     #   对外门面：get_chat_model / get_embeddings / chat_health
├── models/            # SQLAlchemy ORM（knowledge_base / document / conversation / user / kb_member）
├── schemas/           # Pydantic 出入参
└── core/              # 纯策略与基础设施：不依赖 FastAPI 请求对象、不返回 HTTP 语义
    ├── config.py      #   唯一配置入口
    ├── database.py    #   引擎 / 会话
    ├── security.py    #   口令哈希（argon2id）与 Token 签发 / 校验（JWT）
    ├── context.py     #   身份 ContextVar + IdentityContextMiddleware（纯 ASGI）
    └── access.py      #   ACL 判定：level_of / level_map / accessible_kb_ids
```

依赖方向**只能向下**：

```
api  →  services  →  providers / core
                       ↑
              （services 不 import api；providers 不 import services）
```

硬性规则：

1. `services/` 不许出现 `if provider == "xxx"` 这类后端分支；后端差异只能通过 `providers/` 门面暴露。
2. `api/` 不许直接用 SQLAlchemy 会话或 provider 实例，只调 `services/`。
3. `core/config.py` 是唯一读环境变量的地方；其它模块一律 `from inner_rag.core.config import settings`。
4. 新增依赖必须进 `pyproject.toml`；可选后端的重依赖用 `uv sync --extra` 分组，不能变成必装。
5. `core/` 只做判定与查询，**不抛 `HTTPException`**：401/403 由 `api/deps.py` 映射。
   这样权限策略能被脚本等非 HTTP 入口复用，且「谁是策略、谁是协议」界限清楚。

## 2. 插件点总表

| 插件点 | 接口/门面（现状） | 内置实现 | 配置项 | 探活 | 契约测试 |
| --- | --- | --- | --- | --- | --- |
| Chat 模型 | `providers/factory.py::get_chat_model` | ollama / openrouter / deepseek / openai / mock | `LLM_PROVIDER`、`*_CHAT_MODEL`、`*_API_KEY`、`LLM_REASONING_EFFORT` | `chat_health()` → `/api/system/health` | `tests/test_providers.py` |
| Embedding 模型 | `providers/factory.py::get_embeddings` | ollama / openrouter / openai / mock | `EMBEDDING_PROVIDER`、`*_EMBEDDING_MODEL`、`EMBEDDING_MAX_INPUT_CHARS` | `providers_catalog()` → `/api/system/providers` | `tests/test_providers.py` |
| 向量库 | `services/vector_store.py::VectorStoreService`（Phase 4 抽出 `VectorStore`） | **zvec**（Alibaba 开源嵌入式向量库，目标实现，Phase 4 迁移）；当前实现为 ChromaDB（cosine，每库一 collection） | `CHROMA_PERSIST_DIR`（当前）/ `ZVEC_PATH`（Phase 4）、`CHUNK_SIZE`、`CHUNK_OVERLAP` | 建库时 `get_store()` 探活 | `tests/test_vector_store.py` |
| 关系库 | `core/database.py` + Alembic | SQLite（默认）/ PostgreSQL | `DATABASE_URL` | `lifespan` 里 `check_database()` | `tests/test_api.py` |
| 缓存 | `services/cache.py` | 进程内 LRU（query + embedding 两套） | `CACHE_*`、`EMBEDDING_CACHE_SIZE` | 无（进程内） | `tests/test_cache.py` |
| 后台任务 | FastAPI `BackgroundTasks` | 进程内 | — | 文档状态机可观测 | `tests/test_api.py` |
| OCR | `services/ocr.py` | none（默认）/ paddle / vlm（Phase 9） | `OCR_BACKEND`、`OCR_LANG` | 启动时记录后端与可用性 | `tests/test_parser.py` |
| 追踪 / 指标 | `core/observability.py`（Phase 5 新增） | loguru + LangSmith（+ 预留 OTLP） | `LANGSMITH_*`、`LOG_FORMAT`、`METRICS_BACKEND` | `/api/system/metrics` | Phase 5 新增 |
| 评测器 | `services/evaluation.py`（Phase 6 新增） | 指标 + LLM-as-judge | 评测集路径、judge 模型 | 报告产出 | Phase 6 新增 |
| 身份 / 权限 | `core/security.py`（策略原语）+ `core/access.py`（ACL）+ `api/deps.py`（HTTP 映射） | 本地账号（argon2id 口令哈希）+ JWT（HS256）；知识库级 ACL：`owner` / 成员 `read` / 成员 `write` | `AUTH_SECRET_KEY`、`AUTH_TOKEN_TTL_MINUTES`、`ENABLE_DOCS` | `/api/system/health` 免鉴权（白名单另有 `/api/auth/login`） | `tests/test_auth.py` |

「五件套」标准：**接口 + 内置实现 + 配置项 + 探活 + 契约测试**。少任何一件都不算可插拔完成——
尤其是探活与契约测试，这两件最容易漏，漏了就会在换后端时才发现问题。

## 3. 各插件点契约

### 3.1 Chat / Embedding provider

现状（`providers/specs.py`）：

- 每个后端由 `ProviderSpec(name, label, kind, model, base_url, api_key, api_key_env, docs_url, notes,
  model_list_authoritative)` 描述；
- `_build(kind, name)` 把 `.env` 翻成 spec；`chat_spec()` / `embedding_spec()` 负责**解析 + 校验**；
- 校验失败抛 `ProviderError`，消息里直接写明「改哪个变量」；
- `ProviderSpec.identity` = `provider:model`，写进知识库，用于**向量空间一致性**校验
  （`VectorStoreService.ensure_embedding_matches`，不一致会抛 `EmbeddingIdentityMismatch` 并提示
  `scripts/reindex_kb.py`）。

Phase 7 目标形态（把「分支」换成「注册表」）：

```python
# src/inner_rag/plugins/registry.py
ChatBuilder = Callable[[ProviderSpec], BaseChatModel]
EmbeddingBuilder = Callable[[ProviderSpec], Embeddings]


def register_chat(name: str, builder: ChatBuilder) -> None: ...
def register_embedding(name: str, builder: EmbeddingBuilder) -> None: ...
def load_entry_points(group: str = "inner_rag.chat_providers") -> None: ...
def get_chat_builder(name: str) -> ChatBuilder: ...
```

契约（每个新后端都要满足）：

1. **只依赖 spec**：builder 只接收 `ProviderSpec`，不自己去读环境变量。
2. **不联网构造**：实例化时不许发请求（探活是独立方法），否则启动就卡。
3. **注入密钥用 `SecretStr`**：日志与异常里不能出现明文 Key。
4. **错误归一**：网络/鉴权/参数错误统一包成 `ProviderError`，消息含 provider、model 与可变项提示。
5. **身份稳定**：`identity` 变了就必须提示重建索引，不允许静默跨向量空间检索。

### 3.2 向量库（`VectorStore`）

**实现者**：zvec（Alibaba 开源嵌入式向量库，项目选型与目标实现，Phase 4 迁移）/ chroma（当前实现，
迁移完成后降为兼容实现）。下面是 Phase 4 抽出的接口，语义按现有 Chroma 行为定义：

```python
class VectorStore(Protocol):
    async def add_documents(
        self, kb_id: int, docs: list[Document], doc_id: int, filename: str
    ) -> int: ...  # 返回写入分块数
    async def search(
        self,
        kb_id: int,
        query: str,
        k: int,
        strategy: Strategy,
        score_threshold: float | None,
        filter_doc_ids: list[int] | None,
    ) -> tuple[list[tuple[Document, float | None]], int]: ...  # (结果, 被滤掉数)
    async def delete_kb(self, kb_id: int) -> None: ...
    async def delete_document(self, kb_id: int, doc_id: int) -> int: ...
    def count(self, kb_id: int) -> int: ...
```

必须遵守的语义（这也是契约测试要断言的）：

- **相关度口径**：对外一律 `[0, 1]` 且越大越相关（cosine 下 `relevance = 1 - distance`），
  不允许把原始距离当相关度返回；
- **MMR 无分数**：`strategy="mmr"` 的条目 `score=None`，不过阈值过滤，排序时排在有分数之后；
- **阈值过滤计数**：被 `score_threshold` 滤掉的条数要返回，供指标统计「空召回率」；
- **元数据是标量字符串**：`doc_id` / `kb_id` / `chunk_index` 存字符串，`page` 存数字，
  Phase 6 起新增 `page_start` / `page_end`；
- **写入幂等性边界**：同一文档重新向量化前必须先 `delete_document`，避免重复分块累积。

zvec 适配器（Phase 4）要把上面这些语义映射到 zvec SDK，并逐条写进契约测试：

| 契约方法 | zvec 侧动作 | 实现要点 |
| --- | --- | --- |
| `add_documents` | `collection.insert([Doc(id, vectors, fields)])` + `optimize()` | `id` 由 `kb_id/doc_id/chunk_index` 组装，保证重复写入可覆盖；调用前先 `delete_document` |
| `search` | `collection.query(queries=Query(field, vector), topk=k, filter=...)` | `filter` 只用于 doc 级过滤（每个知识库一个 collection）；COSINE 度量返回的值必须**显式换算成 `1 - distance`** 再对外，换算前后都用断言固定住 |
| `count` | `collection.stats` | 必须与关系库里的分块数一致，供 `/api/kb/{id}` 与向量体检对账 |
| `delete_kb` / `delete_document` | `collection.delete(...)` | 删除后 `count` 必须归零；整库重建用新建目录 + 重跑建库，不做原地格式转换 |

索引参数（HNSW + cosine）、向量维度与 collection schema 必须来自构造参数（接口层传入），
适配器不许自己去读配置；换 embedding 导致的维度变化由既有的 `EmbeddingIdentityMismatch` 拦住。

### 3.3 关系库与 Repository

现状：SQLAlchemy 2.x 同步 ORM + Alembic；SQLite 打开 WAL、外键与 `busy_timeout`；PostgreSQL 共用同一套
迁移（`migrations/`）。Phase 7 引入 repository 边界：

```python
class KnowledgeBaseRepository(Protocol):
    def create(...) -> KnowledgeBase: ...
    def list(self) -> list[KnowledgeBase]: ...
    def get(self, kb_id: int) -> KnowledgeBase | None: ...
    def delete(self, kb_id: int) -> None: ...

class DocumentRepository(Protocol):   # create / update_status / list_by_kb / get / delete
class ConversationRepository(Protocol):  # append / history / clear
```

契约：

- **Alembic 是唯一 schema 来源**，不许用 `Base.metadata.create_all()` 建生产表；
- 迁移必须可 `downgrade`（G1 会验证 `upgrade → check → downgrade → upgrade`）；
- SQLite 与 PostgreSQL 行为一致：时间戳存 UTC、JSON 字段用通用类型、字符串长度显式声明；
- 约定「一个请求一个会话」，服务层方法接收 session，不自己 `SessionLocal()`。

### 3.4 缓存

```python
class CacheBackend(Protocol):
    def get(self, namespace: str, key: str) -> Any | None: ...
    def set(self, namespace: str, key: str, value: Any, ttl: float | None = None) -> None: ...
    def invalidate(self, namespace: str, pattern: str | None = None) -> int: ...
```

- query 缓存按 `kb_id` 精确失效（文档增删改后必须让该库全部失效）；
- embedding 缓存按 `identity`（`provider:model`）隔离，跨模型不许命中；
- Phase 7 加 Redis 实现时，序列化必须版本化（`cache_schema_version`），避免上线后读到旧结构。

### 3.5 任务队列

```python
class TaskQueue(Protocol):
    async def submit(
        self, name: str, fn: Callable[..., Awaitable[None]], *args, **kwargs
    ) -> str: ...
    def status(self, task_id: str) -> TaskStatus: ...  # pending/running/succeeded/failed/retrying
```

契约：任务必须**幂等**（重试不会重复入库，靠 `delete_document` + 状态机保证）、失败要留可读原因
（写回 `document.error`）、并发要有上限（避免免费额度被瞬间打爆）。

### 3.6 OCR

`OCR_BACKEND=none|paddle|vlm`：

- `none`：遇到图片 / 扫描件**显式失败**并说明如何开启，不许静默丢内容；
- `paddle`：本地推理，重依赖，按 extra 安装；
- `vlm`（Phase 9）：走 OpenAI 兼容接口传 base64 图片，成本进 trace。

### 3.7 追踪与指标（Phase 5）

`core/observability.py` 暴露 `Tracer` 门面；LangSmith 不可用时降级为本地计时日志。
细节（trace 树、metadata 约定、日志 schema、指标）见 `docs/observability.md`。

### 3.8 身份与访问控制

**范围**：单租户 + 本地账号 + 知识库级 ACL。明确不做多租户、部门隔离、文档级权限、SSO/LDAP、
审计、注册接口与 logout 接口（见 `docs/DEVELOPMENT_PLAN.md` Phase 3 与第 9 节 ADR）。

**分层**：策略在 `core/access.py`，HTTP 映射在 `api/deps.py`（`core/` 不产生 HTTP 语义，见第 1 节规则 5）。

- **账号与口令**：账号存 `users` 表，口令只存 **argon2id** 哈希（`argon2-cffi`，自带随机盐），
  校验用 `PasswordHasher.verify`，不自己实现比较逻辑。从未设置过口令的账号写哨兵值 `"!"`
  ——它在任何输入下都不可能验证通过（迁移期为已存在的库自动创建的管理员账号就是该状态，
  需 `--reset-password` 激活）。
- **登录与 Token**：`POST /api/auth/login` 校验通过后签发 **JWT（HS256，PyJWT）**，载荷只有
  `sub`（user_id）、`iat`、`exp`（`AUTH_TOKEN_TTL_MINUTES`，默认 720 分钟），**不装权限快照**
  ——权限每次请求实时查库，因此「改权限 / 移除成员」立即生效、无需重新登录。
  登录失败（用户名不存在、口令错、账号停用）统一返回同一条 401 文案，不泄露账号是否存在。
- **密钥强度**：`DEBUG=false` 时若 `AUTH_SECRET_KEY` 仍是默认值或短于 32 字节，应用
  **拒绝启动**（`verify_production_secret`）——短 HMAC 密钥可被离线爆破，属于启动期就该炸的错误。
- **身份注入**：`IdentityContextMiddleware`（纯 ASGI，不用 Starlette 的请求对象）解析
  `Authorization: Bearer`，把 `user_id` 写进 `contextvars`；日志格式统一带 `user=`（`core/context.py`
  的 `log_user()`），**不靠每个函数手动传参**。
- **授权（ACL）**：`knowledge_bases.owner_id`（NOT NULL、FK `RESTRICT`、带索引）+ `kb_members
  (kb_id, user_id, permission)`。三级：`owner` > `write`（含 `read`）> `read`；无记录即无权。
- **HTTP 语义**：未登录 / Token 过期、篡改、账号已停用 → **401**（带 `WWW-Authenticate: Bearer`）；
  已登录但无权访问目标 kb → **403**；kb 不存在 → **404**；会话 ID 属于别的库同样 404（不泄露存在性）。
  `GET /api/kb` 只返回 `accessible_kb_ids()` 的结果。
- **检索边界**：权限判定发生在进入服务层**之前**——每个涉及 kb 的路由都挂 `ensure_kb_access` 守卫；
  不允许在 `services/rag.py` 里用「检索后再过滤掉不可见的 kb」来补。
- **免鉴权白名单**：只有 `POST /api/auth/login` 与 `GET /api/system/health`（后者必须免鉴权且永不 5xx）；
  `/docs`、`/redoc`、`/openapi.json` 由 `ENABLE_DOCS` 控制，生产建议关闭。

**已知取舍**（写明是为了不被当成 bug）：Token 存 localStorage（无 logout 接口，客户端丢弃即可）；
停用账号（`is_active=false`）会让已签发 Token 立即失效，但**改口令不会**——JWT 无状态，
短 TTL 是泄漏后的唯一收敛手段（README「配置说明」）。

## 4. 关系库 vs 向量库：各存什么、怎么对账

两个**不同职责**的存储，缺一不可，也不互相替代。

| 维度 | 关系库（SQLite / PostgreSQL） | 向量库（zvec） |
| --- | --- | --- |
| 存什么 | 结构化事实：知识库、文档元数据与状态、会话与消息、引用来源、用户与 ACL | 分块文本 + 其**向量**，以及检索用元数据（`doc_id` / `kb_id` / `chunk_index` / `page_start` / `page_end`） |
| 回答什么问题 | 「有哪些库、哪些文档、处理到哪一步了、谁问了什么」 | 「哪些分块的语义最接近这个问题」 |
| 查询方式 | SQL：等值 / 范围 / 排序 / 事务（ACID） | 近似最近邻（ANN，HNSW + cosine 距离） |
| 索引依据 | 主键 / 外键 / 普通索引 | 向量索引（HNSW 图），依赖 embedding 空间 |
| 一致性角色 | **权威（source of truth）**：文档状态机、`kb.embedding_key`、ACL 都在这里 | **可重建的派生物**：换 embedding 或分块参数后重跑建库即可 |
| 规模量级 | 以行为单位（开发期单文件 SQLite / 部署期 PG） | 与分块数成正比（全库 ~2,900–3,000 个分块、1024 维） |
| 能看到什么 | 文档数、分块数、状态分布 | 只能查到向量数（`count()`），不知道业务状态 |

分工与对账：

1. **先写关系库，再写向量库**：上传文档先落 `documents`（状态 `pending`），解析分块后写向量，
   成功才推进到 `completed`。状态机是「文档能不能被检索」的唯一判据。
2. **对账口径**：向量库 `count(kb_id)` 必须等于关系库中该库 `completed` 文档的分块数之和；
   `scripts/check_vectors.py` 与 `/api/kb/{kb_id}` 都按这个口径体检，不一致就是 bug。
3. **重建而非修补**：向量是派生物，换了 embedding / 分块参数 / 向量库实现后，**不试图原地修改**，
   走「新建库目录 + 重跑建库」，关系库里的业务数据不动（见 `scripts/reindex_kb.py`）。
4. **一致性边界**：不做跨存储的分布式事务（本系统不需要）；允许「向量已写、状态未推进」的中间态，
   但**不允许**「状态 `completed` 而向量缺失」——这由对账脚本与重试兜住。

一句话：关系库存「**我们知道什么**」，向量库存「**怎么找到它**」。

## 5. 怎么加一个新后端（分步指南）

**示例 A：加一个 OpenAI 兼容的 chat 网关 `mygateway`**

1. `core/config.py` 增 `MYGATEWAY_BASE_URL` / `MYGATEWAY_API_KEY` / `MYGATEWAY_CHAT_MODEL`；
2. `providers/specs.py`：加进 `CHAT_PROVIDERS`，在 `_build()` 里补 `ProviderSpec`（含 `docs_url`、
   `api_key_env`、`model_list_authoritative`）；
3. `providers/chat.py`：加一个 builder 分支（或 Phase 7 后 `registry.register_chat("mygateway", ...)`）；
4. `.env.example` 补三行与注释；
5. 测试：`tests/test_providers.py` 加「已配置 → OK / 缺 Key → 可读报错 / 模型不在列表 → warning」；
   需要时在 `tests/test_live_providers.py` 加 `-m live` 用例；
6. 文档：README 的 provider 表加一行。

**示例 B：加一个向量库 `pgvector`**（内置的是 zvec，见 3.2；这里演示再引入第三方）

1. `services/vector_store/`（Phase 4 拆包）下新增 `pgvector.py`，实现第 3.2 节全部方法；
2. 新增配置 `VECTOR_STORE=zvec|chroma|pgvector`，并在 `factory.get_vector_store()` 里查表；
3. 依赖进 `pyproject.toml` 的可选 extra；
4. **跑同一套契约测试**（`tests/contracts/test_vector_store_contract.py`，参数化跑所有实现）；
5. 迁移或建表脚本 + README「换向量库」小节（含「必须重建索引」的警告）。

判定标准：如果为了接一个新后端你改了 `services/rag.py` 或 `api/*.py`，说明抽象漏了，先补接口再继续。

## 6. 错误与降级契约

| 场景 | 行为 | 用户看到什么 |
| --- | --- | --- |
| provider 名未知 / 需要 Key 但没填 / 不支持该能力 | 构造实例时抛 `ProviderError` | HTTP 503 + 文案指明改 `LLM_PROVIDER` / `EMBEDDING_PROVIDER` / 具体 Key 变量 |
| 知识库 embedding 与当前配置不一致 | `EmbeddingIdentityMismatch` | 409/503 + 「切回原模型或跑 `scripts/reindex_kb.py`」 |
| provider 的 `/models` 不含当前模型但该端点不权威 | 只告警 | `/api/system/health` 的 `llm.ok=true` 且 `llm.warning` 有值 |
| provider 的 `/models` 权威且不含当前模型 | 判为不可用 | `llm.ok=false` + 错误里列出可选模型名 |
| 未登录 / Token 过期或篡改 / 账号已停用 | 401（不带内部错误细节） | 「请重新登录」；前端跳登录页 |
| 已登录但无该知识库权限 | 403 | 「无权访问该知识库」；列表接口不返回无权限的库 |
| 目标 kb / 文档 / 会话不存在或属于其它库 | 404 | 「知识库不存在」等；不区分「无权限」与「不存在」以外的信息 |
| LangSmith 未配置或不可达 | 降级为本地日志 + 计时 | 无感知，仅启动/首次请求一条 warning |
| LLM 生成中途失败（流式已发头） | 发一个 `event: error` 帧并结束 | 前端展示错误，不静默截断 |

三条硬约束：

1. **`/api/system/health` 永不 5xx**：探活失败也要 200 + `ok: false`，方便监控判断；
2. **配置错误要在启动或首个请求暴露**，不能等用户提问才炸出一句 401；
3. **降级路径必须有测试**：凡是「优雅降级」的分支，都要有一个离线用例证明它真的降级了。

## 7. 兼容策略

- **配置向后兼容**：新增配置项必须有默认值；改名要同时保留旧名并打 deprecation warning 一个版本。
- **API 向后兼容**：`/api/*` 响应只增字段、不改含义；破坏性变更走 `/api/v2`。
- **向量空间兼容**：`identity` 变更必须显式重建索引（有校验拦着）；分块参数（`CHUNK_SIZE` /
  `CHUNK_OVERLAP`）变更建议重建，至少在评测报告里注明。
- **缓存兼容**：结构变更要 bump 命名空间版本，避免旧值被新代码误读。
- **数据兼容**：迁移必须能在旧数据上跑通（G1 的 `downgrade base` 是最低要求）。
