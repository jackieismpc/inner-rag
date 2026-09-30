"""应用配置：所有可调参数集中在此，通过环境变量或 .env 覆盖。"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ── Application ────────────────────────────────────────────────────
    APP_NAME: str = "inner-rag Knowledge Base"
    APP_VERSION: str = "0.3.0"
    # 写进每条日志与每个 span 的 metadata，用于「按环境 / 按版本对比指标」
    APP_ENV: str = "dev"  # dev | staging | prod
    DEBUG: bool = True
    # /docs /redoc /openapi.json：只在非生产环境开放，避免对外暴露完整接口清单
    ENABLE_DOCS: bool = True

    # ── Auth（本地账号 + JWT）───────────────────────────────────────────
    # 生产（DEBUG=false）必须改成 ≥32 字节的随机密钥，否则拒绝启动（见 core/security.py）；
    # 账号由 `uv run scripts/create_user.py` 创建，系统不提供注册接口。
    AUTH_SECRET_KEY: str = "dev-only-insecure-secret-change-me-in-production"
    # 会话 Token 有效期（分钟）。JWT 无状态、无法单独撤销，短 TTL 是唯一的收敛手段。
    AUTH_TOKEN_TTL_MINUTES: int = 720

    # ── Server ─────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8010

    # ── Database ───────────────────────────────────────────────────────
    # 开发默认 SQLite：零依赖、单文件，gitignore 在 data/ 下。
    # 部署时可整库换成 PostgreSQL（postgresql+psycopg://...），代码与迁移脚本对两者兼容。
    DATABASE_URL: str = "sqlite:///./data/inner_rag.db"
    # 仅 PostgreSQL 使用（SQLite 会忽略这些池参数）
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20
    # SQLite 等锁超时（秒）：后台解析入库与前台查询可能并发写入
    SQLITE_TIMEOUT: int = 30
    # 表结构由 Alembic 管理；仅测试或一次性临时库才开启 create_all
    AUTO_CREATE_TABLES: bool = False
    SQL_ECHO: bool = False

    # ── LLM / Embeddings：provider 选择 ─────────────────────────────────
    # API 优先：开发时直接调云端 API，不依赖本地 Ollama；需要本地/离线时可切 mock。
    #   LLM_PROVIDER       = ollama | openrouter | deepseek | openai | mock
    #   EMBEDDING_PROVIDER = ollama | openrouter | openai | mock（DeepSeek 没有 embedding API）
    LLM_PROVIDER: str = "ollama"
    EMBEDDING_PROVIDER: str = "ollama"

    # Ollama（本地，无需密钥）
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_LLM_MODEL: str = "qwen3:14b"
    OLLAMA_EMBEDDING_MODEL: str = "qwen3-embedding:8b"

    # OpenRouter（OpenAI 兼容的云端聚合网关）
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_CHAT_MODEL: str = "deepseek/deepseek-v3.2"
    # 免费路由：1024 维、上下文只有 512 token，建议配合 EMBEDDING_MAX_INPUT_CHARS
    OPENROUTER_EMBEDDING_MODEL: str = "liquid/lfm-2.5-embedding-350m:free"

    # DeepSeek（只有 chat completion，不支持 embedding）
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"
    DEEPSEEK_API_KEY: str = ""
    # 模型名以官方文档为准（https://api-docs.deepseek.com）：deepseek-flash 是当前主推，
    # deepseek-v4-pro 能力更强；旧别名 deepseek-chat 仍可调用，但已不在 /models 列表里。
    DEEPSEEK_CHAT_MODEL: str = "deepseek-flash"

    # OpenAI（也可指向任何 OpenAI 兼容的自建网关）
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_API_KEY: str = ""
    OPENAI_CHAT_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Mock（离线演示与测试：不联网、不需要密钥，但只有词面相似度）
    MOCK_CHAT_MODEL: str = "mock-chat"
    MOCK_EMBEDDING_MODEL: str = "mock-embedding"
    MOCK_EMBEDDING_DIM: int = 256

    # 云端调用参数（Ollama 同样复用 temperature / max_tokens）
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 2048
    LLM_TIMEOUT: float = 60.0
    LLM_MAX_RETRIES: int = 2
    # 推理强度（DeepSeek 专用）：minimal | low | medium | high，留空则不发送该字段
    LLM_REASONING_EFFORT: str = ""
    # embedding 单条输入的字符上限（0 = 不截断）。分块按字符切、模型按 token 限，
    # 对上下文很小的模型（如 512 token 的免费 embedding）显式截断可避免上游报错。
    EMBEDDING_MAX_INPUT_CHARS: int = 0

    # ── Vector store ───────────────────────────────────────────────────
    # 后端选择：zvec（默认，内嵌单进程）| chroma（旧数据兼容）
    VECTOR_STORE: str = "zvec"
    ZVEC_PATH: str = "./data/zvec_db"
    # 仅当 VECTOR_STORE=chroma 时生效
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    CHROMA_COLLECTION_NAME: str = "rag_documents"

    # ── File upload ────────────────────────────────────────────────────
    UPLOAD_DIR: str = "./data/uploads"
    MAX_FILE_SIZE: int = 104857600  # 100 MB
    ALLOWED_EXTENSIONS: str = (
        "pdf,doc,docx,txt,md,csv,json,xml,html,htm,xls,xlsx,jpg,jpeg,png,gif,bmp,tiff,webp"
    )
    # 服务端本地路径导入：默认关闭（属于文件系统访问能力），需要时显式开启
    ALLOW_LOCAL_IMPORT: bool = False
    LOCAL_IMPORT_ROOT: str = ""  # 非空时，导入路径必须位于该目录之内

    # ── OCR ────────────────────────────────────────────────────────────
    OCR_BACKEND: str = "none"  # none | paddle
    OCR_LANG: str = "ch"
    OCR_RENDER_DPI: int = 200

    # ── RAG ────────────────────────────────────────────────────────────
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K: int = 8  # 向量召回数量
    RERANK_TOP_K: int = 5  # 进入 context 的条数
    RETRIEVAL_SCORE_THRESHOLD: float = 0.3  # cosine 相似度阈值
    MAX_CONTEXT_LENGTH: int = 6000  # context 最大字符数
    HISTORY_MAX_MESSAGES: int = 20  # 送入模型的历史消息条数（最近 N 条）

    # ── Cache ──────────────────────────────────────────────────────────
    # 后端选择：memory（默认，进程内 LRU）| 第三方实现（注册 entry point 后填它的名字）
    CACHE_BACKEND: str = "memory"
    QUERY_CACHE_TTL: int = 300
    QUERY_CACHE_MAX_SIZE: int = 500
    EMBEDDING_CACHE_MAX_SIZE: int = 2000

    # ── Async worker ───────────────────────────────────────────────────
    EMBEDDING_CONCURRENCY: int = 3
    EMBED_BATCH_SIZE: int = 20
    # 后台任务队列：inprocess（默认，进程内 asyncio worker）| inline（同步执行）
    # | 第三方实现（注册 entry point 后填它的名字）
    TASK_QUEUE_BACKEND: str = "inprocess"
    # 同时处理的入库任务数上限：免费 embedding 额度经不起并发冲击
    TASK_QUEUE_CONCURRENCY: int = 2
    # 单任务失败后的重试次数（入库失败多为上游抖动，重试前先退避）
    TASK_QUEUE_MAX_RETRIES: int = 1
    TASK_QUEUE_RETRY_BACKOFF: float = 5.0
    # 队列状态里保留的最近任务记录条数（有界，避免长跑进程里状态表无限增长）
    TASK_QUEUE_HISTORY: int = 200

    # ── CORS ───────────────────────────────────────────────────────────
    CORS_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"

    # ── Logging ────────────────────────────────────────────────────────
    LOG_DIR: str = "./logs"
    LOG_RETRIEVAL: bool = True
    LOG_PROMPT: bool = True
    LOG_LEVEL: str = "INFO"
    # text（人读）| json（一行一个 JSON，供采集端解析）
    LOG_FORMAT: str = "text"
    # 成功请求的 trace 采样率；失败请求恒为 100%（失败样本才是排障依据）
    LOG_SAMPLE_RATE: float = 1.0

    # ── Tracing（LangSmith）────────────────────────────────────────────
    # 默认关闭：测试与 CI 必须零网络、零费用。关掉时 span 只落本地计时日志，
    # 不会因为没接 SaaS 就丢掉耗时归因（降级契约见 docs/architecture.md 第 6 节）。
    LANGSMITH_TRACING: bool = False
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "inner-rag"
    # 留空则用官方 SaaS；自建 / 代理部署时填完整 URL
    LANGSMITH_ENDPOINT: str = ""
    LANGSMITH_WORKSPACE_ID: str = ""

    # ── Metrics ────────────────────────────────────────────────────────
    # none：`/api/system/metrics` 返回 JSON；prometheus：返回 Prometheus 文本
    METRICS_BACKEND: str = "none"
    # 非空时 `/api/system/metrics` 需要 `X-Metrics-Token`（与登录态解耦，便于抓取器直连）
    METRICS_TOKEN: str = ""

    @property
    def allowed_extensions_list(self) -> list[str]:
        return [e.strip().lower() for e in self.ALLOWED_EXTENSIONS.split(",") if e.strip()]

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def embedding_key(self) -> str:
        """当前 embedding 的身份标识（``provider:model``）。

        写入知识库并用于向量空间一致性校验：换了 provider 或模型后，旧知识库
        会被校验拦下，而不是静默地拿两套向量空间互相检索。

        延迟 import 避免 ``core.config`` <-> ``providers.specs`` 循环依赖；
        provider 配置非法时抛 ProviderError，由 API 层映射成 503。
        """
        from inner_rag.providers.specs import embedding_spec

        return embedding_spec().identity

    @property
    def sqlite_file_path(self) -> Path | None:
        """SQLite 文件路径（内存库或非 SQLite 返回 None）。"""
        prefix = "sqlite:///"
        if not self.DATABASE_URL.startswith(prefix):
            return None
        raw = self.DATABASE_URL[len(prefix) :]
        if not raw or raw.startswith(":memory:"):
            return None
        return Path(raw)

    def ensure_dirs(self) -> None:
        for directory in (self.UPLOAD_DIR, self.ZVEC_PATH, self.CHROMA_PERSIST_DIR, self.LOG_DIR):
            Path(directory).mkdir(parents=True, exist_ok=True)
        # SQLite 不会自己建目录，缺少父目录时会报 unable to open database file
        sqlite_path = self.sqlite_file_path
        if sqlite_path is not None:
            sqlite_path.parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
