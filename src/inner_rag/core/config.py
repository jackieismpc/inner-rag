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
    APP_VERSION: str = "0.2.0"
    DEBUG: bool = True
    SECRET_KEY: str = "change-me-in-production"

    # ── Server ─────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8010

    # ── Database ───────────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+psycopg://rag:rag@localhost:5432/rag_db"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20
    # 表结构由 Alembic 管理；仅测试或一次性临时库才开启 create_all
    AUTO_CREATE_TABLES: bool = False
    SQL_ECHO: bool = False

    # ── LLM / Embeddings ───────────────────────────────────────────────
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_LLM_MODEL: str = "qwen3:14b"
    OLLAMA_EMBEDDING_MODEL: str = "qwen3-embedding:8b"

    # ── Vector store ───────────────────────────────────────────────────
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
    QUERY_CACHE_TTL: int = 300
    QUERY_CACHE_MAX_SIZE: int = 500
    EMBEDDING_CACHE_MAX_SIZE: int = 2000

    # ── Async worker ───────────────────────────────────────────────────
    EMBEDDING_CONCURRENCY: int = 3
    EMBED_BATCH_SIZE: int = 20

    # ── CORS ───────────────────────────────────────────────────────────
    CORS_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"

    # ── Logging ────────────────────────────────────────────────────────
    LOG_DIR: str = "./logs"
    LOG_RETRIEVAL: bool = True
    LOG_PROMPT: bool = True
    LOG_LEVEL: str = "INFO"

    @property
    def allowed_extensions_list(self) -> list[str]:
        return [e.strip().lower() for e in self.ALLOWED_EXTENSIONS.split(",") if e.strip()]

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def embedding_key(self) -> str:
        """当前 embedding 的身份标识，写入知识库并用于向量空间一致性校验。"""
        return f"ollama:{self.OLLAMA_EMBEDDING_MODEL}"

    def ensure_dirs(self) -> None:
        for directory in (self.UPLOAD_DIR, self.CHROMA_PERSIST_DIR, self.LOG_DIR):
            Path(directory).mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
