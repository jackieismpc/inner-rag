"""数据库引擎、会话工厂与表结构初始化。"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from loguru import logger
from sqlalchemy import Engine, create_engine, event, inspect
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import QueuePool

from inner_rag.core.config import settings


class Base(DeclarativeBase):
    """所有 ORM 模型的基类（SQLAlchemy 2.0 声明式风格）。"""


def _create_engine() -> Engine:
    url = settings.DATABASE_URL
    if url.startswith("sqlite"):
        # SQLite 不走连接池参数；timeout 即等锁超时，避免并发写入直接报 database is locked
        engine = create_engine(
            url,
            connect_args={
                "check_same_thread": False,
                "timeout": settings.SQLITE_TIMEOUT,
            },
            echo=settings.SQL_ECHO,
        )

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragmas(dbapi_connection: Any, _connection_record: Any) -> None:
            # WAL：读写并发（后台解析入库时仍能正常查询）
            # foreign_keys：SQLite 默认不强制外键，不开启则删除知识库不会级联清理
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute(f"PRAGMA busy_timeout={settings.SQLITE_TIMEOUT * 1000}")
            cursor.close()

        return engine
    return create_engine(
        url,
        poolclass=QueuePool,
        pool_size=settings.DATABASE_POOL_SIZE,
        max_overflow=settings.DATABASE_MAX_OVERFLOW,
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=settings.SQL_ECHO,
    )


engine = _create_engine()
# expire_on_commit=False：commit 之后仍可直接读取已加载属性，
# 避免在异步生成器等场景因会话已关闭而触发 DetachedInstanceError。
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False)


def get_db() -> Iterator[Session]:
    """FastAPI 依赖：请求级数据库会话。"""
    db = SessionLocal()
    try:
        yield db
    except Exception as exc:
        # HTTPException（400/403/404）也会走到这里，不应记成数据库错误
        db.rollback()
        logger.debug(f"请求会话回滚: {exc}")
        raise
    finally:
        db.close()


def init_db() -> None:
    """
    确认表结构就绪。

    生产/开发默认由 Alembic 负责建表（AUTO_CREATE_TABLES=false），
    缺少表时直接报错并给出修复命令，而不是静默地建出一份可能与迁移脚本不一致的结构。
    """
    from inner_rag.models import conversation, document, knowledge_base  # noqa: F401

    if settings.AUTO_CREATE_TABLES:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created via create_all (AUTO_CREATE_TABLES=true)")
        return

    missing = sorted(set(Base.metadata.tables) - set(inspect(engine).get_table_names()))
    if missing:
        msg = (
            f"数据库缺少表 {missing}；请先执行 `uv run alembic upgrade head`，"
            "或临时设置 AUTO_CREATE_TABLES=true"
        )
        raise RuntimeError(msg)
    logger.info("Database schema is up to date")
