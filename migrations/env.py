"""Alembic 迁移环境。

DATABASE_URL 统一来自应用配置（settings），避免 alembic.ini 与 .env 各写一份出现漂移。
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# 必须导入模型模块，Base.metadata 才能感知到全部表
import inner_rag.models  # noqa: F401
from inner_rag.core.config import settings
from inner_rag.core.database import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# % 需要转义，否则 configparser 会把 DSN 里的 % 当成插值语法
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL.replace("%", "%%"))

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """离线模式：只生成 SQL，不连接数据库。"""
    context.configure(
        url=settings.DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
