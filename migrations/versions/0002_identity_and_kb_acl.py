"""identity and kb-level ACL: users / kb_members / knowledge_bases.owner_id

历史数据归属：``knowledge_bases.owner_id`` 是 NOT NULL，已有行的归属只能是某个用户。
迁移在「表里已有数据」时创建一个不可登录的 bootstrap 账号（``admin``，密码哈希为哨兵值
``!``，任何密码都验不过），把历史知识库全部挂到它名下；运维随后执行
``uv run scripts/create_user.py admin --reset-password`` 设置真实密码即可接管这些库。
空库（全新安装）不创建该账号，避免留下一个用不上的默认用户。

Revision ID: 0002_identity_kb_acl
Revises: 0001_initial
Create Date: 2026-09-28
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_identity_kb_acl"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

BOOTSTRAP_USERNAME = "admin"
# 不可登录的哨兵：不是合法 argon2 哈希，verify_password() 对它一律返回 False
BOOTSTRAP_PASSWORD_HASH = "!"

# 轻量表定义：迁移里不需要 ORM，但需要能在 SQLite 与 PostgreSQL 上通用地插入/回查。
# 不标 primary_key：插入时不提供 id，由数据库自增（标成主键反而会触发「主键无默认值」告警）。
_users = sa.table(
    "users",
    sa.Column("id", sa.Integer),
    sa.Column("username", sa.String),
    sa.Column("display_name", sa.String),
    sa.Column("password_hash", sa.String),
    sa.Column("is_active", sa.Boolean),
)


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("username", sa.String(length=64), nullable=False, comment="登录名"),
        sa.Column("display_name", sa.String(length=100), nullable=False, comment="显示名"),
        sa.Column(
            "password_hash", sa.String(length=255), nullable=False, comment="argon2id 哈希，不可逆"
        ),
        sa.Column("is_active", sa.Boolean(), nullable=False, comment="停用后禁止登录"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("username", name="uq_users_username"),
    )

    op.create_table(
        "kb_members",
        sa.Column("kb_id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column(
            "permission",
            sa.Enum("read", "write", name="kbpermission", native_enum=False, length=20),
            nullable=False,
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(["kb_id"], ["knowledge_bases.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("kb_id", "user_id"),
    )
    op.create_index("ix_kb_members_user_id", "kb_members", ["user_id"])

    conn = op.get_bind()
    # 先可空加列：SQLite 不允许对已有行直接加 NOT NULL 列，回填后再收紧
    op.add_column("knowledge_bases", sa.Column("owner_id", sa.Integer(), nullable=True))

    has_existing_kbs = (
        conn.execute(sa.text("SELECT COUNT(*) FROM knowledge_bases")).scalar_one() > 0
    )
    owner_id: int | None = None
    if has_existing_kbs:
        conn.execute(
            sa.insert(_users).values(
                username=BOOTSTRAP_USERNAME,
                display_name="历史数据接管账号",
                password_hash=BOOTSTRAP_PASSWORD_HASH,
                is_active=True,
            )
        )
        owner_id = conn.execute(
            sa.text("SELECT id FROM users WHERE username = :name"), {"name": BOOTSTRAP_USERNAME}
        ).scalar_one()
        conn.execute(
            sa.text("UPDATE knowledge_bases SET owner_id = :owner"),
            {"owner": owner_id},
        )

    # 回填完成后收紧为 NOT NULL：所有行此时都有 owner
    # SQLite 不支持 ALTER COLUMN / 加外键，batch 模式会重建表；PostgreSQL 走同一份代码
    with op.batch_alter_table("knowledge_bases") as batch:
        batch.alter_column("owner_id", existing_type=sa.Integer(), nullable=False)
        batch.create_foreign_key(
            "fk_knowledge_bases_owner_id_users",
            "users",
            ["owner_id"],
            ["id"],
            ondelete="RESTRICT",
        )
        batch.create_index("ix_knowledge_bases_owner_id", ["owner_id"])


def downgrade() -> None:
    with op.batch_alter_table("knowledge_bases") as batch:
        batch.drop_index("ix_knowledge_bases_owner_id")
        batch.drop_constraint("fk_knowledge_bases_owner_id_users", type_="foreignkey")
        batch.drop_column("owner_id")

    op.drop_index("ix_kb_members_user_id", table_name="kb_members")
    op.drop_table("kb_members")
    op.drop_table("users")
