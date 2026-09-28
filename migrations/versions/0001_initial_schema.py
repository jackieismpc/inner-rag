"""initial schema: knowledge_bases / documents / conversations / messages

Revision ID: 0001_initial
Revises:
Create Date: 2026-09-28
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "knowledge_bases",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False, comment="知识库名称"),
        sa.Column("description", sa.Text(), nullable=True, comment="知识库描述"),
        sa.Column("icon", sa.String(length=16), nullable=False, comment="图标 emoji"),
        sa.Column(
            "status",
            sa.Enum("active", "inactive", name="kbstatus", native_enum=False, length=20),
            nullable=False,
        ),
        sa.Column(
            "embedding_model",
            sa.String(length=120),
            nullable=False,
            comment="建库时锁定的 embedding 标识",
        ),
        sa.Column("doc_count", sa.Integer(), nullable=False, comment="已完成文档数"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "documents",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("kb_id", sa.Integer(), nullable=False),
        sa.Column("filename", sa.String(length=500), nullable=False, comment="原始文件名"),
        sa.Column("file_path", sa.String(length=1000), nullable=True, comment="服务端存储路径"),
        sa.Column("file_type", sa.String(length=50), nullable=True, comment="文件类型"),
        sa.Column("file_size", sa.BigInteger(), nullable=False, comment="文件大小(bytes)"),
        sa.Column(
            "status",
            sa.Enum(
                "pending",
                "processing",
                "completed",
                "failed",
                name="docstatus",
                native_enum=False,
                length=20,
            ),
            nullable=False,
        ),
        sa.Column("error_msg", sa.Text(), nullable=True, comment="错误信息"),
        sa.Column("chunk_count", sa.Integer(), nullable=False, comment="分块数量"),
        sa.Column("char_count", sa.Integer(), nullable=False, comment="字符数量"),
        sa.Column("meta_info", sa.JSON(), nullable=True, comment="解析元数据"),
        sa.Column("source_type", sa.String(length=20), nullable=False, comment="upload/local_path"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(["kb_id"], ["knowledge_bases.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_documents_kb_id", "documents", ["kb_id"])
    op.create_index("ix_documents_status", "documents", ["status"])

    op.create_table(
        "conversations",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("kb_id", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(["kb_id"], ["knowledge_bases.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_conversations_kb_id", "conversations", ["kb_id"])

    op.create_table(
        "messages",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("conv_id", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(length=20), nullable=False, comment="user/assistant"),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("sources", sa.JSON(), nullable=True, comment="引用来源"),
        sa.Column("tokens", sa.Integer(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(["conv_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_messages_conv_id", "messages", ["conv_id"])


def downgrade() -> None:
    op.drop_index("ix_messages_conv_id", table_name="messages")
    op.drop_table("messages")
    op.drop_index("ix_conversations_kb_id", table_name="conversations")
    op.drop_table("conversations")
    op.drop_index("ix_documents_status", table_name="documents")
    op.drop_index("ix_documents_kb_id", table_name="documents")
    op.drop_table("documents")
    op.drop_table("knowledge_bases")
