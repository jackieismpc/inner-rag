"""知识库 ORM 模型。"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from inner_rag.core.database import Base

if TYPE_CHECKING:  # 仅用于类型检查，避免模型模块之间循环导入
    from inner_rag.models.conversation import Conversation
    from inner_rag.models.document import Document


class KBStatus(str, enum.Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"


def _enum_values(enum_cls: type[enum.Enum]) -> list[str]:
    """让数据库存枚举的 value（active）而不是 name（ACTIVE）。"""
    return [member.value for member in enum_cls]


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False, comment="知识库名称")
    description: Mapped[str | None] = mapped_column(Text, comment="知识库描述")
    icon: Mapped[str] = mapped_column(String(16), default="📚", comment="图标 emoji")
    status: Mapped[KBStatus] = mapped_column(
        SAEnum(KBStatus, native_enum=False, length=20, values_callable=_enum_values),
        default=KBStatus.ACTIVE,
    )
    # 建库时锁定的 embedding 标识：写入/检索时校验，避免中途换模型导致向量空间不一致
    embedding_model: Mapped[str] = mapped_column(String(120), default="ollama:qwen3-embedding:8b")
    doc_count: Mapped[int] = mapped_column(Integer, default=0, comment="已完成文档数")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    documents: Mapped[list[Document]] = relationship(
        back_populates="knowledge_base", cascade="all, delete-orphan"
    )
    conversations: Mapped[list[Conversation]] = relationship(
        back_populates="knowledge_base", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<KnowledgeBase id={self.id} name={self.name!r}>"
