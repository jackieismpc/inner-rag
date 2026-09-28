"""文档 ORM 模型。"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from inner_rag.core.database import Base
from inner_rag.models.knowledge_base import _enum_values

if TYPE_CHECKING:
    from inner_rag.models.knowledge_base import KnowledgeBase


class DocStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kb_id: Mapped[int] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False, index=True
    )
    filename: Mapped[str] = mapped_column(String(500), nullable=False, comment="原始文件名")
    file_path: Mapped[str | None] = mapped_column(String(1000), comment="服务端存储路径")
    file_type: Mapped[str | None] = mapped_column(String(50), comment="文件类型")
    file_size: Mapped[int] = mapped_column(BigInteger, default=0, comment="文件大小(bytes)")
    status: Mapped[DocStatus] = mapped_column(
        SAEnum(DocStatus, native_enum=False, length=20, values_callable=_enum_values),
        default=DocStatus.PENDING,
        index=True,
    )
    error_msg: Mapped[str | None] = mapped_column(Text, comment="错误信息")
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, comment="分块数量")
    char_count: Mapped[int] = mapped_column(Integer, default=0, comment="字符数量")
    meta_info: Mapped[dict | None] = mapped_column(JSON, comment="解析元数据")
    source_type: Mapped[str] = mapped_column(
        String(20), default="upload", comment="upload/local_path"
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    knowledge_base: Mapped[KnowledgeBase] = relationship(back_populates="documents")

    def __repr__(self) -> str:
        return f"<Document id={self.id} filename={self.filename!r} status={self.status}>"
