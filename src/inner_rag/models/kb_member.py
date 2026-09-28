"""知识库 ACL：被授权用户与权限级别。"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, func
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from inner_rag.core.database import Base
from inner_rag.models.knowledge_base import _enum_values

if TYPE_CHECKING:  # 仅类型检查，避免模型模块间循环导入
    from inner_rag.models.knowledge_base import KnowledgeBase


class KBPermission(str, enum.Enum):
    """成员权限：read 可看可问，write 额外可写文档。

    知识库的拥有者不存在这张表里——owner 是 ``knowledge_bases.owner_id``，
    拥有者权限（改设置 / 删库 / 管成员）不可被成员表覆盖。
    """

    READ = "read"
    WRITE = "write"


class KBMember(Base):
    """(kb_id, user_id) 复合主键：一个用户在一个库里只有一条权限记录。"""

    __tablename__ = "kb_members"

    kb_id: Mapped[int] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"), primary_key=True
    )
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True, index=True
    )
    permission: Mapped[KBPermission] = mapped_column(
        SAEnum(KBPermission, native_enum=False, length=20, values_callable=_enum_values),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    knowledge_base: Mapped[KnowledgeBase] = relationship()

    def __repr__(self) -> str:
        return f"<KBMember kb={self.kb_id} user={self.user_id} perm={self.permission}>"


__all__ = ["KBMember", "KBPermission"]
