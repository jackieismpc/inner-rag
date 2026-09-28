"""用户 ORM 模型（本地账号）。"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from inner_rag.core.database import Base


class User(Base):
    """本地账号：单租户下的「谁在用」。

    密码只存 argon2id 哈希（见 ``core/security.py``），任何地方都不保存明文。
    账号由运维脚本 ``scripts/create_user.py`` 创建，系统不提供注册接口。

    ``is_active`` 用于停用而不是删除：删除用户会连带丢掉他名下的知识库，
    停用只是禁止登录（已签发的 token 会在下一次请求时被拒）。
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, comment="登录名")
    display_name: Mapped[str] = mapped_column(String(100), default="", comment="显示名")
    password_hash: Mapped[str] = mapped_column(
        String(255), nullable=False, comment="argon2id 哈希，不可逆"
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, comment="停用后禁止登录")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self) -> str:
        return f"<User id={self.id} username={self.username!r} active={self.is_active}>"
