"""知识库 Pydantic Schema。"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class KBStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"


class KBMemberPermission(str, Enum):
    """成员权限（与 ``models.kb_member.KBPermission`` 一致，用于出入参）。"""

    READ = "read"
    WRITE = "write"


class KBCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: str | None = None
    icon: str | None = "📚"
    # 留空则由服务端按当前 embedding 配置写入，保证可追溯
    embedding_model: str | None = None


class KBUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    icon: str | None = None
    status: KBStatus | None = None


class KBOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    owner_id: int
    name: str
    description: str | None
    icon: str
    status: str
    embedding_model: str
    doc_count: int
    created_at: datetime
    updated_at: datetime
    # 当前请求者对库的权限（owner/write/read）：前端据此渲染可做的操作，
    # 不靠 403 事后兜底。只能由路由层显式填充（模型属性里没有这个字段）。
    my_permission: str | None = None


class KBMemberAdd(BaseModel):
    """按用户名授权：用户名不存在会 404，避免静默建出一个空授权。"""

    username: str = Field(..., min_length=1, max_length=64)
    permission: KBMemberPermission


class KBMemberOut(BaseModel):
    user_id: int
    username: str
    display_name: str
    permission: KBMemberPermission
