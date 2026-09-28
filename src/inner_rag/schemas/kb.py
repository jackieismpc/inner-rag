"""知识库 Pydantic Schema。"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class KBStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"


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
    name: str
    description: str | None
    icon: str
    status: str
    embedding_model: str
    doc_count: int
    created_at: datetime
    updated_at: datetime
