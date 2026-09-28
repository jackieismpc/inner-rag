"""文档 Pydantic Schema。"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class DocStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    kb_id: int
    filename: str
    file_type: str | None
    file_size: int
    status: str
    error_msg: str | None
    chunk_count: int
    char_count: int
    source_type: str
    created_at: datetime


class LocalPathImport(BaseModel):
    kb_id: int
    path: str = Field(..., description="服务器本地文件或目录路径")
    recursive: bool = Field(default=True, description="是否递归扫描子目录")
