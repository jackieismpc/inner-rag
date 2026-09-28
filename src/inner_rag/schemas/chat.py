"""对话 Pydantic Schema。

引用来源（sources）沿用 JSON 存储，其元素形如：
    {index, filename, page, score, doc_id, content}
其中 score 为真实相关性分数；MMR 召回项没有分数，此时为 null。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    kb_id: int
    conv_id: int | None = None
    question: str = Field(..., min_length=1)
    stream: bool = False
    strategy: str | None = Field(
        default=None, description="检索策略：similarity | mmr | hybrid，默认 hybrid"
    )


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    conv_id: int
    role: str
    content: str
    sources: list[dict[str, Any]] | None = None
    created_at: datetime


class ConversationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    kb_id: int
    title: str
    created_at: datetime
    updated_at: datetime
