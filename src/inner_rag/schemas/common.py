"""通用响应 Schema。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ResponseModel(BaseModel):
    code: int = 200
    message: str = "success"
    data: Any | None = None


class PageData(BaseModel):
    total: int
    items: list[Any]
    page: int
    page_size: int
