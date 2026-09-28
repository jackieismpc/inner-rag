"""认证相关 Schema。"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1)


class UserOut(BaseModel):
    """对外的用户信息：永远不含 password_hash。"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    display_name: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: datetime
    user: UserOut
