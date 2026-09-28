"""请求上下文：把「当前请求是谁」放进 contextvar，供日志串联使用。

为什么是 ASGI 中间件而不是 FastAPI 依赖：同步依赖运行在线程池里，它设置的 contextvar
不会回写到请求上下文，端点读到的仍是空值；纯 ASGI 中间件处于请求任务本身，
下游无论是同步端点（anyio 线程池会传播上下文）还是异步端点都能读到。

本模块只做「标记」，不做「拒绝」：身份是否有效由 ``api/deps.py`` 判定（401）。
因此这里解不开 token 时静默置空——日志里显示 ``-``，而不是改变鉴权语义。
"""

from __future__ import annotations

from collections.abc import Iterable
from contextvars import ContextVar
from typing import Any

from inner_rag.core.security import InvalidToken, decode_access_token

_current_user_id: ContextVar[int | None] = ContextVar("current_user_id", default=None)


def current_user_id() -> int | None:
    """当前请求的 user_id；未登录或不在请求内时为 None。"""
    return _current_user_id.get()


def log_user() -> str:
    """日志里显示的用户标记（未登录为 ``-``）。"""
    user_id = _current_user_id.get()
    return "-" if user_id is None else str(user_id)


def _user_id_from_headers(headers: Iterable[tuple[bytes, bytes]]) -> int | None:
    for name, value in headers:
        if name.lower() != b"authorization":
            continue
        scheme, _, credentials = value.decode("latin-1").partition(" ")
        if scheme.lower() != "bearer" or not credentials:
            return None
        try:
            return decode_access_token(credentials)
        except InvalidToken:
            return None
    return None


class IdentityContextMiddleware:
    """把 Authorization 里的 user_id 写入上下文，供日志使用。"""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        token = _current_user_id.set(_user_id_from_headers(scope.get("headers") or []))
        try:
            await self.app(scope, receive, send)
        finally:
            _current_user_id.reset(token)
