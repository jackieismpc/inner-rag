"""运行日志：loguru sink 配置（text / json）与 request_id / 访问日志中间件。

三条设计取舍：

1. **request_id 与 user 走 sink 的 format，不用 `logger.patch`**：各模块各自
   `from loguru import logger`，patch 只作用于被 patch 的那一个实例；sink 的 format 对写入
   该 sink 的**全部**记录生效，一次配置就能让全链路日志带上上下文（见 `core/context.py`）。
2. **request_id 的生成与访问日志放在同一个 ASGI 中间件**：两者都要包一层 `send`
   （一个要回写响应头、一个要读状态码）。拆成两个中间件会让每个响应多一层包装，
   而纯 ASGI 中间件是唯一能在同步端点里也拿到上下文的位置。
3. **结构化字段靠 `logger.bind(event=..., **fields)`**：text 模式下人读的仍是原消息，
   json 模式下绑定的字段被序列化成顶层键 —— 一处埋点，两种格式都成立。
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from collections.abc import Iterable
from typing import Any

from loguru import logger

from inner_rag.core.config import settings
from inner_rag.core.context import bind_request_id, current_request_id, log_user
from inner_rag.core.metrics import metrics

TEXT_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | "
    "rid={extra[request_id]} | user={extra[user]} | {message}\n"
)

# 已经作为顶层键写入 JSON 的字段，不再重复展开
# （`payload` 是 json 模式的中间产物，见 `json_line`）
_RESERVED_FIELDS = frozenset({"request_id", "user", "event", "payload"})


def text_line(record: Any) -> str:
    """text 格式：把上下文塞进 extra，再交给固定模板。"""
    record["extra"]["request_id"] = current_request_id() or "-"
    record["extra"]["user"] = log_user()
    return TEXT_FORMAT


def json_line(record: Any) -> str:
    """json 格式：一行一个 JSON，便于采集端直接 `json.loads`（DoD 要求可解析）。

    为什么返回 `{extra[payload]}` 模板而不是直接返回 JSON 串：loguru 会把 format 的
    返回值**再当模板格式化一次**，裸 JSON 里的 `{` `}` 会被当成占位符并抛 KeyError。
    """
    extra = record["extra"]
    payload: dict[str, Any] = {
        "ts": record["time"].isoformat(),
        "level": record["level"].name,
        "event": extra.get("event", "log"),
        "message": record["message"],
        "request_id": current_request_id() or "-",
        "user": log_user(),
    }
    payload.update({key: value for key, value in extra.items() if key not in _RESERVED_FIELDS})
    if record["exception"] is not None:
        payload["error"] = f"{record['exception'].type.__name__}: {record['exception'].value}"
    extra["payload"] = json.dumps(payload, ensure_ascii=False, default=str)
    return "{extra[payload]}\n"


def _formatter() -> Any:
    return json_line if settings.LOG_FORMAT == "json" else text_line


def configure_logging() -> None:
    """配置 loguru sink（启动时调用一次；重复调用也只会留一份 sink）。"""
    logger.remove()
    logger.add(sys.stdout, level=settings.LOG_LEVEL, format=_formatter())
    logger.add(
        f"{settings.LOG_DIR}/app.log",
        rotation="10 MB",
        retention="7 days",
        level=settings.LOG_LEVEL,
        encoding="utf-8",
        format=_formatter(),
    )


def _request_id_from_headers(headers: Iterable[tuple[bytes, bytes]]) -> str | None:
    """透传上游（网关 / 前端）给的 request_id，便于跨系统串联。"""
    for name, value in headers:
        if name.lower() == b"x-request-id":
            given = value.decode("latin-1").strip()
            if given:
                return given
    return None


class RequestIdMiddleware:
    """生成 / 透传 request_id，回写响应头，并在响应开始时发一行访问日志。

    request_id 是「日志 ↔ trace ↔ 工单」的唯一钥匙：`core/observability.py` 把它写进每个
    span 的 metadata，`docs/observability.md` 的排障入口就是拿它去串日志。
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _request_id_from_headers(scope.get("headers") or []) or uuid.uuid4().hex
        reset = bind_request_id(request_id)
        started = time.perf_counter()
        client_host = (scope.get("client") or ("-",))[0]
        try:
            await self.app(
                scope, receive, self._wrapped_send(scope, send, request_id, started, client_host)
            )
        finally:
            reset()

    def _wrapped_send(
        self, scope: dict, send: Any, request_id: str, started: float, client_host: str
    ) -> Any:
        async def send_wrapper(message: dict) -> None:
            if message["type"] == "http.response.start":
                status = message["status"]
                # 状态码与用户都在这里读：此刻仍在 IdentityContextMiddleware 的作用域内，
                # 等中间件 finally 复位后再读就只剩 "-" 了。
                self._log_access(scope, request_id, status, started, client_host)
                headers = message.setdefault("headers", [])
                headers.append((b"x-request-id", request_id.encode("latin-1")))
            await send(message)

        return send_wrapper

    def _log_access(
        self, scope: dict, request_id: str, status: int, started: float, client_host: str
    ) -> None:
        route = scope.get("route")
        path = getattr(route, "path", None) or scope.get("path", "")
        endpoint = f"{scope.get('method', '')} {path}"
        duration_ms = (time.perf_counter() - started) * 1000

        logger.bind(
            event="http_access",
            method=scope.get("method", ""),
            path=path,
            status=status,
            duration_ms=round(duration_ms, 2),
            client=client_host,
        ).info(f"[HTTP] {endpoint} -> {status} ({duration_ms:.1f}ms)")

        metrics.increment(
            "rag_requests_total", labels={"endpoint": endpoint, "status": str(status)}
        )
        metrics.observe("rag_request_duration_ms", duration_ms, labels={"endpoint": endpoint})
