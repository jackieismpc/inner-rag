"""追踪门面（Tracer）：把一次请求的关键步骤记成可回放的 span 树。

设计取舍：

- **LangSmith 只是可选的 sink**：默认关闭（测试与 CI 零网络、零费用）。关掉时 span 照样计时
  并落到本地日志——「耗时归因」是排障刚需，不能因为没接 SaaS 就消失（降级契约见
  `docs/architecture.md` 第 6 节）。
- **上报失败只计数、绝不抛出**：追踪是旁路，让它影响请求成功率等于把观测系统变成故障源。
  失败计入 `tracing_errors_total`，排障看这个计数而不是看日志噪音。
- **span 树用 contextvar 维护**：子 span 自动挂到最近的父 span，调用方只写
  `async with tracer.span("vector.search", kb_id=kb_id)`，不用手动传 parent。
- **只有根 span 做采样**：一个请求的 trace 要么完整要么没有，半棵树没法排障；
  失败请求（span 带 error）不受采样率限制——失败样本才是排障依据。
- **trace 只记元数据与统计**：不记 Prompt 与回答正文，避免把用户内容送到第三方
  （正文日志仍由既有的 `LOG_PROMPT` 单独控制）。
"""

from __future__ import annotations

import os
import random
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from inner_rag.core.config import settings
from inner_rag.core.context import current_request_id, current_user_id
from inner_rag.core.metrics import metrics

_current_span: ContextVar[Span | None] = ContextVar("current_span", default=None)


def base_metadata() -> dict[str, Any]:
    """每个 span 都带的上下文。缺值就不带该键（见 `docs/observability.md` 2.2）。"""
    payload: dict[str, Any] = {"app_version": settings.APP_VERSION, "env": settings.APP_ENV}
    request_id = current_request_id()
    if request_id is not None:
        payload["request_id"] = request_id
    user_id = current_user_id()
    if user_id is not None:
        payload["user_id"] = user_id
    return payload


def base_tags() -> list[str]:
    return [f"env:{settings.APP_ENV}", f"version:{settings.APP_VERSION}"]


@dataclass
class Span:
    """一个步骤的计时与元数据容器。

    `run` 是 LangSmith 的 RunTree：未启用追踪（或未被采样）时为 None，
    其余行为（计时、本地日志）不受影响。
    """

    name: str
    parent: Span | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    duration_ms: float = 0.0
    run: Any = None
    sampled: bool = True

    def set(self, **fields: Any) -> None:
        """补记运行期才知道的字段（命中数、token 用量、是否被采样）。"""
        self.metadata.update(fields)

    @property
    def trace_id(self) -> str | None:
        """LangSmith 的 run id（整棵树的根 id）；未启用追踪时为 None。

        评测要用它把「逐题分数」挂回对应 trace：没有这个 id，feedback 只能堆在
        实验维度上，无法下钻到某次具体调用（见 docs/evaluation.md 6.3）。
        """
        if self.run is None:
            return None
        return str(getattr(self.run, "id", "") or "") or None


class Tracer:
    """追踪门面：对外只有 `span()` 一个入口，后端（LangSmith / 本地日志）由配置决定。"""

    def __init__(self) -> None:
        self.enabled = False
        self.reason = "未初始化"
        self._client: Any = None

    # ── 初始化 ────────────────────────────────────────────────────────

    def configure(self) -> None:
        """按配置初始化（幂等，lifespan 里调一次）。任何问题都降级为本地计时，不阻断启动。"""
        self.enabled = False
        self._client = None

        if not settings.LANGSMITH_TRACING:
            self.reason = "LANGSMITH_TRACING=false（默认关闭）"
        elif not settings.LANGSMITH_API_KEY:
            self.reason = "LANGSMITH_TRACING=true 但 LANGSMITH_API_KEY 为空"
        else:
            try:
                from langsmith import Client

                self._client = Client(
                    api_url=settings.LANGSMITH_ENDPOINT or None,
                    api_key=settings.LANGSMITH_API_KEY,
                    workspace_id=settings.LANGSMITH_WORKSPACE_ID or None,
                )
            except Exception as exc:  # 降级契约要求：初始化失败不能拖垮启动
                self.reason = f"LangSmith 客户端初始化失败: {exc}"
            else:
                self.enabled = True
                self.reason = f"已启用，项目 {settings.LANGSMITH_PROJECT}"

        if self.enabled:
            self._export_env()
            logger.info(f"[TRACE] LangSmith 追踪已启用: {self.reason}")
        else:
            logger.warning(f"[TRACE] 追踪未启用，span 只落本地计时日志: {self.reason}")

    def _export_env(self) -> None:
        """把开关与端点写回环境变量。

        LangChain 的自动追踪由 langsmith SDK 自己读环境变量，而本项目的配置只从
        `core/config.py` 读（不隐式依赖进程环境），因此这里显式桥接一次——
        这样模型调用本身产生的 run 也会挂在我们建的 span 树下。
        """
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT
        os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
        if settings.LANGSMITH_ENDPOINT:
            os.environ["LANGSMITH_ENDPOINT"] = settings.LANGSMITH_ENDPOINT
        if settings.LANGSMITH_WORKSPACE_ID:
            os.environ["LANGSMITH_WORKSPACE_ID"] = settings.LANGSMITH_WORKSPACE_ID

    def status(self) -> dict[str, Any]:
        """追踪状态（启动日志与排障用：trace 看不到时第一件事就是看这一行）。"""
        return {
            "enabled": self.enabled,
            "backend": "langsmith" if self.enabled else "local",
            "project": settings.LANGSMITH_PROJECT if self.enabled else None,
            "reason": self.reason,
        }

    # ── span ──────────────────────────────────────────────────────────

    @asynccontextmanager
    async def span(
        self, name: str, run_type: str = "chain", **metadata: Any
    ) -> AsyncIterator[Span]:
        """开一个 span（只能 `async with`）。

        为什么只提供异步版本：埋点全在 async 链路上，给两套协议只会让调用方纠结用哪个；
        计时在 span 内部完成，与 with 块里有没有 await 无关。
        """
        parent = _current_span.get()
        span = Span(name=name, parent=parent, metadata=base_metadata() | metadata)
        # 采样只在根 span 判定：半棵树的 trace 没法排障；父 span 没建 run 时子 span 跟着不建
        span.sampled = should_sample() if parent is None else parent.run is not None
        if self.enabled and span.sampled:
            span.run = self._create_run(span, run_type)

        token = _current_span.set(span)
        started = time.perf_counter()
        try:
            yield span
        except Exception as exc:
            span.error = str(exc)
            raise
        finally:
            span.duration_ms = (time.perf_counter() - started) * 1000
            _current_span.reset(token)
            self._finish(span, run_type)

    def _create_run(self, span: Span, run_type: str) -> Any:
        from langsmith.run_trees import RunTree

        common: dict[str, Any] = {
            "name": span.name,
            "run_type": run_type,
            "tags": base_tags(),
            "extra": {"metadata": dict(span.metadata)},
        }
        parent_run = span.parent.run if span.parent is not None else None
        if parent_run is not None:
            return parent_run.create_child(**common)
        return RunTree(
            project_name=settings.LANGSMITH_PROJECT,
            ls_client=self._client,
            **common,
        )

    def _finish(self, span: Span, run_type: str) -> None:
        """结束 span：先补记失败样本，再上报，最后落本地日志（上报失败不影响最后一步）。"""
        if span.run is None and span.error is not None and self.enabled and span.parent is None:
            # 未被采样但请求失败了：失败样本 100% 记录，这里补一棵只有根节点的树
            # （只补根节点——异常会逐层往上抛，父 span 自己也会走到这里，补子节点会重复建树）
            span.run = self._create_run(span, run_type)
        if span.run is not None:
            span.run.end(error=span.error, metadata=span.metadata)
            try:
                self._upload(span.run)
            except Exception as exc:
                metrics.increment("tracing_errors_total")
                logger.warning(f"[TRACE] 上报失败 span={span.name}: {exc}")
        self._log(span)

    def _upload(self, run: Any) -> None:
        run.post()
        run.patch()

    @property
    def client(self) -> Any:
        """LangSmith 客户端（未启用时为 None）；评测用它同步 dataset 与回写分数。"""
        return self._client if self.enabled else None

    def current_trace_id(self) -> str | None:
        """当前调用链的根 run id（供评测回写 feedback）；未启用追踪时为 None。"""
        span = _current_span.get()
        if span is None:
            return None
        root = span
        while root.parent is not None:
            root = root.parent
        return root.trace_id

    def _log(self, span: Span) -> None:
        """本地计时日志：LangSmith 开着时它是 DEBUG 冗余，关着时它就是唯一的耗时来源。"""
        level = "DEBUG" if self.enabled else "INFO"
        logger.bind(
            event="span",
            span=span.name,
            duration_ms=round(span.duration_ms, 2),
            **span.metadata,
        ).log(level, f"[TRACE] {span.name} | {span.duration_ms:.1f}ms")


def should_sample() -> bool:
    """成功请求的采样判定；失败请求在 `_finish` 里补记，不走这里。

    采样用的是普通随机数：它只决定「要不要留一份 trace」，不涉及安全，
    用 `secrets` 只会白白增加开销。
    """
    rate = settings.LOG_SAMPLE_RATE
    return rate >= 1.0 or random.random() < rate


tracer = Tracer()
