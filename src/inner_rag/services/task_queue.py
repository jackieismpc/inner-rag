"""后台任务队列：``TaskQueue`` 契约 + 两个内置实现（进程内 / 同步）。

为什么要抽这一层：入库是长任务（解析 → 分块 → 嵌入 → 写向量），而 FastAPI 的
``BackgroundTasks`` 把任务挂在请求生命周期上——**没有并发上限**（一次上传 20 个文件就把
免费 embedding 额度打爆）、**没有重试**（上游抖一下就永久 failed）、**没有进度**
（只能靠文档状态猜）。这三个问题都不是业务代码能解决的，得由队列负责。

契约（每个实现都要满足）：

* ``submit`` 立即返回 task_id，任务在后台跑；失败要留**可读原因**（由业务函数写回
  ``document.error_msg``，队列只负责记录最后一次异常）；
* 任务必须**幂等**：重试不重复入库（依赖 ``delete_document`` + 文档状态机保证）；
* 并发有上限，且状态可查（``pending`` / ``running`` / ``retrying`` / ``succeeded`` / ``failed``）。

已知边界：进程内实现的任务只存在于本进程，重启即丢；此时文档会停在 ``pending`` 状态，
用 ``POST /api/doc/{id}/reprocess`` 或重跑建库兜住。需要跨副本 / 持久化队列时注册一个
arq / celery 实现即可（见 `docs/architecture.md` 第 5 节），业务代码不用改。
"""

from __future__ import annotations

import asyncio
import time
from collections import Counter, OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol
from uuid import uuid4

from loguru import logger

from inner_rag.core.config import settings
from inner_rag.core.metrics import metrics
from inner_rag.plugins.registry import task_queues

TaskFn = Callable[..., Awaitable[Any]]


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    RETRYING = "retrying"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass
class TaskRecord:
    """一个任务的执行记录（保留最近 N 条，见 ``TASK_QUEUE_HISTORY``）。"""

    id: str
    name: str
    status: TaskStatus = TaskStatus.PENDING
    attempts: int = 0
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "attempts": self.attempts,
            "error": self.error,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
        }


@dataclass
class _Job:
    record: TaskRecord
    fn: TaskFn
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


class TaskQueue(Protocol):
    """任务队列契约（五件套里的「接口」）。

    ``start`` / ``stop`` 由应用生命周期（``lifespan``）调用：需要 worker 的实现在这里
    起停；同步实现留空即可。
    """

    name: str

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def submit(self, name: str, fn: TaskFn, *args: Any, **kwargs: Any) -> str: ...

    def status(self, task_id: str) -> TaskRecord | None: ...

    def summary(self) -> dict[str, Any]: ...


class InProcessTaskQueue:
    """进程内 asyncio 队列：有界并发 + 有限重试 + 有界状态历史。

    worker 数量 = ``TASK_QUEUE_CONCURRENCY``：并发上限同时也是**额度保护**——
    免费 embedding 路由对突发并发很敏感。
    """

    name = "inprocess"

    def __init__(
        self,
        concurrency: int,
        max_retries: int,
        retry_backoff: float,
        history: int,
    ) -> None:
        self._concurrency = max(1, concurrency)
        self._max_retries = max(0, max_retries)
        self._retry_backoff = retry_backoff
        self._history = max(1, history)
        self._pending: asyncio.Queue[_Job] = asyncio.Queue()
        self._records: OrderedDict[str, TaskRecord] = OrderedDict()
        self._workers: list[asyncio.Task[None]] = []

    # ── 生命周期 ───────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._workers:
            return
        self._workers = [
            asyncio.create_task(self._worker(index)) for index in range(self._concurrency)
        ]
        logger.info(f"[QUEUE] {self.name} 已启动：并发上限 {self._concurrency}")

    async def stop(self) -> None:
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        # 未跑完的任务直接丢弃：任务只存在于本进程，重启会丢；这里只要让运维知道丢了多少
        dropped = self._pending.qsize()
        if dropped:
            logger.warning(f"[QUEUE] 停止时有 {dropped} 个任务未执行（需重新提交）")

    # ── 提交与查询 ─────────────────────────────────────────────────────

    async def submit(self, name: str, fn: TaskFn, *args: Any, **kwargs: Any) -> str:
        if not self._workers:
            msg = (
                "任务队列未启动：需要 worker 的实现必须在应用 lifespan 里 `await task_queue.start()`"
                "（测试或脚本可用 TASK_QUEUE_BACKEND=inline）"
            )
            raise RuntimeError(msg)
        record = TaskRecord(id=uuid4().hex[:12], name=name)
        self._remember(record)
        await self._pending.put(_Job(record=record, fn=fn, args=args, kwargs=kwargs))
        metrics.increment("rag_task_submitted_total", labels={"name": name})
        return record.id

    def status(self, task_id: str) -> TaskRecord | None:
        return self._records.get(task_id)

    def summary(self) -> dict[str, Any]:
        counts = Counter(record.status.value for record in self._records.values())
        return {
            "backend": self.name,
            "concurrency": self._concurrency,
            "queued": self._pending.qsize(),
            "counts": dict(counts),
        }

    # ── 内部 ───────────────────────────────────────────────────────────

    def _remember(self, record: TaskRecord) -> None:
        """保留最近 N 条记录：长跑进程里状态表必须有界，否则就是内存泄漏。"""
        self._records[record.id] = record
        while len(self._records) > self._history:
            self._records.popitem(last=False)

    async def _worker(self, index: int) -> None:
        while True:
            job = await self._pending.get()
            try:
                await self._run(job)
            except asyncio.CancelledError:
                logger.warning(f"[QUEUE] worker#{index} 被取消，任务 {job.record.name} 未完成")
                raise
            finally:
                self._pending.task_done()

    async def _run(self, job: _Job) -> None:
        record = job.record
        while True:
            record.status = TaskStatus.RUNNING
            record.attempts += 1
            started = time.perf_counter()
            try:
                await job.fn(*job.args, **job.kwargs)
            except Exception as exc:
                record.error = str(exc)[:500]
                metrics.increment(
                    "rag_task_total", labels={"name": record.name, "status": "failed"}
                )
                if record.attempts <= self._max_retries:
                    record.status = TaskStatus.RETRYING
                    logger.warning(
                        f"[QUEUE] 任务 {record.name} 第 {record.attempts} 次失败，"
                        f"{self._retry_backoff}s 后重试: {exc}"
                    )
                    metrics.increment("rag_task_retried_total", labels={"name": record.name})
                    # 退避放在 worker 内：其他 worker 继续处理队列，不会因为一次抖动全线等待
                    await asyncio.sleep(self._retry_backoff)
                    continue
                record.status = TaskStatus.FAILED
                record.finished_at = time.time()
                logger.error(f"[QUEUE] 任务 {record.name} 最终失败（{record.attempts} 次）: {exc}")
                return

            record.status = TaskStatus.SUCCEEDED
            record.finished_at = time.time()
            metrics.observe(
                "rag_task_duration_ms",
                (time.perf_counter() - started) * 1000,
                labels={"name": record.name},
            )
            metrics.increment("rag_task_total", labels={"name": record.name, "status": "succeeded"})
            logger.info(f"[QUEUE] 任务 {record.name} 完成（{record.attempts} 次尝试）")
            return


class InlineTaskQueue:
    """同步执行：``submit`` 直接跑完再返回。

    用于测试、脚本与「上传后立刻要结果」的小部署。刻意不做重试与并发控制——它的语义就是
    「立刻跑」：失败直接抛给调用方，由调用方决定怎么呈现（HTTP 层映射成响应，脚本记日志）。
    """

    name = "inline"

    async def start(self) -> None:
        """无需 worker：留空是为了满足契约，调用方（lifespan）不用区分实现。"""

    async def stop(self) -> None:
        """同上。"""

    async def submit(self, name: str, fn: TaskFn, *args: Any, **kwargs: Any) -> str:
        await fn(*args, **kwargs)
        return uuid4().hex[:12]

    def status(self, task_id: str) -> TaskRecord | None:
        return None  # 同步执行没有可查询的中间状态

    def summary(self) -> dict[str, Any]:
        return {"backend": self.name, "concurrency": 1, "queued": 0, "counts": {}}


def _build_inprocess() -> TaskQueue:
    return InProcessTaskQueue(
        concurrency=settings.TASK_QUEUE_CONCURRENCY,
        max_retries=settings.TASK_QUEUE_MAX_RETRIES,
        retry_backoff=settings.TASK_QUEUE_RETRY_BACKOFF,
        history=settings.TASK_QUEUE_HISTORY,
    )


task_queues.register("inprocess", _build_inprocess)
task_queues.register("inline", InlineTaskQueue)


def build_task_queue(name: str | None = None) -> TaskQueue:
    """按名字（默认取 ``TASK_QUEUE_BACKEND``）构造队列；未知名字抛可读错误。

    不做静默降级：把「任务跑在哪个队列里」悄悄换掉，会让「为什么重试没生效」这类问题
    极难排查（与 ``build_vector_store`` / ``build_cache_backend`` 同一原则）。
    """
    task_queues.load_entry_points()
    backend = (name or settings.TASK_QUEUE_BACKEND).strip().lower()
    factory: Callable[[], TaskQueue] | None = task_queues.get(backend)
    if factory is None:
        msg = f"不支持的 TASK_QUEUE_BACKEND: {backend!r}（可选 {'/'.join(task_queues.names())}）"
        raise ValueError(msg)
    return factory()


task_queue = build_task_queue()
