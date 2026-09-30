"""任务队列插件点测试：进程内实现的并发上限 / 重试 / 状态历史，以及配置校验。

HTTP 链路用的是 ``inline`` 后端（见 ``tests/conftest.py``，保证上传用例确定性），
因此队列自身的行为在这里直接断言。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest

from inner_rag.services.task_queue import (
    InlineTaskQueue,
    InProcessTaskQueue,
    TaskStatus,
    build_task_queue,
)


def _queue(
    *,
    concurrency: int = 2,
    max_retries: int = 0,
    retry_backoff: float = 0.0,
    history: int = 10,
) -> InProcessTaskQueue:
    return InProcessTaskQueue(
        concurrency=concurrency,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
        history=history,
    )


async def _wait_until(predicate: Callable[[], bool], timeout: float = 2.0) -> None:
    """等队列把任务跑完：轮询状态而不是 sleep 固定时长，避免用例变慢或变脆。"""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"等待超时（{timeout}s）")


async def _noop() -> None:
    return None


# ── 契约与配置 ────────────────────────────────────────────────────────


async def test_submit_before_start_is_rejected() -> None:
    """需要 worker 的实现没启动就提交，必须显性报错：否则任务会静默丢失。"""
    queue = _queue()

    with pytest.raises(RuntimeError) as excinfo:
        await queue.submit("job", _noop)
    assert "start" in str(excinfo.value)


def test_build_task_queue_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError) as excinfo:
        build_task_queue("celery")
    message = str(excinfo.value)
    assert "TASK_QUEUE_BACKEND" in message
    assert "inprocess" in message  # 可选值来自注册表


async def test_inline_backend_runs_immediately_and_propagates_error() -> None:
    queue = InlineTaskQueue()
    ran: list[int] = []

    async def job(value: int) -> None:
        ran.append(value)

    assert await queue.submit("job", job, 1)
    assert ran == [1]

    async def boom() -> None:
        raise ValueError("上游挂了")

    with pytest.raises(ValueError):  # 同步执行：失败直接抛给调用方
        await queue.submit("boom", boom)


# ── 进程内实现 ────────────────────────────────────────────────────────


async def test_concurrency_is_capped_and_all_tasks_succeed() -> None:
    """并发上限是「额度保护」的核心：超出的任务必须排队，而不是一起打上游。"""
    queue = _queue(concurrency=2)
    await queue.start()
    running = 0
    peak = 0

    async def job() -> None:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.02)
        running -= 1

    try:
        for _ in range(6):
            await queue.submit("job", job)
        await _wait_until(lambda: queue.summary()["counts"].get("succeeded", 0) == 6)
    finally:
        await queue.stop()

    assert peak == 2
    assert queue.summary()["queued"] == 0


async def test_transient_failure_is_retried_then_succeeds() -> None:
    queue = _queue(max_retries=1)
    await queue.start()
    attempts = 0

    async def flaky() -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("免费 embedding 路由偶发 Connection error")

    def state(task_id: str) -> TaskStatus | None:
        record = queue.status(task_id)
        return record.status if record else None

    try:
        task_id = await queue.submit("flaky", flaky)
        await _wait_until(lambda: state(task_id) in {TaskStatus.SUCCEEDED, TaskStatus.FAILED})
        record = queue.status(task_id)
    finally:
        await queue.stop()

    assert record is not None
    assert record.status is TaskStatus.SUCCEEDED
    assert record.attempts == 2


async def test_permanent_failure_records_reason_and_stops_retrying() -> None:
    queue = _queue(max_retries=1)
    await queue.start()

    async def boom() -> None:
        raise RuntimeError("解析器不支持这个文件")

    def state(task_id: str) -> TaskStatus | None:
        record = queue.status(task_id)
        return record.status if record else None

    try:
        task_id = await queue.submit("boom", boom)
        await _wait_until(lambda: state(task_id) is TaskStatus.FAILED)
        record = queue.status(task_id)
    finally:
        await queue.stop()

    assert record is not None
    assert record.attempts == 2  # 首次 + 1 次重试
    assert record.error == "解析器不支持这个文件"
    assert queue.summary()["counts"] == {"failed": 1}


async def test_status_history_is_bounded() -> None:
    """状态表必须有界：长跑进程里每个任务都留一条就是内存泄漏。"""
    queue = _queue(history=2)
    await queue.start()
    try:
        first = await queue.submit("job", _noop)
        await queue.submit("job", _noop)
        await queue.submit("job", _noop)
    finally:
        await queue.stop()

    assert queue.status(first) is None
    assert queue.status("不存在") is None
