"""LangSmith 真实连通性用例（live：需 Key 与网络，默认不跑）。

运行::

    export LANGSMITH_API_KEY=...
    uv run pytest -m live -q

这些用例验证的是「离线用例验证不了」的那一半：trace 真的到了服务端、feedback 真的
挂在了那条 trace 上。离线侧只保证调用序列正确（见 tests/test_eval_pipeline.py）。
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any

import pytest

pytestmark = pytest.mark.live


def _skip_reason() -> str | None:
    if not os.environ.get("LANGSMITH_API_KEY"):
        return "未设置 LANGSMITH_API_KEY（live 用例不写死密钥，只从环境变量读）"
    return None


@pytest.fixture(scope="module")
def client() -> Any:
    if reason := _skip_reason():
        pytest.skip(reason)

    # conftest 为了让离线用例零网络，把 LANGSMITH_TRACING 压成了 false。
    # live 用例要的就是联网，所以自己把开关打开——否则会静悄悄全 skip，
    # 看起来像「跑过了」，实际一条都没验。
    from inner_rag.core.config import settings
    from inner_rag.core.observability import tracer

    settings.LANGSMITH_TRACING = True
    settings.LANGSMITH_API_KEY = os.environ["LANGSMITH_API_KEY"]
    tracer.configure()
    if not tracer.enabled or tracer.client is None:
        pytest.skip(f"追踪未启用：{tracer.reason}")
    return tracer.client


@pytest.fixture(scope="module")
def project_id(client: Any) -> str | None:
    from benchmark import langsmith_sync

    return langsmith_sync.project_id(client)


async def _wait_for(client: Any, run_id: str, project_id: str | None, timeout: float = 30.0) -> Any:
    """轮询读回 run：上报是后台批量的，立刻读会 404。

    必须是 async：`runs.retrieve` 在这个客户端上返回 coroutine，同步调用只会拿到一个
    永远没被 await 的协程对象（连 RuntimeWarning 都出来了），读回的 id 自然是空。
    """
    import asyncio
    import inspect

    deadline = time.time() + timeout
    last: Exception | None = None
    while time.time() < deadline:
        try:
            result = client.runs.retrieve(run_id, project_id=project_id)
            if inspect.isawaitable(result):
                result = await result
            if str(getattr(result, "id", "")) == str(run_id):
                return result
        except Exception as exc:  # 404 = 还在队列里
            last = exc
        await asyncio.sleep(2)
    pytest.fail(f"{timeout}s 内没读回 run {run_id}：{last}")


async def test_trace_reaches_server_and_can_be_read_back(
    client: Any, project_id: str | None
) -> None:
    """建一条带子 span 的 trace，服务端必须能按同一个 run id 读回。

    `run.post()` 不抛异常 ≠ 上报成功（只是进了本地队列），所以必须回读。
    """
    from inner_rag.core.metrics import metrics
    from inner_rag.core.observability import tracer

    marker = f"live-{uuid.uuid4().hex[:8]}"
    async with tracer.span("live.check", marker=marker) as span:
        async with tracer.span("live.child") as child:
            child.set(ok=True)
        span.set(marker=marker)
        run_id = span.trace_id

    assert run_id, "没有拿到 run id：追踪未启用或未被采样"
    assert metrics.snapshot()["counters"].get("tracing_errors_total", 0) == 0

    run = await _wait_for(client, run_id, project_id)
    assert str(getattr(run, "id", "")) == str(run_id)


async def test_feedback_attaches_to_that_trace(client: Any, project_id: str | None) -> None:
    """分数必须挂在具体 trace 上：否则只能看到实验维度的平均分，无法下钻。"""
    from inner_rag.core.observability import tracer

    async with tracer.span("live.feedback-target") as span:
        span.set(purpose="live feedback check")
        run_id = span.trace_id
    assert run_id
    await _wait_for(client, run_id, project_id)

    from benchmark import langsmith_sync

    written = langsmith_sync.push_feedback(
        client,
        [
            {
                "id": "live-probe",
                "trace_id": run_id,
                "judge_correct": 1,
                "citation_precision": 0.5,
                "faithfulness": 1.0,
            }
        ],
    )
    assert written == 3, "三个分数应各写一条"

    deadline = time.time() + 30
    keys: set[str] = set()
    while time.time() < deadline and len(keys) < 3:
        keys = {fb.key for fb in client.list_feedback(run_ids=[run_id])}
        if len(keys) < 3:
            time.sleep(2)
    assert {"correctness", "citation_precision", "faithfulness"} <= keys


async def test_dataset_sync_is_idempotent(client: Any) -> None:
    """重复同步不该报错也不该无限膨胀：dataset 按名字幂等，example 按 eval_id 对齐。"""
    from benchmark import dataset as ds
    from benchmark import langsmith_sync

    items = ds.load_eval_set()
    name = "inner-rag-dragon-v1"
    assert langsmith_sync.sync_dataset(client, items, name) == name
    assert langsmith_sync.sync_dataset(client, items, name) == name
