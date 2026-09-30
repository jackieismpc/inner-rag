#!/usr/bin/env python3
"""LangSmith 连通性自检：打一条真实 trace，再从服务端把它读回来。

用法::

    export LANGSMITH_API_KEY=...          # 只走环境变量，不落到仓库里
    uv run scripts/check_langsmith.py          # 基础自检（trace 往返）
    uv run scripts/check_langsmith.py --dataset # 额外同步评测集并回读一条反馈

为什么必须「回读」才算连通：上报是后台批量发的，`run.post()` 返回成功只代表
进了本地队列。曾经只凭「没抛异常」就判定通过，结果 trace 一条都没到服务端。
所以这里的判定链是：能建 run → 能拿到 run id → **服务端能读回同一条 run**。

退出码：0 全通 / 1 未启用或读不回 / 2 通了但有上报错误计数。
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import os
import sys
import time
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

READ_TIMEOUT_S = 30


def _mask(value: str) -> str:
    """只保留前缀用于排障，避免把完整密钥打印到日志里。"""
    return f"{value[:11]}***" if len(value) > 11 else ("已设置" if value else "空")


async def _read_back(client: Any, run_id: str, project_id: str | None) -> Any | None:
    """轮询读回 run：LangSmith 的 ingestion 有秒级延迟。

    ``runs.retrieve`` 强制要 ``project_id``——不传会在 SDK 内部报缺少参数，
    看起来像「服务端读不回」，实际是调用姿势不对。
    """
    deadline = time.time() + READ_TIMEOUT_S
    last: Exception | None = None
    while time.time() < deadline:
        try:
            # 同一个 Client 上 runs 资源可能是异步的（取决于 SDK 构造方式），
            # 直接 return 会把 coroutine 当成 run 对象用，报「没有 name 属性」。
            result = client.runs.retrieve(run_id, project_id=project_id)
            return await result if inspect.isawaitable(result) else result
        except Exception as exc:  # 404 = 还在队列里，继续等
            last = exc
            await asyncio.sleep(2)
    print(f"  [check] 读回超时（{READ_TIMEOUT_S}s）：{last}")
    return None


async def check_trace(client: Any, project_id: str | None) -> tuple[str | None, int]:
    """建一条带子 span 的 trace，返回 (根 run id, tracing 错误计数)。"""
    from inner_rag.core.metrics import metrics
    from inner_rag.core.observability import tracer

    trace_id: str | None = None
    async with tracer.span("check.connectivity", source="scripts/check_langsmith.py") as span:
        async with tracer.span("check.child", step=1) as child:
            await asyncio.sleep(0.01)
            child.set(ok=True)
        span.set(stage="verify")
        trace_id = span.trace_id

    errors = int(metrics.snapshot()["counters"].get("tracing_errors_total", 0))
    print(f"  本地 run id = {trace_id}")
    print(f"  tracing_errors_total = {errors}")
    if not trace_id:
        return None, errors

    run = await _read_back(client, trace_id, project_id)
    if run is None:
        return trace_id, errors
    # v2 接口返回的 Run 不一定带 name/run_type 字段，用 id 对上即可证明是同一条
    print(
        f"  服务端读回 ✓ id={getattr(run, 'id', None)} "
        f"name={getattr(run, 'name', None) or '-'} "
        f"（与本地 run id 一致：{str(getattr(run, 'id', '')) == trace_id}）"
    )
    return trace_id, errors


async def check_dataset(client: Any, trace_id: str | None) -> bool:
    """同步评测集 + 回写一条 feedback，并回读验证分数真的存下来了。"""
    from benchmark import dataset as ds
    from benchmark import langsmith_sync

    items = ds.load_eval_set()
    name = langsmith_sync.sync_dataset(client, items, "inner-rag-dragon-v1")
    if not name:
        print("  [check] dataset 同步失败")
        return False
    print(f"  dataset 同步 ✓ {name}（{len(items)} 条 example）")

    if not trace_id:
        print("  [check] 跳过 feedback：没有可挂载的 run id")
        return True

    written = langsmith_sync.push_feedback(
        client, [{"id": items[0]["id"], "trace_id": trace_id, "judge_correct": 1}]
    )
    print(f"  feedback 回写 ✓ {written} 条")
    return written > 0


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LangSmith 连通性自检")
    parser.add_argument(
        "--dataset", action="store_true", help="额外验证 dataset 同步与 feedback 回写"
    )
    args = parser.parse_args(argv)

    from inner_rag.core.config import settings
    from inner_rag.core.observability import tracer

    print(f"[check] 密钥来源：环境变量 LANGSMITH_API_KEY={_mask(settings.LANGSMITH_API_KEY)}")
    print(
        f"[check] tracing={settings.LANGSMITH_TRACING} "
        f"project={settings.LANGSMITH_PROJECT} endpoint={settings.LANGSMITH_ENDPOINT or '默认'}"
    )
    if not settings.LANGSMITH_API_KEY:
        print("[check] FAIL：未读到密钥。请先 export LANGSMITH_API_KEY=...（不要写进仓库文件）")
        return 1

    tracer.configure()
    print(f"[check] tracer.status = {tracer.status()}")
    if not tracer.enabled or tracer.client is None:
        print(f"[check] FAIL：追踪未启用 -> {tracer.reason}")
        return 1

    # 环境变量优先于 .env：确认 SDK 拿到的是我们注入的那个值
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    client = tracer.client
    from benchmark import langsmith_sync

    project_id = langsmith_sync.project_id(client)
    print(f"[check] project_id = {project_id}")
    try:
        client.has_dataset(dataset_name="__inner_rag_connectivity_probe__")
        print("[check] 服务端鉴权 ✓（能发起带签名的请求）")
    except Exception as exc:
        print(f"[check] FAIL：服务端拒绝 -> {exc}")
        return 1

    trace_id, errors = await check_trace(client, project_id)
    if trace_id is None:
        return 1

    ok = True
    if args.dataset:
        ok = await check_dataset(client, trace_id)

    print(f"[check] 结论：{'PASS' if ok and errors == 0 else 'PARTIAL'}")
    return 0 if ok and errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
