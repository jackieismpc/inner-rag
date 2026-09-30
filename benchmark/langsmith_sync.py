"""LangSmith 联动（可选）：评测集同步为 dataset，逐题分数回写为 feedback。

设计前提来自 `docs/evaluation.md` 6.3：**LangSmith 关闭时必须能完整跑完评测**。
所以这里所有的对外调用都走同一个门面——拿不到客户端就是 no-op，评测结果照样写本地报告。
追踪后端是「看得更清楚」的手段，不是评测的依赖。

本模块未用真实 Key 做过端到端验证（Phase 6 的环境里没有 ``LANGSMITH_API_KEY``）：
调用序列用假客户端在 ``tests/test_eval_pipeline.py`` 里断言过，真实连通性待有 Key 时补验。
"""

from __future__ import annotations

from typing import Any, Protocol


class LangSmithClient(Protocol):
    """只声明本模块用到的四个方法，便于用假客户端做单测（不联网）。"""

    def has_dataset(self, dataset_name: str) -> bool: ...
    def create_dataset(self, dataset_name: str, description: str) -> Any: ...
    def create_example(self, **kwargs: Any) -> Any: ...
    def create_feedback(self, **kwargs: Any) -> Any: ...


def client_or_none() -> LangSmithClient | None:
    """按配置拿客户端；未启用或初始化失败都返回 None（评测照常跑完）。"""
    from inner_rag.core.config import settings
    from inner_rag.core.observability import tracer

    if not settings.LANGSMITH_TRACING or not settings.LANGSMITH_API_KEY:
        return None
    tracer.configure()
    return tracer.client


def sync_dataset(
    client: LangSmithClient | None, items: list[dict[str, Any]], dataset_name: str
) -> str | None:
    """把评测集同步为 LangSmith dataset，返回 dataset id；失败返回 None（不抛给调用方）。

    ``id`` 作为 example 的 metadata（``eval_id``）：LangSmith 的 example 主键是它自己的 uuid，
    用评测集 id 做外部键才能在两次实验之间对齐同一道题。
    """
    if client is None:
        return None
    try:
        if not client.has_dataset(dataset_name=dataset_name):
            client.create_dataset(
                dataset_name=dataset_name,
                description="inner-rag 龙族评测集（docs/datasets/dragon_king/eval_v1.jsonl）",
            )
        for item in items:
            client.create_example(
                dataset_name=dataset_name,
                inputs={"question": item["question"]},
                outputs={"expected_answer": item.get("expected_answer", "")},
                metadata={
                    "eval_id": item["id"],
                    "category": item.get("category"),
                    "expect_refusal": item.get("expect_refusal"),
                    "anchor_pages": [c["page"] for c in item.get("citations", []) if "page" in c],
                },
            )
    except Exception as exc:  # 回写失败不影响本地报告
        print(f"[langsmith] 同步 dataset 失败（忽略，继续跑评测）：{exc}")
        return None
    return dataset_name


def push_feedback(client: LangSmithClient | None, results: list[dict[str, Any]]) -> int:
    """逐题把分数回写为 feedback；返回成功条数。

    只回写有 ``trace_id`` 的题：feedback 挂在具体 trace 上才有下钻价值，
    没有 trace（追踪关闭或未被采样）时宁可少写也不写一堆对不上的分数。
    """
    if client is None:
        return 0
    written = 0
    for result in results:
        trace_id = result.get("trace_id")
        if not trace_id:
            continue
        scores = {
            "correctness": result.get("judge_correct"),
            "citation_precision": result.get("citation_precision"),
            "faithfulness": result.get("faithfulness"),
        }
        for key, value in scores.items():
            if value is None:
                continue
            try:
                client.create_feedback(
                    trace_id=trace_id,
                    key=key,
                    score=float(value),
                    comment=f"{result.get('id')}: {result.get('judge_reason') or ''}"[:200],
                )
                written += 1
            except Exception as exc:  # 逐条失败不该中断其余回写
                print(f"[langsmith] feedback 回写失败 {result.get('id')}/{key}: {exc}")
    return written
