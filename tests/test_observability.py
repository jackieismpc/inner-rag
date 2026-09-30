"""可观测性测试（Phase 5）：request_id、结构化日志、span 树与指标。

每个用例对应一个失效场景（与 `docs/observability.md` 第 6 节的 DoD 一一对应）：

1. tracing 默认关闭 → 零网络调用；
2. 追踪上报失败 → 请求照旧 200，且计入 `tracing_errors_total`；
3. request_id 在响应头与日志里是同一个值（它是日志 ↔ trace 的唯一钥匙）；
4. `LOG_FORMAT=json` 时每行日志都能被 `json.loads`；
5. span 树的父子层级正确（能不能定位到最慢的一步，取决于这棵树对不对）；
6. 指标按标签累计，Prometheus 导出可被抓取器解析。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from loguru import logger

from inner_rag.core.config import settings
from inner_rag.core.logging import json_line
from inner_rag.core.metrics import metrics
from inner_rag.core.observability import tracer

QUESTION = "系统支持哪些检索能力？"

_LANGSMITH_ENV_VARS = (
    "LANGSMITH_TRACING",
    "LANGSMITH_PROJECT",
    "LANGSMITH_API_KEY",
    "LANGSMITH_ENDPOINT",
    "LANGSMITH_WORKSPACE_ID",
)


@pytest.fixture
def json_logs() -> Iterator[list[str]]:
    """临时挂一个 JSON sink 收集日志：结构化格式是本阶段的产物，必须能被解析。"""
    lines: list[str] = []
    sink_id = logger.add(lines.append, level="DEBUG", format=json_line)
    yield lines
    logger.remove(sink_id)


@pytest.fixture(autouse=True)
def reset_metrics() -> Iterator[None]:
    metrics.reset()
    yield
    metrics.reset()


@pytest.fixture
def tracing_enabled(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """临时打开 LangSmith 追踪（假 Key，且不产生任何网络请求）。

    为什么在 teardown 里显式 `monkeypatch.undo()` 再 `configure()`：tracer 是进程级单例，
    必须在 settings 复位**之后**重新初始化，否则后续用例会带着「追踪已开启」的状态跑。
    """
    monkeypatch.setattr(settings, "LANGSMITH_TRACING", True)
    monkeypatch.setattr(settings, "LANGSMITH_API_KEY", "not-a-real-key")
    for name in _LANGSMITH_ENV_VARS:
        monkeypatch.setenv(name, "")
    tracer.configure()
    yield tracer
    monkeypatch.undo()
    tracer.configure()


@pytest.fixture
def chat_ready(client: TestClient, kb: dict, monkeypatch: pytest.MonkeyPatch) -> dict:
    """知识库里有一篇能召回的文档。阈值调 0：假 embedding 的分数本来就没语义，
    否则「有没有查到」会取决于哈希碰撞而不是检索链路。"""
    monkeypatch.setattr(settings, "RETRIEVAL_SCORE_THRESHOLD", 0.0)
    response = client.post(
        "/api/doc/upload",
        data={"kb_id": str(kb["id"])},
        files={
            "files": ("note.txt", "本系统支持文档上传、向量检索与混合检索。".encode(), "text/plain")
        },
    )
    assert response.status_code == 200, response.text
    return kb


def test_tracing_is_local_by_default() -> None:
    """默认不接 LangSmith：离线与 CI 必须零网络、零费用。"""
    status = tracer.status()
    assert status["backend"] == "local"
    assert status["enabled"] is False


def test_no_network_when_tracing_disabled(
    chat_ready: dict, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """关闭 tracing 时，一次问答不应产生任何真实 HTTP 调用（含 LangSmith 上报）。"""

    def forbid(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("离线用例不允许发起网络请求")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbid)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", forbid)

    response = client.post("/api/chat/send", json={"kb_id": chat_ready["id"], "question": QUESTION})
    assert response.status_code == 200, response.text


def test_tracing_upload_failure_does_not_break_request(
    chat_ready: dict, client: TestClient, tracing_enabled: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """上报失败只计数：追踪是旁路，不能把观测系统变成故障源。"""
    assert tracing_enabled.enabled, tracing_enabled.reason

    def boom(_run: Any) -> None:
        raise RuntimeError("langsmith 不可达")

    monkeypatch.setattr(tracer, "_upload", boom)

    response = client.post("/api/chat/send", json={"kb_id": chat_ready["id"], "question": QUESTION})
    assert response.status_code == 200, response.text
    assert metrics.snapshot()["counters"].get("tracing_errors_total", 0) >= 1


def test_json_log_lines_are_parseable(
    chat_ready: dict, client: TestClient, json_logs: list[str]
) -> None:
    """`LOG_FORMAT=json` 的每行日志都必须能被采集端解析（DoD 2）。"""
    response = client.post("/api/chat/send", json={"kb_id": chat_ready["id"], "question": QUESTION})
    assert response.status_code == 200, response.text
    assert json_logs, "日志 sink 没有收到任何记录"

    events = [json.loads(line)["event"] for line in json_logs]
    assert "http_access" in events


def test_request_id_matches_between_response_and_logs(
    chat_ready: dict, client: TestClient, json_logs: list[str]
) -> None:
    """同一个 request_id 必须同时出现在响应头与日志里，否则串不起来。"""
    response = client.post("/api/chat/send", json={"kb_id": chat_ready["id"], "question": QUESTION})
    assert response.status_code == 200, response.text
    header_rid = response.headers.get("x-request-id")
    assert header_rid, "响应头缺少 X-Request-ID"

    records = [json.loads(line) for line in json_logs]
    access = [item for item in records if item["event"] == "http_access"]
    assert access, "没有访问日志"
    assert access[-1]["request_id"] == header_rid


def test_request_id_is_passed_through(
    chat_ready: dict, client: TestClient, json_logs: list[str]
) -> None:
    """上游（网关 / 前端）传来的 request_id 要原样透传，否则跨系统串联断掉。"""
    response = client.post(
        "/api/chat/send",
        json={"kb_id": chat_ready["id"], "question": QUESTION},
        headers={"X-Request-ID": "gateway-123"},
    )
    assert response.status_code == 200, response.text
    assert response.headers.get("x-request-id") == "gateway-123"

    records = [json.loads(line) for line in json_logs]
    access = [item for item in records if item["event"] == "http_access"]
    assert access[-1]["request_id"] == "gateway-123"


def test_span_tree_nests_chat_steps(
    chat_ready: dict, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """span 树的层级决定「能不能定位到最慢的一步」，父子关系错了整棵 trace 就废了。"""
    recorded: list[tuple[str | None, str]] = []

    def spy(span: Any) -> None:
        recorded.append((span.parent.name if span.parent else None, span.name))

    monkeypatch.setattr(tracer, "_log", spy)
    response = client.post("/api/chat/send", json={"kb_id": chat_ready["id"], "question": QUESTION})
    assert response.status_code == 200, response.text

    assert ("rag.request", "retrieve") in recorded
    assert ("retrieve", "vector.search") in recorded
    assert ("vector.search", "embed.query") in recorded
    assert ("rag.request", "prompt.build") in recorded
    assert ("rag.request", "llm.generate") in recorded


def test_metrics_endpoint_reports_request_and_retrieval(
    chat_ready: dict, client: TestClient
) -> None:
    """/metrics 要能反映请求数、检索耗时与空召回（排障的第一个入口）。"""
    response = client.post("/api/chat/send", json={"kb_id": chat_ready["id"], "question": QUESTION})
    assert response.status_code == 200, response.text

    payload = client.get("/api/system/metrics").json()["data"]
    assert payload["scope"] == "process"
    counters = payload["metrics"]["counters"]
    histograms = payload["metrics"]["histograms"]

    assert any(key.startswith("rag_requests_total{") for key in counters)
    retrieve_key = next(key for key in histograms if key.startswith("rag_retrieve_duration_ms"))
    assert histograms[retrieve_key]["count"] == 1
    assert histograms[retrieve_key]["p95"] >= 0


def test_metrics_endpoint_requires_token_when_configured(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """配了 METRICS_TOKEN 就只认 token：抓取器没有登录态，但接口不能因此裸奔。"""
    monkeypatch.setattr(settings, "METRICS_TOKEN", "metrics-secret")
    assert client.get("/api/system/metrics").status_code == 401
    assert (
        client.get("/api/system/metrics", headers={"X-Metrics-Token": "metrics-secret"}).status_code
        == 200
    )


def test_prometheus_export_is_parseable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`METRICS_BACKEND=prometheus` 输出标准 exposition 文本（可被引擎直接抓取）。"""
    monkeypatch.setattr(settings, "METRICS_BACKEND", "prometheus")
    metrics.increment(
        "rag_requests_total", labels={"endpoint": "GET /api/system/metrics", "status": "200"}
    )

    body = client.get("/api/system/metrics").text
    assert "# TYPE rag_requests_total counter" in body
    assert 'rag_requests_total{endpoint="GET /api/system/metrics",status="200"} 1.0' in body
