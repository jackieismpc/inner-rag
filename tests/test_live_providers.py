"""Phase 2 G2：真实 provider 联网验收（默认 deselect）。

运行方式::

    uv run pytest -m live -q

前置条件：``.env`` 里填好 ``OPENROUTER_API_KEY``（embedding 用免费路由，
chat 会产生少量费用）；DeepSeek 用例另需 ``DEEPSEEK_API_KEY``。
没有 Key 的用例会 skip，而不是失败。

成本控制：只嵌入 docs/samples/acceptance.txt 这一份小文档，且 embedding
结果进 EmbeddingCache，同一次运行内不会重复计费。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk

from inner_rag.core.config import settings
from inner_rag.providers import get_chat_model, get_embeddings, reset_cache
from inner_rag.services.embedding import embedding_service

pytestmark = pytest.mark.live

ACCEPTANCE_FILE = Path(__file__).resolve().parents[1] / "docs" / "samples" / "acceptance.txt"


def _require_openrouter_key() -> None:
    if not settings.OPENROUTER_API_KEY.strip():
        pytest.skip("未配置 OPENROUTER_API_KEY（见 .env.example），跳过真实 provider 验收")


def _require_deepseek_key() -> None:
    if not settings.DEEPSEEK_API_KEY.strip():
        pytest.skip("未配置 DEEPSEEK_API_KEY（见 .env.example），跳过 DeepSeek 验收")


@pytest.fixture
def openrouter(monkeypatch: pytest.MonkeyPatch):
    """把 chat 与 embedding 都切到 OpenRouter 真实 API。

    注意：这里刻意不关闭 provider 的 HTTP 客户端。实测在 fixture teardown 里调
    ``model.root_async_client.close()`` 之后，后续用例会报
    ``Cannot send a request, as the client has been closed``（客户端被复用），
    而事件循环结束时那行 ``Event loop is closed`` 只会在 ``-rA`` 这类详细输出里出现，
    不影响任何断言。
    """
    _require_openrouter_key()
    monkeypatch.setattr(settings, "LLM_PROVIDER", "openrouter")
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "openrouter")
    # 免费 embedding 路由上下文只有 512 token，验收时显式截断
    monkeypatch.setattr(settings, "EMBEDDING_MAX_INPUT_CHARS", 400)
    reset_cache()
    embedding_service._embeddings = None
    yield
    reset_cache()
    embedding_service._embeddings = None


@pytest.fixture
def deepseek(monkeypatch: pytest.MonkeyPatch):
    """chat 切到 DeepSeek 官方 API；embedding 保持 conftest 的 mock，不额外产生费用。"""
    _require_deepseek_key()
    monkeypatch.setattr(settings, "LLM_PROVIDER", "deepseek")
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "mock")
    reset_cache()
    yield
    reset_cache()


def _parse_sse(body: str) -> list[dict]:
    events = []
    for line in body.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: ") :]))
    return events


async def test_live_chat_completion_and_streaming(openrouter: None) -> None:
    model = get_chat_model()
    answer = await model.ainvoke("用一句话回答：1+1 等于几？")
    assert answer.content.strip()

    # 直接走模型时产出的是 AIMessageChunk；应用内的 RAG 链路在 LLM 后面接了
    # StrOutputParser，所以 SSE 拿到的是字符串（见 services/rag.py::_build_chain）
    streamed = ""
    async for piece in model.astream("用一句话回答：2+2 等于几？"):
        assert isinstance(piece, AIMessageChunk)
        assert isinstance(piece.content, str)
        streamed += piece.content
    assert streamed.strip()


async def test_live_embeddings_return_consistent_dimensions(openrouter: None) -> None:
    embeddings = get_embeddings()
    texts = ["缓存分片会被回收。", "齿轮与量子无关。"]
    vectors = await embeddings.aembed_documents(texts)
    assert len(vectors) == len(texts)
    dims = {len(vector) for vector in vectors}
    assert len(dims) == 1 and dims.pop() > 0
    assert all(isinstance(value, float) for value in vectors[0])


async def test_live_deepseek_chat_and_health(deepseek: None, client: TestClient) -> None:
    """DeepSeek 走官方集成：真能回答，且 /health 不会把模型名变化误报成 degraded。"""
    answer = await get_chat_model().ainvoke("只回复两个字：收到")
    assert answer.content.strip()

    health = client.get("/api/system/health").json()
    assert health["status"] == "healthy", health
    assert health["llm"]["provider"] == "deepseek"
    assert health["llm"]["ok"] is True
    assert health["llm"]["error"] is None


def test_live_health_reports_healthy(openrouter: None, client: TestClient) -> None:
    body = client.get("/api/system/health").json()
    assert body["status"] == "healthy", body
    assert body["llm"]["provider"] == "openrouter"
    assert body["embedding"]["provider"] == "openrouter"


def test_live_acceptance_upload_ask_and_follow_up(
    openrouter: None, client: TestClient, kb: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """docs/samples/acceptance.txt 的判定标准 1~5（检索阈值置 0，专注验证链路）。"""
    monkeypatch.setattr(settings, "RETRIEVAL_SCORE_THRESHOLD", 0.0)

    content = ACCEPTANCE_FILE.read_text(encoding="utf-8")
    uploaded = client.post(
        "/api/doc/upload",
        data={"kb_id": str(kb["id"])},
        files={"files": ("acceptance.txt", content.encode("utf-8"), "text/plain")},
    )
    assert uploaded.status_code == 200, uploaded.text
    doc_id = uploaded.json()["data"]["doc_ids"][0]

    # 2. 文档处理完成
    doc = client.get(f"/api/doc/{doc_id}").json()["data"]
    assert doc["status"] == "completed", doc
    assert doc["chunk_count"] > 0

    # 3~4. 流式问答：sources 指向本文档，done 的答案来自文档内容
    streamed = client.post(
        "/api/chat/stream",
        json={
            "kb_id": kb["id"],
            "question": "紫罗兰色的量子齿轮会在每月第三个星期二回收多少缓存分片？",
        },
    )
    assert streamed.status_code == 200, streamed.text
    events = _parse_sse(streamed.text)
    kinds = [event["type"] for event in events]
    assert "sources" in kinds and "done" in kinds, events

    sources = next(event["data"] for event in events if event["type"] == "sources")
    assert sources, "检索没有命中任何参考片段"
    assert sources[0]["filename"] == "acceptance.txt"
    assert sources[0]["score"] is None or sources[0]["score"] > 0

    done = next(event["data"] for event in events if event["type"] == "done")
    assert "三分之二" in done, done
    conv_id = next(event["data"] for event in events if event["type"] == "conv_id")

    # 5. 追问依赖多轮历史
    follow_up = client.post(
        "/api/chat/send",
        json={"kb_id": kb["id"], "conv_id": conv_id, "question": "第一句里说的是什么颜色的齿轮？"},
    )
    assert follow_up.status_code == 200, follow_up.text
    answer = follow_up.json()["data"]["message"]["content"]
    assert "紫罗兰" in answer, answer
