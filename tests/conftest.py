"""pytest 全局配置。

所有测试都在离线环境运行：
* embedding 用确定性的假实现替换，不访问 Ollama；
* LLM 用 RunnableLambda 替换，不访问网络；
* 数据库用临时 SQLite 文件（与开发默认一致，部署时可换 PostgreSQL）。
"""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="inner-rag-tests-"))
os.environ.update(
    {
        "DATABASE_URL": f"sqlite:///{_TMP / 'test.db'}",
        "AUTO_CREATE_TABLES": "true",
        "UPLOAD_DIR": str(_TMP / "uploads"),
        "CHROMA_PERSIST_DIR": str(_TMP / "chroma"),
        "LOG_DIR": str(_TMP / "logs"),
        "OCR_BACKEND": "none",
        "LOG_RETRIEVAL": "false",
        "LOG_PROMPT": "false",
        # provider 配置必须与开发者本机的 .env 解耦（os.environ 优先于 .env）：
        # 否则本地一把 provider 切到云端（或打开截断），「离线测试」就会随本机配置漂移，
        # 甚至真的发出网络请求。需要其他 provider 的用例自行 monkeypatch settings。
        "LLM_PROVIDER": "mock",
        "EMBEDDING_PROVIDER": "mock",
        "EMBEDDING_MAX_INPUT_CHARS": "0",
    }
)

# ruff: noqa: E402  —— 环境变量必须在上面的 import 之前设好
import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from inner_rag.core.database import init_db
from inner_rag.main import app
from inner_rag.services.cache import embedding_cache, query_cache
from inner_rag.services.embedding import embedding_service
from inner_rag.services.rag import rag_service
from inner_rag.services.vector_store import vector_service

FAKE_DIM = 64


class FakeEmbeddings(Embeddings):
    """确定性字符哈希 embedding：语义无用，但可复现，足以验证链路。"""

    def __init__(self, dim: int = FAKE_DIM) -> None:
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for char in text:
            vec[ord(char) % self.dim] += 1.0
        norm = math.sqrt(sum(value * value for value in vec)) or 1.0
        return [value / norm for value in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


FAKE_ANSWER = "这是测试模型的回答。"


def fake_llm() -> RunnableLambda:
    """替代 ChatOllama：支持 ainvoke / astream，不产生任何网络请求。"""
    return RunnableLambda(lambda _: AIMessage(content=FAKE_ANSWER))


@pytest.fixture(autouse=True)
def offline_providers(request: pytest.FixtureRequest) -> Iterator[None]:
    """把 embedding / LLM 换成离线假实现，并清空缓存，保证测试互不影响。

    带 ``@pytest.mark.live`` 的用例是真实 provider 冒烟测试，跳过替换，
    否则它们会拿假模型去验证真 API。
    """
    if request.node.get_closest_marker("live") is not None:
        query_cache._cache.clear()
        embedding_cache._cache.clear()
        yield
        query_cache._cache.clear()
        embedding_cache._cache.clear()
        return

    original_embeddings = embedding_service._embeddings
    original_get_llm = rag_service._get_llm
    embedding_service._embeddings = FakeEmbeddings()
    rag_service._get_llm = fake_llm  # type: ignore[method-assign]
    query_cache._cache.clear()
    embedding_cache._cache.clear()
    yield
    embedding_service._embeddings = original_embeddings
    rag_service._get_llm = original_get_llm  # type: ignore[method-assign]
    query_cache._cache.clear()
    embedding_cache._cache.clear()


@pytest.fixture(scope="session", autouse=True)
def database() -> None:
    # 测试库用 create_all（AUTO_CREATE_TABLES=true）建表，避免测试依赖 alembic 迁移
    init_db()


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client
        # 测试结束顺手清库，避免用例之间互相影响
        for item in test_client.get("/api/kb", params={"page_size": 100}).json()["data"]["items"]:
            test_client.delete(f"/api/kb/{item['id']}")


@pytest.fixture
def kb(client: TestClient) -> Iterator[dict]:
    response = client.post("/api/kb", json={"name": "pytest-kb", "description": "测试知识库"})
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    yield data
    vector_service.delete_kb(data["id"])
