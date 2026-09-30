"""pytest 全局配置。

所有测试都在离线环境运行：
* embedding 用确定性的假实现替换，不访问 Ollama；
* LLM 用 RunnableLambda 替换，不访问网络；
* 数据库用临时 SQLite 文件（与开发默认一致，部署时可换 PostgreSQL）。

鉴权说明：Phase 3 起所有业务接口都需要登录，因此 ``client`` / ``other_client`` 是
两个**已登录**的客户端（各自对应一个新建用户），未登录场景用 ``anonymous_client``。
账号没有注册接口，测试里直接写库造账号（与运维脚本 create_user.py 做的事一致）。
"""

from __future__ import annotations

import asyncio
import math
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

_TMP = Path(tempfile.mkdtemp(prefix="inner-rag-tests-"))
os.environ.update(
    {
        "DATABASE_URL": f"sqlite:///{_TMP / 'test.db'}",
        "AUTO_CREATE_TABLES": "true",
        "UPLOAD_DIR": str(_TMP / "uploads"),
        "ZVEC_PATH": str(_TMP / "zvec"),
        "CHROMA_PERSIST_DIR": str(_TMP / "chroma"),
        "LOG_DIR": str(_TMP / "logs"),
        "OCR_BACKEND": "none",
        "LOG_RETRIEVAL": "false",
        "LOG_PROMPT": "false",
        # provider 与向量库配置必须与开发者本机的 .env 解耦（os.environ 优先于 .env）：
        # 否则本地把 provider 切到云端（或打开截断）、把 VECTOR_STORE 换成别的后端，
        # 「离线测试」就会随本机配置漂移，甚至真的发出网络请求。
        # 需要其他 provider / 后端的用例自行 monkeypatch settings。
        "LLM_PROVIDER": "mock",
        "EMBEDDING_PROVIDER": "mock",
        "EMBEDDING_MAX_INPUT_CHARS": "0",
        "VECTOR_STORE": "zvec",
        # 追踪与日志格式同理：开发者本机开了 LangSmith 也不能让离线用例联网
        "LANGSMITH_TRACING": "false",
        "LOG_FORMAT": "text",
    }
)

# ruff: noqa: E402  —— 环境变量必须在上面的 import 之前设好
import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from inner_rag.core.database import SessionLocal, init_db
from inner_rag.core.security import hash_password
from inner_rag.main import app
from inner_rag.models import User
from inner_rag.services.cache import embedding_cache, query_cache
from inner_rag.services.embedding import embedding_service
from inner_rag.services.rag import rag_service
from inner_rag.services.vector_store import vector_service

FAKE_DIM = 64
TEST_PASSWORD = "test-password"


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


def create_user(
    username: str | None = None,
    *,
    password: str = TEST_PASSWORD,
    is_active: bool = True,
    display_name: str = "测试用户",
) -> User:
    """造一个账号。登录名默认随机，避免用例之间撞唯一索引。"""
    db = SessionLocal()
    try:
        user = User(
            username=username or f"user-{uuid4().hex[:8]}",
            display_name=display_name,
            password_hash=hash_password(password),
            is_active=is_active,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    finally:
        db.close()


def set_user_active(user_id: int, active: bool) -> None:
    """直接改库：模拟「账号被停用后，已签发的 token 应当失效」。"""
    db = SessionLocal()
    try:
        db.get(User, user_id).is_active = active  # type: ignore[union-attr]
        db.commit()
    finally:
        db.close()


def bearer(token: str) -> dict[str, str]:
    """构造 Authorization 头（用例里临时换 token 时用）。"""
    return {"Authorization": f"Bearer {token}"}


def login(client: TestClient, user: User, password: str = TEST_PASSWORD) -> str:
    """登录并返回 access_token。"""
    response = client.post(
        "/api/auth/login", json={"username": user.username, "password": password}
    )
    assert response.status_code == 200, response.text
    return response.json()["data"]["access_token"]


# ── fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def offline_providers(request: pytest.FixtureRequest) -> Iterator[None]:
    """把 embedding / LLM 换成离线假实现，并清空缓存，保证测试互不影响。

    带 ``@pytest.mark.live`` 的用例是真实 provider 冒烟测试，跳过替换，
    否则它们会拿假模型去验证真 API。
    """
    if request.node.get_closest_marker("live") is not None:
        query_cache.clear()
        embedding_cache.clear()
        yield
        query_cache.clear()
        embedding_cache.clear()
        return

    original_embeddings = embedding_service._embeddings
    original_get_llm = rag_service._get_llm
    embedding_service._embeddings = FakeEmbeddings()
    rag_service._get_llm = fake_llm  # type: ignore[method-assign]
    query_cache.clear()
    embedding_cache.clear()
    yield
    embedding_service._embeddings = original_embeddings
    rag_service._get_llm = original_get_llm  # type: ignore[method-assign]
    query_cache.clear()
    embedding_cache.clear()


@pytest.fixture(scope="session", autouse=True)
def database() -> None:
    # 测试库用 create_all（AUTO_CREATE_TABLES=true）建表，避免测试依赖 alembic 迁移
    init_db()


@pytest.fixture
def user() -> User:
    return create_user()


@pytest.fixture
def other_user() -> User:
    return create_user()


@pytest.fixture
def anonymous_client() -> Iterator[TestClient]:
    """未登录客户端（验证 401 / 公开接口用）。"""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def client(user: User) -> Iterator[TestClient]:
    """已登录客户端：多数用例都用它，测试结束顺手清掉它自己的知识库。"""
    with TestClient(app) as test_client:
        test_client.headers.update(bearer(login(test_client, user)))
        yield test_client
        for item in test_client.get("/api/kb", params={"page_size": 100}).json()["data"]["items"]:
            test_client.delete(f"/api/kb/{item['id']}")


@pytest.fixture
def other_client(other_user: User) -> Iterator[TestClient]:
    """另一个用户的已登录客户端（验证跨库 403 隔离用）。"""
    with TestClient(app) as test_client:
        test_client.headers.update(bearer(login(test_client, other_user)))
        yield test_client


@pytest.fixture
def kb(client: TestClient) -> Iterator[dict]:
    response = client.post("/api/kb", json={"name": "pytest-kb", "description": "测试知识库"})
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    yield data
    # 同步 fixture 没有事件循环可用，直接跑协程回收向量（此时循环未在运行）
    asyncio.run(vector_service.delete_kb(data["id"]))


@pytest.fixture
def doc_id(client: TestClient, kb: dict) -> int:
    """在 kb 里落一个已处理完成的文档（验证文档级权限用）。"""
    response = client.post(
        "/api/doc/upload",
        data={"kb_id": str(kb["id"])},
        files={"files": ("note.txt", "知识库测试文档内容。".encode(), "text/plain")},
    )
    assert response.status_code == 200, response.text
    return response.json()["data"]["doc_ids"][0]
