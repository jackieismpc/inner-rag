"""Phase 2：多 provider 抽象层测试（全部离线，不发任何网络请求）。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from inner_rag.core.config import Settings, settings
from inner_rag.providers import (
    ProviderError,
    chat_health,
    chat_models,
    get_chat_model,
    get_embeddings,
    provider_catalog,
    reset_cache,
)
from inner_rag.providers.chat import MockChatModel
from inner_rag.providers.embeddings import MockEmbeddings, TruncatingEmbeddings
from inner_rag.providers.specs import chat_spec, embedding_spec, normalize
from inner_rag.services.cache import embedding_cache
from inner_rag.services.rag import rag_service


@pytest.fixture(autouse=True)
def clean_provider_cache():
    """每个用例都从干净的 provider 缓存开始，避免相互污染。"""
    reset_cache()
    yield
    reset_cache()


def _use_openrouter(monkeypatch: pytest.MonkeyPatch, key: str = "sk-test-key") -> None:
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", key)
    monkeypatch.setattr(settings, "LLM_PROVIDER", "openrouter")
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "openrouter")
    reset_cache()


def _use_deepseek(monkeypatch: pytest.MonkeyPatch, key: str = "sk-deepseek") -> None:
    monkeypatch.setattr(settings, "DEEPSEEK_API_KEY", key)
    monkeypatch.setattr(settings, "LLM_PROVIDER", "deepseek")
    reset_cache()


# ── provider 解析 ──────────────────────────────────────────────────────


def test_provider_name_is_normalized() -> None:
    assert normalize("  OpenRouter ") == "openrouter"
    assert normalize("LOCAL") == "ollama"
    assert normalize("fake") == "mock"


def test_unknown_provider_raises_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "LLM_PROVIDER", "gemini")
    with pytest.raises(ProviderError) as excinfo:
        get_chat_model()
    message = str(excinfo.value)
    assert "gemini" in message
    assert "LLM_PROVIDER" in message
    assert "openrouter" in message  # 提示可选值


def test_missing_api_key_names_the_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "LLM_PROVIDER", "openrouter")
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "")
    with pytest.raises(ProviderError) as excinfo:
        get_chat_model()
    message = str(excinfo.value)
    assert "OPENROUTER_API_KEY" in message
    assert ".env" in message


def test_deepseek_has_no_embedding_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """DeepSeek 只有 chat completion：必须给出明确解释，而不是 404/500。"""
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "deepseek")
    with pytest.raises(ProviderError) as excinfo:
        get_embeddings()
    message = str(excinfo.value)
    assert "deepseek" in message
    assert "embedding" in message
    # 但同一个 deepseek 作为 chat provider 是合法的
    monkeypatch.setattr(settings, "DEEPSEEK_API_KEY", "sk-ds")
    assert chat_spec("deepseek").model == settings.DEEPSEEK_CHAT_MODEL


def test_provider_name_is_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "sk-test")
    spec = chat_spec("OpenAI")
    assert spec.name == "openai"
    assert spec.configured


def test_chat_and_embedding_providers_are_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """chat 走云端、embedding 走本地是常见组合，两者必须能各自配置。"""
    monkeypatch.setattr(settings, "LLM_PROVIDER", "mock")
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setattr(settings, "OLLAMA_BASE_URL", "http://ollama.local:11434")
    monkeypatch.setattr(settings, "OLLAMA_EMBEDDING_MODEL", "bge-m3")

    assert isinstance(get_chat_model(), MockChatModel)
    embeddings = get_embeddings()
    assert isinstance(embeddings, OllamaEmbeddings)
    assert embeddings.model == "bge-m3"


# ── 参数映射 ───────────────────────────────────────────────────────────


def test_chat_model_params_are_mapped(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_openrouter(monkeypatch)
    monkeypatch.setattr(settings, "OPENROUTER_CHAT_MODEL", "some-org/some-model")
    monkeypatch.setattr(settings, "LLM_TEMPERATURE", 0.11)
    monkeypatch.setattr(settings, "LLM_MAX_TOKENS", 123)
    reset_cache()

    model = get_chat_model()
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "some-org/some-model"
    assert str(model.openai_api_base).startswith(settings.OPENROUTER_BASE_URL)
    assert model.temperature == 0.11


def test_ollama_chat_params_are_mapped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(settings, "OLLAMA_BASE_URL", "http://127.0.0.1:12345")
    monkeypatch.setattr(settings, "OLLAMA_LLM_MODEL", "qwen3:14b")
    reset_cache()

    model = get_chat_model()
    assert isinstance(model, ChatOllama)
    assert model.model == "qwen3:14b"
    assert str(model.base_url) == "http://127.0.0.1:12345"


def test_deepseek_uses_official_langchain_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """DeepSeek 走官方 langchain-deepseek：字段名与 OpenAI 兼容层不同。"""
    _use_deepseek(monkeypatch)
    monkeypatch.setattr(settings, "DEEPSEEK_CHAT_MODEL", "deepseek-flash")
    monkeypatch.setattr(settings, "LLM_MAX_TOKENS", 512)
    reset_cache()

    model = get_chat_model()
    assert isinstance(model, ChatDeepSeek)
    assert model.model_name == "deepseek-flash"
    assert str(model.api_base) == settings.DEEPSEEK_BASE_URL
    # 官方集成用 max_tokens，而不是兼容层的 max_completion_tokens
    assert model.max_tokens == 512
    assert model.temperature == settings.LLM_TEMPERATURE
    assert model.max_retries == settings.LLM_MAX_RETRIES


def test_deepseek_default_model_matches_official_docs() -> None:
    """默认模型名必须与官方文档一致：deepseek-chat 已从 /models 列表下架。"""
    assert Settings.model_fields["DEEPSEEK_CHAT_MODEL"].default == "deepseek-flash"


def test_deepseek_reasoning_effort_is_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_deepseek(monkeypatch)
    monkeypatch.setattr(settings, "LLM_REASONING_EFFORT", "   ")
    reset_cache()
    model = get_chat_model()
    assert isinstance(model, ChatDeepSeek)
    assert model.reasoning_effort is None  # 留空表示不下发该参数

    monkeypatch.setattr(settings, "LLM_REASONING_EFFORT", "high")
    reset_cache()
    model = get_chat_model()
    assert isinstance(model, ChatDeepSeek)
    assert model.reasoning_effort == "high"


def test_true_cloud_embedding_uses_openai_compatible_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_openrouter(monkeypatch)
    monkeypatch.setattr(
        settings, "OPENROUTER_EMBEDDING_MODEL", "liquid/lfm-2.5-embedding-350m:free"
    )
    reset_cache()

    embeddings = get_embeddings()
    assert isinstance(embeddings, OpenAIEmbeddings)
    assert embeddings.model == "liquid/lfm-2.5-embedding-350m:free"
    # 非 OpenAI 官方模型不能走 tiktoken 上下文校验
    assert embeddings.check_embedding_ctx_length is False


def test_provider_instances_are_reused() -> None:
    """同一份配置不应每个请求都重建客户端。"""
    assert get_chat_model() is get_chat_model()
    assert get_embeddings() is get_embeddings()


def test_embedding_truncation_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    class RecordingEmbeddings(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            seen.extend(texts)
            return [[float(len(text))] for text in texts]

        def embed_query(self, text: str) -> list[float]:
            seen.append(text)
            return [float(len(text))]

    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "mock")
    monkeypatch.setattr(settings, "EMBEDDING_MAX_INPUT_CHARS", 8)
    reset_cache()

    wrapper = get_embeddings()
    assert isinstance(wrapper, TruncatingEmbeddings)
    wrapper._inner = RecordingEmbeddings()
    wrapper.embed_documents(["0123456789", "short"])
    wrapper.embed_query("0123456789")
    assert seen == ["01234567", "short", "01234567"]


# ── mock provider 契约 ─────────────────────────────────────────────────


async def test_mock_chat_model_answers_and_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "LLM_PROVIDER", "mock")
    reset_cache()
    model = get_chat_model()

    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content="参考文档：\n[1] a\n\n---\n\n[2] b"),
        HumanMessage(content="问题？"),
    ]
    answer = await model.ainvoke(messages)
    assert "2 段参考" in answer.content
    assert "问题？" in answer.content

    chunks = [chunk.content async for chunk in model.astream(messages)]
    assert len(chunks) > 1  # 真的是分片流式，而不是一次性返回
    assert "".join(chunks) == answer.content


def test_mock_embeddings_are_deterministic_and_word_overlapping() -> None:
    embeddings = MockEmbeddings()
    assert embeddings.dim == settings.MOCK_EMBEDDING_DIM
    assert len(embeddings.embed_query("缓存分片")) == settings.MOCK_EMBEDDING_DIM
    # 同样输入必须完全一致（可复现）
    assert embeddings.embed_query("缓存分片") == embeddings.embed_query("缓存分片")

    def cosine(left: list[float], right: list[float]) -> float:
        return sum(a * b for a, b in zip(left, right, strict=True))

    related = cosine(
        embeddings.embed_query("缓存分片回收策略"), embeddings.embed_query("缓存分片回收")
    )
    unrelated = cosine(
        embeddings.embed_query("缓存分片回收策略"), embeddings.embed_query("今天天气很好")
    )
    assert related > unrelated


def test_mock_provider_makes_zero_network_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "LLM_PROVIDER", "mock")
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "mock")
    reset_cache()

    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("mock provider 不应发起任何网络请求")

    monkeypatch.setattr("httpx.get", boom)
    monkeypatch.setattr("httpx.post", boom)

    assert chat_health()["ok"] is True
    assert chat_models()["models"] == [settings.MOCK_CHAT_MODEL]
    assert provider_catalog()["embedding"][-1]["name"] == "mock"


# ── embedding 身份与缓存隔离 ───────────────────────────────────────────


def test_embedding_key_is_provider_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_openrouter(monkeypatch)
    assert settings.embedding_key == (f"openrouter:{settings.OPENROUTER_EMBEDDING_MODEL}")
    assert embedding_spec().identity == settings.embedding_key


def test_embedding_cache_is_isolated_per_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "mock")
    first = embedding_cache._make_key("同一段文本")
    monkeypatch.setattr(settings, "MOCK_EMBEDDING_MODEL", "another-embedding")
    second = embedding_cache._make_key("同一段文本")
    assert first != second


# ── API 层：健康检查 / provider 目录 / 降级 ────────────────────────────


def test_health_reports_config_error_without_5xx(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "LLM_PROVIDER", "openrouter")
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "")
    reset_cache()

    response = client.get("/api/system/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["llm"]["ok"] is False
    assert "OPENROUTER_API_KEY" in body["llm"]["error"]
    # embedding 不受 chat 配置影响（conftest 固定为 mock，因此这里恒为 ok）
    assert body["embedding"]["ok"] is True


def test_health_is_ok_with_mock_provider(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "LLM_PROVIDER", "mock")
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "mock")
    reset_cache()

    body = client.get("/api/system/health").json()
    assert body["status"] == "healthy"
    assert body["llm"]["provider"] == "mock"
    assert body["llm"]["model"] == settings.MOCK_CHAT_MODEL
    assert body["embedding"]["provider"] == "mock"


def test_health_flags_missing_model_for_authoritative_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """模型列表可信的 provider：模型不在列表里就是配置错误，并给出当前可选值。"""
    _use_openrouter(monkeypatch)
    monkeypatch.setattr(settings, "OPENROUTER_CHAT_MODEL", "ghost/model")
    monkeypatch.setattr(
        "inner_rag.providers.factory._discover_models",
        lambda spec: (["some-org/some-model"], None),
    )
    reset_cache()

    health = chat_health()
    assert health["ok"] is False
    assert health["model_available"] is False
    assert "ghost/model" in health["error"]
    assert "some-org/some-model" in health["error"]
    assert health["warning"] is None


def test_health_warns_instead_of_failing_when_model_list_is_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DeepSeek 的 /models 只列主推模型，旧别名仍可调用：告警，但不判 degraded。"""
    _use_deepseek(monkeypatch)
    monkeypatch.setattr(settings, "DEEPSEEK_CHAT_MODEL", "deepseek-chat")
    monkeypatch.setattr(
        "inner_rag.providers.factory._discover_models",
        lambda spec: (["deepseek-flash", "deepseek-v4-pro"], None),
    )
    reset_cache()

    assert chat_spec().model_list_authoritative is False
    health = chat_health()
    assert health["ok"] is True
    assert health["model_available"] is False
    assert health["error"] is None
    assert "deepseek-chat" in health["warning"]
    assert "deepseek-flash" in health["warning"]


def test_health_endpoint_is_healthy_for_deepseek_default_model(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DeepSeek chat + 云端 embedding 这套真实组合应当是 healthy。"""
    _use_deepseek(monkeypatch)
    monkeypatch.setattr(settings, "DEEPSEEK_CHAT_MODEL", "deepseek-flash")
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "mock")
    monkeypatch.setattr(
        "inner_rag.providers.factory._discover_models",
        lambda spec: (["deepseek-flash", "deepseek-v4-pro"], None),
    )
    reset_cache()

    body = client.get("/api/system/health").json()
    assert body["status"] == "healthy"
    assert body["llm"]["provider"] == "deepseek"
    assert body["llm"]["model"] == "deepseek-flash"
    assert body["llm"]["model_available"] is True
    assert body["llm"]["warning"] is None


def test_providers_endpoint_lists_capabilities_without_secrets(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "sk-super-secret-value"
    monkeypatch.setattr(settings, "LLM_PROVIDER", "openrouter")
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", secret)
    reset_cache()

    response = client.get("/api/system/providers")
    assert response.status_code == 200
    assert secret not in response.text  # 绝不回显密钥

    body = response.json()
    assert body["active"] == {"llm": "openrouter", "embedding": settings.EMBEDDING_PROVIDER}

    chat_names = [item["name"] for item in body["data"]["chat"]]
    embedding_names = [item["name"] for item in body["data"]["embedding"]]
    assert chat_names == ["ollama", "openrouter", "deepseek", "openai", "mock"]
    assert "deepseek" not in embedding_names
    active = next(item for item in body["data"]["chat"] if item["name"] == "openrouter")
    assert active["active"] is True
    assert active["configured"] is True
    assert active["api_key_env"] == "OPENROUTER_API_KEY"
    assert active["model"] == settings.OPENROUTER_CHAT_MODEL


def test_config_endpoint_exposes_active_models(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "LLM_PROVIDER", "mock")
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "mock")
    reset_cache()

    body = client.get("/api/system/config").json()
    assert body["llm_provider"] == "mock"
    assert body["llm_model"] == settings.MOCK_CHAT_MODEL
    assert body["embedding_model"] == (
        f"{settings.MOCK_EMBEDDING_MODEL}-{settings.MOCK_EMBEDDING_DIM}d"
    )
    assert body["embedding_key"] == settings.embedding_key


def test_models_endpoint_follows_active_provider(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "LLM_PROVIDER", "mock")
    reset_cache()

    body = client.get("/api/system/models").json()
    assert body["provider"] == "mock"
    assert body["models"] == [settings.MOCK_CHAT_MODEL]
    assert body["error"] is None


def _upload_text(client: TestClient, kb_id: int, content: str = "缓存分片与检索策略") -> int:
    response = client.post(
        "/api/doc/upload",
        data={"kb_id": str(kb_id)},
        files={"files": ("note.txt", content.encode(), "text/plain")},
    )
    assert response.status_code == 200, response.text
    return response.json()["data"]["doc_ids"][0]


def test_mock_provider_end_to_end_upload_and_ask(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """降级路径：全部走 mock，也应该能完成「上传 -> 检索 -> 回答 + 引用」。"""
    monkeypatch.setattr(settings, "LLM_PROVIDER", "mock")
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "mock")
    monkeypatch.setattr(settings, "RETRIEVAL_SCORE_THRESHOLD", 0.0)
    # conftest 默认用假 LLM 保证离线；这里要验证真实 provider 工厂链路
    monkeypatch.setattr(rag_service, "_get_llm", get_chat_model)
    reset_cache()

    # 知识库会锁定 embedding 身份，所以必须在切换 provider 之后再建库
    created = client.post("/api/kb", json={"name": "mock-kb", "description": "离线 mock"})
    kb = created.json()["data"]
    assert kb["embedding_model"] == settings.embedding_key

    _upload_text(client, kb["id"])
    response = client.post(
        "/api/chat/send", json={"kb_id": kb["id"], "question": "缓存分片怎么回收？"}
    )
    assert response.status_code == 200, response.text

    message = response.json()["data"]["message"]
    assert message["content"].startswith("[mock]")
    assert message["sources"]
    assert message["sources"][0]["filename"] == "note.txt"


def test_chat_send_returns_503_on_misconfigured_provider(
    client: TestClient, kb: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 先造出可命中的内容：chat() 在检索为空时会直接返回「未找到」，不碰模型
    monkeypatch.setattr(settings, "RETRIEVAL_SCORE_THRESHOLD", 0.0)
    _upload_text(client, kb["id"])

    monkeypatch.setattr(settings, "LLM_PROVIDER", "openrouter")
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "")
    monkeypatch.setattr(rag_service, "_get_llm", get_chat_model)
    reset_cache()

    response = client.post(
        "/api/chat/send", json={"kb_id": kb["id"], "question": "缓存分片怎么回收？"}
    )
    assert response.status_code == 503, response.text
    detail = response.json()["detail"]
    assert "OPENROUTER_API_KEY" in detail and ".env" in detail


def test_upload_surfaces_misconfigured_embedding(
    client: TestClient, kb: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """embedding provider 没配好时，文档应显式失败并给出可读原因。"""
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "deepseek")
    reset_cache()

    doc_id = _upload_text(client, kb["id"])
    doc = client.get(f"/api/doc/{doc_id}").json()["data"]
    assert doc["status"] == "failed"
    assert "embedding" in doc["error_msg"]
