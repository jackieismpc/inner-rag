"""插件注册表测试：第三方后端「只加实现 + 配置」就能跑通的证明。

覆盖三类失效场景（其余行为由各插件点自己的契约测试覆盖）：

1. 内置实现之间撞名 —— 属于编码错误，必须显式失败而不是静默覆盖；
2. 第三方 entry point 加载失败 —— 只告警并跳过，不能让服务起不来；
3. 第三方 entry point 与内置同名 —— 保留内置实现，防止安装一个包就悄悄换掉后端。
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from inner_rag.core.config import settings
from inner_rag.plugins.registry import (
    ALL_REGISTRIES,
    ChatProvider,
    PluginError,
    Registry,
    chat_providers,
    plugin_status,
)
from inner_rag.providers import get_chat_model, provider_catalog, reset_cache
from inner_rag.providers.specs import ProviderSpec, chat_spec
from inner_rag.services.rag import rag_service

DUMMY_ANSWER = "[dummy] 第三方 provider 的回答"


def _dummy_spec() -> ProviderSpec:
    return ProviderSpec(
        name="dummy", label="Dummy（测试用第三方 provider）", kind="chat", model="dummy-chat"
    )


class DummyChatModel(BaseChatModel):
    """最小可用的第三方 chat 后端：不联网、固定回答。"""

    @property
    def _llm_type(self) -> str:
        return "dummy-chat"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=DUMMY_ANSWER))])


def _dummy_plugin() -> ChatProvider:
    return ChatProvider(spec=_dummy_spec, build=lambda spec: DummyChatModel())


# ── 注册表本身 ─────────────────────────────────────────────────────────


def test_duplicate_registration_is_rejected() -> None:
    registry: Registry[str] = Registry("test", "test backend", "test.group", "TEST_BACKEND")
    registry.register("a", "first")

    with pytest.raises(PluginError) as excinfo:
        registry.register("a", "second")
    assert "test backend" in str(excinfo.value)
    assert registry.get("a") == "first"  # 原来的实现没被覆盖


class _FakeEntryPoint:
    def __init__(self, name: str, impl: Any) -> None:
        self.name = name
        self._impl = impl

    def load(self) -> Any:
        if isinstance(self._impl, Exception):
            raise self._impl
        return self._impl


def test_entry_point_failure_is_skipped_and_duplicate_keeps_builtin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """第三方包坏了只告警；与内置同名时以内置为准。"""
    registry: Registry[str] = Registry("test", "test backend", "test.group", "TEST_BACKEND")
    registry.register("builtin", "内置实现")
    monkeypatch.setattr(
        "inner_rag.plugins.registry.entry_points",
        lambda group: [
            _FakeEntryPoint("builtin", "第三方实现"),
            _FakeEntryPoint("broken", ImportError("依赖没装")),
            _FakeEntryPoint("third", "第三方实现"),
        ],
    )

    assert registry.load_entry_points() == ["third"]
    assert registry.get("builtin") == "内置实现"
    assert registry.get("third") == "第三方实现"
    assert registry.get("broken") is None
    assert registry.third_party_names() == ("third",)
    # 幂等：重复调用不会再加载一遍（也不会把 third 重复注册进去）
    assert registry.load_entry_points() == []
    assert registry.third_party_names() == ("third",)


def test_configured_name_reads_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """插件点自己知道「哪个配置项驱动它」：接口与报错都不用再维护第二张映射表。"""
    registry: Registry[str] = Registry("x", "x backend", "x.group", "VECTOR_STORE")
    registry.register("memory", "impl")

    monkeypatch.setattr(settings, "VECTOR_STORE", "Memory")
    assert registry.configured_name() == "memory"  # 大小写归一，与 build_* 的解析口径一致
    assert registry.is_active() is True

    monkeypatch.setattr(settings, "VECTOR_STORE", "no-such-backend")
    assert registry.is_active() is False


def test_plugin_status_covers_every_registry() -> None:
    report = plugin_status()
    assert set(report) == {registry.key for registry in ALL_REGISTRIES}
    # 显式列出全部插件点：这行才能抓住「某个 registry 从 ALL_REGISTRIES 里被删掉」，
    # 光比对 ALL_REGISTRIES 自身抓不住（两边一起少，断言照样过）。
    assert set(report) == {
        "chat",
        "embedding",
        "vector_store",
        "cache",
        "task_queue",
        "rerank",
        "query_rewrite",
    }

    vector_store = report["vector_store"]
    assert vector_store["settings_key"] == "VECTOR_STORE"
    assert vector_store["configured"] == settings.VECTOR_STORE.strip().lower()
    assert vector_store["active"] is True
    assert "memory" in vector_store["available"]
    assert vector_store["entry_point_group"] == "inner_rag.vector_stores"
    assert vector_store["third_party"] == []

    # 每个插件点的配置项都真实存在（写错配置项名会让 active 永远为 False），
    # 且至少注册了一个实现（注册语句在实现模块的导入期执行，所以这条也在守「模块被导入了吗」）
    for entry in report.values():
        assert hasattr(settings, entry["settings_key"]), entry["settings_key"]
        assert entry["available"], f"{entry['settings_key']} 没有任何已注册实现"


# ── 端到端：不改业务代码地接入一个第三方 provider ──────────────────────


def _upload(client: TestClient, kb_id: int) -> None:
    response = client.post(
        "/api/doc/upload",
        data={"kb_id": str(kb_id)},
        files={"files": ("note.txt", "第三方 provider 演练内容。".encode(), "text/plain")},
    )
    assert response.status_code == 200, response.text


def test_third_party_chat_provider_works_without_business_code_change(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """只注册实现（spec + 构造器）+ 改一行 .env，问答链路就能用上它。"""
    # 直接往注册表里塞一个第三方实现（等价于第三方包用 entry point 注册，见上一个用例）
    monkeypatch.setitem(chat_providers._impls, "dummy", _dummy_plugin())
    monkeypatch.setattr(settings, "LLM_PROVIDER", "dummy")
    monkeypatch.setattr(settings, "RETRIEVAL_SCORE_THRESHOLD", 0.0)
    # conftest 默认把 _get_llm 换成假模型；这里要验证真实工厂链路
    monkeypatch.setattr(rag_service, "_get_llm", get_chat_model)
    reset_cache()

    # 1) 解析与目录：provider 名字、可选列表、third_party 标记都随注册表变化
    spec = chat_spec()
    assert spec.name == "dummy" and spec.identity == "dummy:dummy-chat"
    dummy_entry = next(item for item in provider_catalog()["chat"] if item["name"] == "dummy")
    assert dummy_entry["third_party"] is True
    assert dummy_entry["active"] is True

    # 2) 真实链路：建库 -> 上传 -> 提问，回答来自第三方后端
    kb = client.post("/api/kb", json={"name": "dummy-kb"}).json()["data"]
    _upload(client, kb["id"])
    response = client.post("/api/chat/send", json={"kb_id": kb["id"], "question": "演练问题"})

    assert response.status_code == 200, response.text
    message = response.json()["data"]["message"]
    assert message["content"] == DUMMY_ANSWER
    assert message["sources"]

    reset_cache()


# ── 状态接口：插件点必须能从外部看见 ───────────────────────────────────


def test_plugins_endpoint_reports_every_plugin_point(client: TestClient) -> None:
    response = client.get("/api/system/plugins")
    assert response.status_code == 200, response.text
    data = response.json()["data"]

    assert set(data) == {
        "chat",
        "embedding",
        "vector_store",
        "cache",
        "task_queue",
        "rerank",
        "query_rewrite",
    }
    assert data["vector_store"]["configured"] == settings.VECTOR_STORE.strip().lower()
    assert data["cache"]["configured"] == settings.CACHE_BACKEND.strip().lower()
    assert data["task_queue"]["configured"] == settings.TASK_QUEUE_BACKEND.strip().lower()
    assert data["rerank"]["configured"] == settings.RERANK_BACKEND.strip().lower()
    assert data["query_rewrite"]["configured"] == settings.QUERY_REWRITE_BACKEND.strip().lower()
    assert all(item["active"] for item in data.values())
    # 前端要展示「有哪些后端可选」，所以列表必须非空
    assert all(item["available"] for item in data.values())


def test_plugins_endpoint_requires_login(anonymous_client: TestClient) -> None:
    assert anonymous_client.get("/api/system/plugins").status_code == 401


def test_health_reflects_plugin_status(client: TestClient) -> None:
    """health 的 status 要把「插件点配置写错」也算进降级，而不是只看 provider 探活。"""
    body = client.get("/api/system/health", params={"probe": "false"}).json()
    plugins = body["plugins"]
    assert set(plugins) == {
        "chat",
        "embedding",
        "vector_store",
        "cache",
        "task_queue",
        "rerank",
        "query_rewrite",
    }
    assert all(item["active"] for item in plugins.values())
    assert body["status"] in {"healthy", "degraded"}
