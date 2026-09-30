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
from inner_rag.plugins.registry import ChatProvider, PluginError, Registry, chat_providers
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
    registry: Registry[str] = Registry("test backend", "test.group")
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
    registry: Registry[str] = Registry("test backend", "test.group")
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
