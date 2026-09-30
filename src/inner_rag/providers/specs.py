"""Provider 元数据与解析：把 .env 里的配置翻译成具体的运行时参数。

业务代码只依赖 ``inner_rag.providers.factory``，用哪家模型后端完全由 .env 决定：

* ``LLM_PROVIDER``       = ollama | openrouter | deepseek | openai | mock
* ``EMBEDDING_PROVIDER`` = ollama | openrouter | openai | mock

Phase 7 起「有哪些 provider」不再由本模块的 if 分支决定，而是由 ``inner_rag.plugins`` 的注册表
决定：本模块把内置 provider 注册进去（spec 工厂与模型构造器成对，见 ``_CHAT_SPECS`` /
``_EMBEDDING_SPECS``），第三方包通过 entry point（``inner_rag.chat_providers`` /
``inner_rag.embedding_providers``）注册自己的实现；报错文案里的「可选值」也从注册表实时取，
新增 provider 不会漏改提示。

约定：

* provider 名大小写不敏感，前后空白会被忽略，另有少量别名（local/fake/oai...）；
* provider 名称未知、该 provider 不支持所需能力（DeepSeek 只有 chat API，没有
  embedding 接口），或缺少 API Key 时，统一抛 ``ProviderError``，消息里直接写明
  该去 .env 改哪个变量 —— 而不是等到真正调模型时抛一个用户看不懂的 401。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from inner_rag.core.config import settings
from inner_rag.plugins.registry import (
    ChatProvider,
    EmbeddingProvider,
    chat_providers,
    embedding_providers,
)

CHAT = "chat"
EMBEDDING = "embedding"

ALIASES: dict[str, str] = {
    "local": "ollama",
    "oai": "openai",
    "open-router": "openrouter",
    "fake": "mock",
    "test": "mock",
}


class ProviderError(RuntimeError):
    """provider 配置错误：名称未知、不支持所需能力，或缺少 API Key。"""


@dataclass(frozen=True)
class ProviderSpec:
    """某家 provider 解析后的运行时参数。"""

    name: str
    label: str
    kind: str
    model: str
    base_url: str = ""
    api_key: str = ""
    api_key_env: str = ""
    docs_url: str = ""
    notes: str = ""
    # provider 的 /models 是否完整列出可调用的模型名。DeepSeek 的 /models 只列当前主推
    # 模型，旧别名（如 deepseek-chat）仍能调用，这类 provider 上「模型不在列表里」
    # 只能给告警，不能判成不可用。
    model_list_authoritative: bool = True

    @property
    def identity(self) -> str:
        """``provider:model``，写入知识库并用于向量空间一致性校验。"""
        return f"{self.name}:{self.model}"

    @property
    def requires_api_key(self) -> bool:
        return bool(self.api_key_env)

    @property
    def configured(self) -> bool:
        """所需 API Key 是否已填写（不需要 Key 的 provider 恒为 True）。"""
        return not self.requires_api_key or bool(self.api_key.strip())


def normalize(name: str) -> str:
    """provider 名归一化：去空白、转小写、展开别名。"""
    cleaned = (name or "").strip().lower()
    return ALIASES.get(cleaned, cleaned)


# ── 内置 provider 的 spec 工厂 ────────────────────────────────────────
# 只读配置、不做校验，因此配置不全时也能用于展示（/api/system/providers）。


def _ollama_chat() -> ProviderSpec:
    return ProviderSpec(
        name="ollama",
        label="Ollama（本地）",
        kind=CHAT,
        model=settings.OLLAMA_LLM_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        docs_url="https://ollama.com",
        notes="无需 API Key，需要本地运行 ollama serve",
    )


def _openrouter_chat() -> ProviderSpec:
    return ProviderSpec(
        name="openrouter",
        label="OpenRouter（云端聚合，OpenAI 兼容）",
        kind=CHAT,
        model=settings.OPENROUTER_CHAT_MODEL,
        base_url=settings.OPENROUTER_BASE_URL,
        api_key=settings.OPENROUTER_API_KEY,
        api_key_env="OPENROUTER_API_KEY",
        docs_url="https://openrouter.ai/docs",
        notes="一个 Key 可路由数百个模型；embedding 走 /embeddings",
    )


def _deepseek_chat() -> ProviderSpec:
    return ProviderSpec(
        name="deepseek",
        label="DeepSeek（云端，OpenAI 兼容）",
        kind=CHAT,
        model=settings.DEEPSEEK_CHAT_MODEL,
        base_url=settings.DEEPSEEK_BASE_URL,
        api_key=settings.DEEPSEEK_API_KEY,
        api_key_env="DEEPSEEK_API_KEY",
        docs_url="https://api-docs.deepseek.com",
        notes=(
            "官方只有 chat completion，不提供 embedding 接口；"
            "模型名以官方文档为准（deepseek-flash / deepseek-v4-pro），"
            "/models 不列出旧别名"
        ),
        model_list_authoritative=False,
    )


def _openai_chat() -> ProviderSpec:
    return ProviderSpec(
        name="openai",
        label="OpenAI（云端，OpenAI 兼容）",
        kind=CHAT,
        model=settings.OPENAI_CHAT_MODEL,
        base_url=settings.OPENAI_BASE_URL,
        api_key=settings.OPENAI_API_KEY,
        api_key_env="OPENAI_API_KEY",
        docs_url="https://platform.openai.com/docs",
        notes="也可指向任何 OpenAI 兼容的自建网关（改 OPENAI_BASE_URL）",
    )


def _mock_chat() -> ProviderSpec:
    return ProviderSpec(
        name="mock",
        label="Mock（离线假模型）",
        kind=CHAT,
        model=settings.MOCK_CHAT_MODEL,
        docs_url="docs/DEVELOPMENT_PLAN.md",
        notes="不联网、不需要密钥，用于本地演示、CI 与降级验收",
    )


def _ollama_embedding() -> ProviderSpec:
    return ProviderSpec(
        name="ollama",
        label="Ollama（本地）",
        kind=EMBEDDING,
        model=settings.OLLAMA_EMBEDDING_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        docs_url="https://ollama.com",
        notes="向量维度取决于本地模型，换模型需重建索引",
    )


def _openrouter_embedding() -> ProviderSpec:
    return ProviderSpec(
        name="openrouter",
        label="OpenRouter（云端聚合，OpenAI 兼容）",
        kind=EMBEDDING,
        model=settings.OPENROUTER_EMBEDDING_MODEL,
        base_url=settings.OPENROUTER_BASE_URL,
        api_key=settings.OPENROUTER_API_KEY,
        api_key_env="OPENROUTER_API_KEY",
        docs_url="https://openrouter.ai/docs/api_reference/embeddings",
        notes=(
            "注意模型上下文长度：liquid/lfm-2.5-embedding-350m:free 只有 512 token，"
            "用它可以配 EMBEDDING_MAX_INPUT_CHARS=400；免费路由的数据可能被用于训练"
        ),
    )


def _openai_embedding() -> ProviderSpec:
    return ProviderSpec(
        name="openai",
        label="OpenAI（云端，OpenAI 兼容）",
        kind=EMBEDDING,
        model=settings.OPENAI_EMBEDDING_MODEL,
        base_url=settings.OPENAI_BASE_URL,
        api_key=settings.OPENAI_API_KEY,
        api_key_env="OPENAI_API_KEY",
        docs_url="https://platform.openai.com/docs/guides/embeddings",
        notes="text-embedding-3-small 为 1536 维，改动需重建索引",
    )


def _mock_embedding() -> ProviderSpec:
    return ProviderSpec(
        name="mock",
        label="Mock（离线确定性 embedding）",
        kind=EMBEDDING,
        model=f"{settings.MOCK_EMBEDDING_MODEL}-{settings.MOCK_EMBEDDING_DIM}d",
        docs_url="docs/DEVELOPMENT_PLAN.md",
        notes="哈希词袋向量，不联网；仅供演示与测试，不具备真实语义检索能力",
    )


# 注册顺序 = `.env.example` 与 `/api/system/providers` 的展示顺序；模型构造器按同名键取
# （见 chat.CHAT_BUILDERS / embeddings.EMBEDDING_BUILDERS），少一个会在注册时 KeyError。
_CHAT_SPECS: dict[str, Callable[[], ProviderSpec]] = {
    "ollama": _ollama_chat,
    "openrouter": _openrouter_chat,
    "deepseek": _deepseek_chat,
    "openai": _openai_chat,
    "mock": _mock_chat,
}
_EMBEDDING_SPECS: dict[str, Callable[[], ProviderSpec]] = {
    "ollama": _ollama_embedding,
    "openrouter": _openrouter_embedding,
    "openai": _openai_embedding,
    "mock": _mock_embedding,
}

_builtins_registered = False


def _ensure_builtins() -> None:
    """注册内置 provider（幂等），随后再发现第三方实现。

    放在解析入口调用而不是 import 期：specs 与 chat / embeddings 互相引用，import 期注册会读到
    半初始化的模块（属性还没定义）。解析发生在调用时，那时各模块都已加载完。先内置后第三方，
    保证同名时以内置实现为准。
    """
    global _builtins_registered
    if not _builtins_registered:
        # 延迟 import：chat / embeddings 的构造器要以 ProviderSpec 为入参，只能在运行时取
        from inner_rag.providers.chat import CHAT_BUILDERS
        from inner_rag.providers.embeddings import EMBEDDING_BUILDERS

        for name, spec_factory in _CHAT_SPECS.items():
            chat_providers.register(
                name, ChatProvider(spec=spec_factory, build=CHAT_BUILDERS[name])
            )
        for name, spec_factory in _EMBEDDING_SPECS.items():
            embedding_providers.register(
                name, EmbeddingProvider(spec=spec_factory, build=EMBEDDING_BUILDERS[name])
            )
        _builtins_registered = True

    chat_providers.load_entry_points()
    embedding_providers.load_entry_points()


# ── 解析与校验 ────────────────────────────────────────────────────────


def _resolve(kind: str, requested: str | None) -> ChatProvider | EmbeddingProvider:
    """按 kind 解析出 provider 插件（含名字归一化与「可读报错」）。"""
    _ensure_builtins()
    is_chat = kind == CHAT
    registry = chat_providers if is_chat else embedding_providers
    env_var = "LLM_PROVIDER" if is_chat else "EMBEDDING_PROVIDER"
    raw = requested if requested is not None else ""
    name = normalize(raw)

    plugin = registry.get(name)
    if plugin is None:
        if not is_chat and name in chat_providers:
            # 比「未知 provider」更准确的解释：这家只提供 chat 接口
            msg = (
                f"{env_var}={name} 不支持 embedding：该后端只提供 chat completion 接口。"
                f"请在 .env 中改用 {'/'.join(registry.names())}"
            )
            raise ProviderError(msg)
        msg = (
            f"未知的 {env_var}={raw!r}，可选：{'/'.join(registry.names())}"
            "（见 .env.example；第三方实现见 docs/architecture.md 第 5 节）"
        )
        raise ProviderError(msg)

    spec = plugin.spec()
    if not spec.configured:
        msg = (
            f"{env_var}={name} 需要 API Key：请在 .env 中填写 "
            f"{spec.api_key_env}=... 后重启服务（模板见 .env.example）"
        )
        raise ProviderError(msg)
    return plugin


def chat_plugin(provider: str | None = None) -> ChatProvider:
    """解析 chat provider 插件；默认取 ``LLM_PROVIDER``。

    工厂用它一次拿到 spec 与构造器，省掉「解析出 spec 再查表找 builder」这一步
    （那一步查不到时只能写一个不可能触发的分支）。
    """
    return cast(
        "ChatProvider", _resolve(CHAT, settings.LLM_PROVIDER if provider is None else provider)
    )


def embedding_plugin(provider: str | None = None) -> EmbeddingProvider:
    """解析 embedding provider 插件；默认取 ``EMBEDDING_PROVIDER``。"""
    return cast(
        "EmbeddingProvider",
        _resolve(EMBEDDING, settings.EMBEDDING_PROVIDER if provider is None else provider),
    )


def chat_spec(provider: str | None = None) -> ProviderSpec:
    """解析 chat provider 的运行时参数。"""
    return chat_plugin(provider).spec()


def embedding_spec(provider: str | None = None) -> ProviderSpec:
    """解析 embedding provider 的运行时参数。"""
    return embedding_plugin(provider).spec()


# ── 展示（/api/system/providers）───────────────────────────────────────


def _catalog_entry(
    name: str,
    plugin: ChatProvider | EmbeddingProvider,
    kind: str,
    active_env: str,
    builtin_names: dict[str, Any],
) -> dict[str, Any]:
    spec = plugin.spec()
    return {
        "name": name,
        "label": spec.label,
        "kind": kind,
        "model": spec.model,
        "base_url": spec.base_url,
        "api_key_env": spec.api_key_env,
        "configured": spec.configured,
        "active": normalize(active_env) == name,
        "docs_url": spec.docs_url,
        "notes": spec.notes,
        "model_list_authoritative": spec.model_list_authoritative,
        "third_party": name not in builtin_names,
    }


def provider_catalog() -> dict[str, list[dict[str, Any]]]:
    """列出所有支持的 provider 及其配置状态（含第三方实现，不含任何密钥内容）。"""
    _ensure_builtins()
    return {
        CHAT: [
            _catalog_entry(name, plugin, CHAT, settings.LLM_PROVIDER, _CHAT_SPECS)
            for name, plugin in chat_providers.items()
        ],
        EMBEDDING: [
            _catalog_entry(name, plugin, EMBEDDING, settings.EMBEDDING_PROVIDER, _EMBEDDING_SPECS)
            for name, plugin in embedding_providers.items()
        ],
    }
