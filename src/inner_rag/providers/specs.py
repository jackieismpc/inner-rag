"""Provider 元数据与解析：把 .env 里的配置翻译成具体的运行时参数。

业务代码只依赖 ``inner_rag.providers.factory``，用哪家模型后端完全由 .env 决定：

* ``LLM_PROVIDER``       = ollama | openrouter | deepseek | openai | mock
* ``EMBEDDING_PROVIDER`` = ollama | openrouter | openai | mock

约定：

* provider 名大小写不敏感，前后空白会被忽略，另有少量别名（local/fake/oai...）；
* provider 名称未知、该 provider 不支持所需能力（DeepSeek 只有 chat API，没有
  embedding 接口），或缺少 API Key 时，统一抛 ``ProviderError``，消息里直接写明
  该去 .env 改哪个变量 —— 而不是等到真正调模型时抛一个用户看不懂的 401。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from inner_rag.core.config import settings

CHAT = "chat"
EMBEDDING = "embedding"

CHAT_PROVIDERS: tuple[str, ...] = ("ollama", "openrouter", "deepseek", "openai", "mock")
EMBEDDING_PROVIDERS: tuple[str, ...] = ("ollama", "openrouter", "openai", "mock")

# 只有 chat、没有 embedding 的后端，用于给出比「未知 provider」更准确的报错
CHAT_ONLY_PROVIDERS: tuple[str, ...] = ("deepseek",)

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


# ── 各 provider 的参数来源 ────────────────────────────────────────────


def _build(kind: str, name: str) -> ProviderSpec:
    """构造 spec。只读配置、不做校验，因此配置不全时也能用于展示。"""
    if kind == CHAT:
        if name == "ollama":
            return ProviderSpec(
                name=name,
                label="Ollama（本地）",
                kind=kind,
                model=settings.OLLAMA_LLM_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
                docs_url="https://ollama.com",
                notes="无需 API Key，需要本地运行 ollama serve",
            )
        if name == "openrouter":
            return ProviderSpec(
                name=name,
                label="OpenRouter（云端聚合，OpenAI 兼容）",
                kind=kind,
                model=settings.OPENROUTER_CHAT_MODEL,
                base_url=settings.OPENROUTER_BASE_URL,
                api_key=settings.OPENROUTER_API_KEY,
                api_key_env="OPENROUTER_API_KEY",
                docs_url="https://openrouter.ai/docs",
                notes="一个 Key 可路由数百个模型；embedding 走 /embeddings",
            )
        if name == "deepseek":
            return ProviderSpec(
                name=name,
                label="DeepSeek（云端，OpenAI 兼容）",
                kind=kind,
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
        if name == "openai":
            return ProviderSpec(
                name=name,
                label="OpenAI（云端，OpenAI 兼容）",
                kind=kind,
                model=settings.OPENAI_CHAT_MODEL,
                base_url=settings.OPENAI_BASE_URL,
                api_key=settings.OPENAI_API_KEY,
                api_key_env="OPENAI_API_KEY",
                docs_url="https://platform.openai.com/docs",
                notes="也可指向任何 OpenAI 兼容的自建网关（改 OPENAI_BASE_URL）",
            )
        return ProviderSpec(
            name="mock",
            label="Mock（离线假模型）",
            kind=kind,
            model=settings.MOCK_CHAT_MODEL,
            docs_url="docs/DEVELOPMENT_PLAN.md",
            notes="不联网、不需要密钥，用于本地演示、CI 与降级验收",
        )

    if name == "ollama":
        return ProviderSpec(
            name=name,
            label="Ollama（本地）",
            kind=kind,
            model=settings.OLLAMA_EMBEDDING_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            docs_url="https://ollama.com",
            notes="向量维度取决于本地模型，换模型需重建索引",
        )
    if name == "openrouter":
        return ProviderSpec(
            name=name,
            label="OpenRouter（云端聚合，OpenAI 兼容）",
            kind=kind,
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
    if name == "openai":
        return ProviderSpec(
            name=name,
            label="OpenAI（云端，OpenAI 兼容）",
            kind=kind,
            model=settings.OPENAI_EMBEDDING_MODEL,
            base_url=settings.OPENAI_BASE_URL,
            api_key=settings.OPENAI_API_KEY,
            api_key_env="OPENAI_API_KEY",
            docs_url="https://platform.openai.com/docs/guides/embeddings",
            notes="text-embedding-3-small 为 1536 维，改动需重建索引",
        )
    return ProviderSpec(
        name="mock",
        label="Mock（离线确定性 embedding）",
        kind=kind,
        model=f"{settings.MOCK_EMBEDDING_MODEL}-{settings.MOCK_EMBEDDING_DIM}d",
        docs_url="docs/DEVELOPMENT_PLAN.md",
        notes="哈希词袋向量，不联网；仅供演示与测试，不具备真实语义检索能力",
    )


# ── 解析与校验 ────────────────────────────────────────────────────────


def _resolve(kind: str, requested: str | None) -> ProviderSpec:
    env_var = "LLM_PROVIDER" if kind == CHAT else "EMBEDDING_PROVIDER"
    allowed = CHAT_PROVIDERS if kind == CHAT else EMBEDDING_PROVIDERS
    raw = requested if requested is not None else ""
    name = normalize(raw)

    if kind == EMBEDDING and name in CHAT_ONLY_PROVIDERS:
        msg = (
            f"{env_var}={name} 不支持 embedding：DeepSeek 只提供 chat completion 接口。"
            f"请在 .env 中改用 {'/'.join(EMBEDDING_PROVIDERS)}"
        )
        raise ProviderError(msg)

    if name not in allowed:
        msg = f"未知的 {env_var}={raw!r}，可选：{'/'.join(allowed)}（见 .env.example）"
        raise ProviderError(msg)

    spec = _build(kind, name)
    if not spec.configured:
        msg = (
            f"{env_var}={name} 需要 API Key：请在 .env 中填写 "
            f"{spec.api_key_env}=... 后重启服务（模板见 .env.example）"
        )
        raise ProviderError(msg)
    return spec


def chat_spec(provider: str | None = None) -> ProviderSpec:
    """解析 chat provider；默认取 ``LLM_PROVIDER``。"""
    return _resolve(CHAT, settings.LLM_PROVIDER if provider is None else provider)


def embedding_spec(provider: str | None = None) -> ProviderSpec:
    """解析 embedding provider；默认取 ``EMBEDDING_PROVIDER``。"""
    return _resolve(EMBEDDING, settings.EMBEDDING_PROVIDER if provider is None else provider)


def _catalog_entry(kind: str, name: str, active_env: str) -> dict[str, Any]:
    spec = _build(kind, name)
    return {
        "name": spec.name,
        "label": spec.label,
        "kind": kind,
        "model": spec.model,
        "base_url": spec.base_url,
        "api_key_env": spec.api_key_env,
        "configured": spec.configured,
        "active": normalize(active_env) == spec.name,
        "docs_url": spec.docs_url,
        "notes": spec.notes,
        "model_list_authoritative": spec.model_list_authoritative,
    }


def provider_catalog() -> dict[str, list[dict[str, Any]]]:
    """列出所有支持的 provider 及其配置状态（不含任何密钥内容）。"""
    return {
        CHAT: [_catalog_entry(CHAT, name, settings.LLM_PROVIDER) for name in CHAT_PROVIDERS],
        EMBEDDING: [
            _catalog_entry(EMBEDDING, name, settings.EMBEDDING_PROVIDER)
            for name in EMBEDDING_PROVIDERS
        ],
    }
