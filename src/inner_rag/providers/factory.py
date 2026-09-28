"""Provider 工厂：业务层获取模型实例的唯一入口。

设计要点：

* 惰性 + 缓存：同一份配置只构造一次客户端，避免每个请求都重建连接池；
* 只读 .env（settings），不接受任何来自 HTTP 请求的 provider 参数，
  避免调用方指定任意 base_url（SSRF）；
* 缺 Key / provider 非法统一抛 ``ProviderError``，由上层映射成 503 + 可读文案。
"""

from __future__ import annotations

import hashlib
from typing import Any, cast

import httpx
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from loguru import logger

from inner_rag.core.config import settings
from inner_rag.providers.chat import build_chat_model
from inner_rag.providers.embeddings import build_embeddings
from inner_rag.providers.specs import (
    ProviderError,
    ProviderSpec,
    chat_spec,
    embedding_spec,
)

# 探活用短超时：health 会被前端轮询，不能让它挂着
PROBE_TIMEOUT = 5.0

_CACHE: dict[str, Any] = {}


def _fingerprint(secret: str) -> str:
    """密钥指纹：用于缓存 key，避免把明文密钥当字典键。"""
    if not secret:
        return "-"
    return hashlib.blake2b(secret.encode(), digest_size=4).hexdigest()


# ── 模型实例 ──────────────────────────────────────────────────────────


def get_chat_model(provider: str | None = None) -> BaseChatModel:
    """获取 chat 模型实例（进程内复用）。"""
    spec = chat_spec(provider)
    cache_key = (
        f"chat|{spec.identity}|{spec.base_url}|{_fingerprint(spec.api_key)}"
        f"|{settings.LLM_TEMPERATURE}|{settings.LLM_MAX_TOKENS}"
        f"|{settings.LLM_TIMEOUT}|{settings.LLM_MAX_RETRIES}"
    )
    model = _CACHE.get(cache_key)
    if model is None:
        model = build_chat_model(spec)
        _CACHE[cache_key] = model
    return cast(BaseChatModel, model)


def get_embeddings(provider: str | None = None) -> Embeddings:
    """获取 embedding 实例（进程内复用）。"""
    spec = embedding_spec(provider)
    cache_key = (
        f"embed|{spec.identity}|{spec.base_url}|{_fingerprint(spec.api_key)}"
        f"|{settings.EMBEDDING_MAX_INPUT_CHARS}"
    )
    embeddings = _CACHE.get(cache_key)
    if embeddings is None:
        embeddings = build_embeddings(spec)
        _CACHE[cache_key] = embeddings
    return cast(Embeddings, embeddings)


def reset_cache() -> None:
    """清空实例缓存：配置变更（测试、脚本切模型）后调用。"""
    _CACHE.clear()


# ── 探活与模型发现 ────────────────────────────────────────────────────


def _discover_models(spec: ProviderSpec) -> tuple[list[str], str | None]:
    """查询 provider 的模型列表，返回 (模型名列表, 错误信息)。"""
    if spec.name == "mock":
        return [spec.model], None

    base_url = spec.base_url.rstrip("/")
    url = f"{base_url}/api/tags" if spec.name == "ollama" else f"{base_url}/models"
    headers = {"Authorization": f"Bearer {spec.api_key}"} if spec.api_key else {}

    try:
        response = httpx.get(url, headers=headers, timeout=PROBE_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        logger.debug(f"[PROVIDER] {spec.name} 探活失败: {exc}")
        return [], f"无法连接 {spec.name}（{url}）: {exc}"

    if spec.name == "ollama":
        names = [str(item.get("name", "")) for item in payload.get("models", [])]
    else:
        names = [str(item.get("id", "")) for item in payload.get("data", [])]
    return [name for name in names if name], None


def chat_health(probe: bool = True) -> dict[str, Any]:
    """chat provider 的健康状态：配置是否正确 + 模型是否可用。"""
    try:
        spec = chat_spec()
    except ProviderError as exc:
        return {
            "provider": settings.LLM_PROVIDER,
            "model": None,
            "ok": False,
            "model_available": None,
            "error": str(exc),
        }

    result: dict[str, Any] = {
        "provider": spec.name,
        "model": spec.model,
        "ok": True,
        "model_available": True,
        "error": None,
    }
    if not probe or spec.name == "mock":
        return result

    models, error = _discover_models(spec)
    if error is not None:
        result.update({"ok": False, "model_available": None, "error": error})
        return result

    # Ollama 必须真的把模型拉下来才能对话；云端模型列表可能分页/别名化，只做提示
    available = spec.model in models if models else None
    result["model_available"] = available
    if available is False:
        result.update(
            {
                "ok": False,
                "error": (
                    f"{spec.name} 可达，但模型 {spec.model} 不在返回的模型列表中，"
                    f"请检查配置或先拉取模型"
                ),
            }
        )
    return result


def embedding_health() -> dict[str, Any]:
    """embedding provider 的健康状态。

    只做配置校验，不发网络请求：embedding 探活要消耗额度，而 /health 会被轮询。
    """
    try:
        spec = embedding_spec()
    except ProviderError as exc:
        return {
            "provider": settings.EMBEDDING_PROVIDER,
            "model": None,
            "ok": False,
            "probed": False,
            "error": str(exc),
        }
    return {
        "provider": spec.name,
        "model": spec.model,
        "ok": True,
        "probed": False,
        "error": None,
    }


def chat_models() -> dict[str, Any]:
    """当前 chat provider 可用的模型列表（供前端展示）。"""
    try:
        spec = chat_spec()
    except ProviderError as exc:
        return {
            "provider": settings.LLM_PROVIDER,
            "model": None,
            "models": [],
            "error": str(exc),
        }
    models, error = _discover_models(spec)
    return {"provider": spec.name, "model": spec.model, "models": models, "error": error}
