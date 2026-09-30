"""系统状态 API：健康检查、检索统计、缓存管理、指标、provider 与模型列表。"""

from __future__ import annotations

import asyncio
import secrets

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import PlainTextResponse

from inner_rag.api.deps import get_current_user
from inner_rag.core.config import settings
from inner_rag.core.metrics import metrics
from inner_rag.core.observability import tracer
from inner_rag.models import User
from inner_rag.plugins import plugin_status
from inner_rag.providers import (
    chat_health,
    chat_models,
    embedding_health,
    provider_catalog,
)
from inner_rag.providers.specs import ProviderError, chat_spec, embedding_spec
from inner_rag.services.cache import embedding_cache, query_cache
from inner_rag.services.lexical import lexical_index
from inner_rag.services.retrieval_log import RetrievalStats
from inner_rag.services.task_queue import task_queue

router = APIRouter(prefix="/api/system", tags=["系统"])

# /health 与 /metrics 必须免鉴权：监控与抓取不能依赖登录（/health 还永不 5xx，只报 degraded）。
# 其余系统接口（provider 目录、统计、缓存、运行时配置、模型列表）需要登录。


def _safe(callable_) -> str | None:
    """读取可能因配置非法而失败的字段，失败返回 None（用于展示类接口）。"""
    try:
        return callable_()
    except ProviderError:
        return None


@router.get("/health")
async def health_check(
    probe: bool = Query(default=True, description="是否真实探测 provider（离线环境可关）"),
):
    """探活当前 chat / embedding provider 与各插件点。

    provider 配置非法（未知名称、缺 API Key）时返回 200 + status=degraded，
    并在 error 里写清该改哪个环境变量 —— 探活接口本身不应该 5xx。

    插件点配置非法同样只降级（``plugins.<key>.active = false``），因为「后端选错」不是
    「进程活着但没法服务」——后续请求会各自抛出可读错误，比在这里 5xx 更好定位。
    """
    # 探活是网络调用，放到线程里跑，避免阻塞事件循环
    llm = await asyncio.to_thread(chat_health, probe)
    embedding = embedding_health()
    plugins = plugin_status()
    healthy = llm["ok"] and embedding["ok"] and all(item["active"] for item in plugins.values())
    return {
        "status": "healthy" if healthy else "degraded",
        "version": settings.APP_VERSION,
        "llm": llm,
        "embedding": embedding,
        "plugins": plugins,
    }


@router.get("/metrics")
async def get_metrics(x_metrics_token: str | None = Header(default=None)):
    """进程内指标（延迟分位、缓存命中、token 用量、空召回等）。

    免登录是为了让抓取器（Prometheus / curl）直连；需要门禁时配 `METRICS_TOKEN`，
    比对用 `compare_digest`（定长时间比较，避免按响应耗时逐字节试探）。
    """
    expected = settings.METRICS_TOKEN
    if expected and not (x_metrics_token and secrets.compare_digest(x_metrics_token, expected)):
        raise HTTPException(status_code=401, detail="缺少或错误的 X-Metrics-Token")

    if settings.METRICS_BACKEND == "prometheus":
        return PlainTextResponse(metrics.render_prometheus(), media_type="text/plain")
    return {
        "data": {
            "metrics": metrics.snapshot(),
            "tracing": tracer.status(),
            # 进程内累计是单 worker 语义：多副本部署时每个副本各记一份，看板要按实例聚合
            "scope": "process",
        }
    }


@router.get("/providers")
async def list_providers(user: User = Depends(get_current_user)):
    """列出支持的 provider、当前选择与 Key 是否已配置（不回显任何密钥）。"""
    return {
        "data": provider_catalog(),
        "active": {
            "llm": settings.LLM_PROVIDER,
            "embedding": settings.EMBEDDING_PROVIDER,
        },
        "embedding_key": _safe(lambda: settings.embedding_key),
    }


@router.get("/plugins")
async def list_plugins(user: User = Depends(get_current_user)):
    """所有插件点（chat / embedding / 向量库 / 缓存 / 队列 / 精排 / 查询改写）的当前实现与可选实现。

    存在的意义：替换演练与排障都要能回答「现在到底跑的哪个后端」。逐插件点手工统计
    （改配置项就要改接口）迟早会漏，所以这里直接遍历注册表，第三方实现也一并列出。
    """
    return {"data": plugin_status()}


@router.get("/stats")
async def get_stats(user: User = Depends(get_current_user)):
    """检索命中率 + 缓存状态 + 后台队列状态。"""
    return {
        "retrieval": RetrievalStats.summary(),
        "query_cache": query_cache.stats(),
        "embedding_cache": embedding_cache.stats(),
        "task_queue": task_queue.summary(),
    }


@router.post("/cache/clear")
async def clear_cache(
    user: User = Depends(get_current_user), kb_id: int | None = Query(default=None)
):
    """手动清除缓存：指定 kb_id 仅清该知识库，否则全清。"""
    if kb_id is not None:
        cleared = await query_cache.invalidate_kb(kb_id)
        lexical_index.invalidate(kb_id)
    else:
        cleared = query_cache.clear()
        lexical_index.invalidate()
    return {"message": "缓存已清除", "cleared_keys": cleared}


@router.get("/config")
async def get_config(user: User = Depends(get_current_user)):
    """返回前端可用的非敏感运行时配置。"""
    return {
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "llm_provider": settings.LLM_PROVIDER,
        "llm_model": _safe(lambda: chat_spec().model),
        "embedding_provider": settings.EMBEDDING_PROVIDER,
        "embedding_model": _safe(lambda: embedding_spec().model),
        "embedding_key": _safe(lambda: settings.embedding_key),
        "top_k": settings.TOP_K,
        "rerank_top_k": settings.RERANK_TOP_K,
        "score_threshold": settings.RETRIEVAL_SCORE_THRESHOLD,
        "chunk_size": settings.CHUNK_SIZE,
        "chunk_overlap": settings.CHUNK_OVERLAP,
        "ocr_backend": settings.OCR_BACKEND,
        "allowed_extensions": settings.allowed_extensions_list,
        "max_file_size": settings.MAX_FILE_SIZE,
        "extra": {"base_url": f"http://127.0.0.1:{settings.PORT}"},
    }


@router.get("/models")
async def list_models(user: User = Depends(get_current_user)):
    """当前 chat provider 的可用模型列表（provider 不可达时 error 里有原因）。"""
    return await asyncio.to_thread(chat_models)
