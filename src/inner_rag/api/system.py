"""系统状态 API：健康检查、检索统计、缓存管理、模型列表。"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Query

from inner_rag.core.config import settings
from inner_rag.services.cache import embedding_cache, query_cache
from inner_rag.services.rag import rag_service
from inner_rag.services.retrieval_log import RetrievalStats

router = APIRouter(prefix="/api/system", tags=["系统"])


@router.get("/health")
async def health_check():
    # 探活是网络调用，放到线程里跑，避免阻塞事件循环
    ollama_ok = await asyncio.to_thread(rag_service.test_connection)
    status = "healthy" if ollama_ok else "degraded"
    return {
        "status": status,
        "ollama": ollama_ok,
        "llm_model": settings.OLLAMA_LLM_MODEL,
        "embedding_model": settings.OLLAMA_EMBEDDING_MODEL,
        "version": settings.APP_VERSION,
    }


@router.get("/stats")
async def get_stats():
    """检索命中率 + 缓存状态。"""
    return {
        "retrieval": RetrievalStats.summary(),
        "query_cache": query_cache.stats(),
        "embedding_cache": embedding_cache.stats(),
    }


@router.post("/cache/clear")
async def clear_cache(kb_id: int | None = Query(default=None)):
    """手动清除缓存：指定 kb_id 仅清该知识库，否则全清。"""
    if kb_id is not None:
        cleared = await query_cache.invalidate_kb(kb_id)
    else:
        cleared = await query_cache.invalidate_all()
    return {"message": "缓存已清除", "cleared_keys": cleared}


@router.get("/config")
async def get_config():
    """返回前端可用的非敏感运行时配置。"""
    return {
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
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
async def list_models():
    return {"models": await asyncio.to_thread(rag_service.list_models)}
