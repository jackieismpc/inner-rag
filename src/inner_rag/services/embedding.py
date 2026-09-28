"""Embedding 服务：统一出口，负责缓存、批量与并发控制。

LangChain 的向量库在内部会同步调用 embed_documents / embed_query，
业务侧则走异步批量嵌入。两条路径共享同一个 EmbeddingCache，
因此「入库前预热」与「向量库内部嵌入」不会重复请求模型。
"""

from __future__ import annotations

import asyncio

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from loguru import logger

from inner_rag.core.config import settings
from inner_rag.services.cache import embedding_cache


def _cached_embed_documents(embeddings: Embeddings, texts: list[str]) -> list[list[float]]:
    """同步批量嵌入：逐条走缓存，只把未命中的文本发给模型。"""
    resolved: list[list[float] | None] = [embedding_cache.get_sync(text) for text in texts]
    missing = [text for text, vec in zip(texts, resolved, strict=True) if vec is None]
    if missing:
        computed = iter(embeddings.embed_documents(missing))
        for index, vector in enumerate(resolved):
            if vector is None:
                fresh = next(computed)
                resolved[index] = fresh
                embedding_cache.set_sync(texts[index], fresh)
    return _finalize(resolved, len(texts))


def _cached_embed_query(embeddings: Embeddings, text: str) -> list[float]:
    cached = embedding_cache.get_sync(text)
    if cached is not None:
        return cached
    vector = embeddings.embed_query(text)
    embedding_cache.set_sync(text, vector)
    return vector


def _finalize(vectors: list[list[float] | None], expected: int) -> list[list[float]]:
    resolved = [vec for vec in vectors if vec is not None]
    if len(resolved) != expected:
        msg = f"embedding 结果数量不完整: {len(resolved)}/{expected}"
        raise RuntimeError(msg)
    return resolved


class EmbeddingService(Embeddings):
    """Ollama Embedding 的统一入口（缓存 + 批量 + 并发限制）。"""

    def __init__(self) -> None:
        self._embeddings: OllamaEmbeddings | None = None
        self._semaphore: asyncio.Semaphore | None = None

    # ── 惰性初始化 ─────────────────────────────────────────────────────

    @property
    def embeddings(self) -> OllamaEmbeddings:
        if self._embeddings is None:
            self._embeddings = OllamaEmbeddings(
                base_url=settings.OLLAMA_BASE_URL,
                model=settings.OLLAMA_EMBEDDING_MODEL,
            )
            logger.info(f"[EMBEDDING] 初始化 {self.identity}")
        return self._embeddings

    def _get_semaphore(self) -> asyncio.Semaphore:
        """信号量在事件循环内惰性创建。"""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(settings.EMBEDDING_CONCURRENCY)
        return self._semaphore

    @property
    def identity(self) -> str:
        """embedding 身份标识，用于向量空间一致性校验。"""
        return settings.embedding_key

    # ── LangChain Embeddings 接口（同步路径）───────────────────────────

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return _cached_embed_documents(self.embeddings, texts)

    def embed_query(self, text: str) -> list[float]:
        return _cached_embed_query(self.embeddings, text)

    # ── 异步路径（入库与检索）──────────────────────────────────────────

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors, miss_indices = await embedding_cache.get_batch(texts)
        if not miss_indices:
            logger.debug(f"[EMBEDDING] 全部命中缓存 ({len(texts)} 条)")
            return _finalize(vectors, len(texts))

        miss_texts = [texts[i] for i in miss_indices]
        computed = await self._embed_in_batches(miss_texts)
        await embedding_cache.set_batch(miss_texts, computed)
        for index, vector in zip(miss_indices, computed, strict=True):
            vectors[index] = vector
        logger.debug(
            f"[EMBEDDING] {len(texts) - len(miss_indices)} 命中 / {len(miss_indices)} 未命中"
        )
        return _finalize(vectors, len(texts))

    async def aembed_query(self, text: str) -> list[float]:
        cached = await embedding_cache.get(text)
        if cached is not None:
            return cached
        async with self._get_semaphore():
            vector = await self.embeddings.aembed_query(text)
        await embedding_cache.set(text, vector)
        return vector

    async def _embed_in_batches(self, texts: list[str]) -> list[list[float]]:
        result: list[list[float]] = []
        batch_size = settings.EMBED_BATCH_SIZE
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            async with self._get_semaphore():
                result.extend(await self.embeddings.aembed_documents(batch))
        return result


embedding_service = EmbeddingService()
