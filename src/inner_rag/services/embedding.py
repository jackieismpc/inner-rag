"""Embedding 服务：统一出口，负责缓存、批量与并发控制。

LangChain 的向量库在内部会同步调用 embed_documents / embed_query，
业务侧则走异步批量嵌入。两条路径共享同一个 EmbeddingCache，
因此「入库前预热」与「向量库内部嵌入」不会重复请求模型。

具体用哪家 embedding 后端由 ``inner_rag.providers`` 决定（.env 里的
EMBEDDING_PROVIDER），本模块不关心是 Ollama 还是云端 API。
"""

from __future__ import annotations

import asyncio
import math

from langchain_core.embeddings import Embeddings
from loguru import logger

from inner_rag.core.config import settings
from inner_rag.core.metrics import metrics
from inner_rag.core.observability import tracer
from inner_rag.providers import get_embeddings
from inner_rag.services.cache import embedding_cache


class EmbeddingIdentityMismatch(RuntimeError):
    """知识库记录的 embedding 与当前配置不一致。"""


def ensure_embedding_matches(kb_id: int, embedding_model: str | None) -> None:
    """校验知识库的向量空间与当前 embedding 配置一致，避免静默检索到无意义的向量。

    向量空间一致性是 embedding 身份的属性，不是某个向量库后端的属性：换 provider / 模型后，
    同一个库里的新旧向量不能混着检索（命中会是噪声，而不是报错）。
    """
    if embedding_model and embedding_model != settings.embedding_key:
        msg = (
            f"知识库 kb={kb_id} 建库时使用 {embedding_model}，"
            f"当前配置为 {settings.embedding_key}；"
            "请切回原 embedding 模型，或执行 scripts/reindex_kb.py 重建索引"
        )
        raise EmbeddingIdentityMismatch(msg)


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
    """Embedding 的统一入口（缓存 + 批量 + 并发限制）。"""

    def __init__(self) -> None:
        self._embeddings: Embeddings | None = None
        self._semaphore: asyncio.Semaphore | None = None

    # ── 惰性初始化 ─────────────────────────────────────────────────────

    @property
    def embeddings(self) -> Embeddings:
        if self._embeddings is None:
            # provider 配置非法时（未知 provider / 缺 API Key）
            # get_embeddings 会抛 ProviderError，由 API 层映射成 503。
            self._embeddings = get_embeddings()
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
        """批量嵌入：入库时的主要成本项，因此单独成 span 并记缓存命中率。"""
        if not texts:
            return []

        async with tracer.span("embed.documents", count=len(texts), identity=self.identity) as span:
            vectors, miss_indices = await embedding_cache.get_batch(texts)
            hits = len(texts) - len(miss_indices)
            span.set(cache_hits=hits, cache_misses=len(miss_indices))
            if hits:
                metrics.increment("rag_cache_hits_total", hits, labels={"namespace": "embedding"})
            if not miss_indices:
                return _finalize(vectors, len(texts))

            miss_texts = [texts[i] for i in miss_indices]
            span.set(batch_count=math.ceil(len(miss_texts) / settings.EMBED_BATCH_SIZE))
            computed = await self._embed_in_batches(miss_texts)
            await embedding_cache.set_batch(miss_texts, computed)
            for index, vector in zip(miss_indices, computed, strict=True):
                vectors[index] = vector
            return _finalize(vectors, len(texts))

    async def aembed_query(self, text: str) -> list[float]:
        async with tracer.span("embed.query") as span:
            cached = await embedding_cache.get(text)
            span.set(cache_hit=cached is not None)
            if cached is not None:
                metrics.increment("rag_cache_hits_total", labels={"namespace": "embedding"})
                return cached
            async with self._get_semaphore():
                vector = await self.embeddings.aembed_query(text)
            await embedding_cache.set(text, vector)
            return vector

    async def _embed_in_batches(self, texts: list[str]) -> list[list[float]]:
        """分批**并发**嵌入：并发上限由 ``EMBEDDING_CONCURRENCY`` 决定，失败按批重试。

        两条容易踩的坑，都在这里处理掉：

        1. **并发上限必须真的生效**：早期实现是 `for batch: async with semaphore: await ...`，
           每一批都在等上一批返回，信号量形同虚设，入库吞吐被单次往返时间锁死
           （11165 页的全库要跑成小时级）。改成「一次派发全部批次 + 信号量限流 + 按下标回填」后，
           真实并发等于配置值，输出顺序仍与输入严格一致；
        2. **单批失败必须能重试，且不能拖住其它批**：免费/共享 embedding 后端会偶发 429 与超时。
           失败时若直接让异常穿透 ``gather``，**其余未完成的批次会继续在后台跑**——
           调用方以为已经失败并重试，实际两轮请求叠在一起，既打满额度又让状态无法解释。
           所以这里：单批指数退避重试；重试仍失败则取消所有兄弟任务并抛出。

        批量大时按 ``EMBED_PROGRESS_EVERY`` 打进度：全库入库是分钟级任务，
        没有进度日志就只能靠猜（OpenAI 客户端默认 600s 超时那种挂起尤其难查）。
        """
        batch_size = settings.EMBED_BATCH_SIZE
        batches = [texts[start : start + batch_size] for start in range(0, len(texts), batch_size)]
        if not batches:
            return []

        semaphore = self._get_semaphore()
        max_attempts = max(1, settings.EMBED_MAX_ATTEMPTS)
        report_every = max(1, settings.EMBED_PROGRESS_EVERY)
        verbose = len(batches) >= report_every
        done = 0

        async def embed_batch(batch: list[str]) -> list[list[float]]:
            nonlocal done
            last_error: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    async with semaphore:
                        vectors = await self.embeddings.aembed_documents(batch)
                except Exception as exc:  # 上游抖动：退避后重试同一批
                    last_error = exc
                    if attempt < max_attempts:
                        await asyncio.sleep(settings.EMBED_RETRY_BACKOFF * attempt)
                    continue
                done += 1
                if verbose and (done % report_every == 0 or done == len(batches)):
                    logger.info(f"[EMBED] 分批进度 {done}/{len(batches)}")
                return vectors
            assert last_error is not None  # 只有全部尝试都失败才会走到这里
            raise last_error

        tasks = [asyncio.create_task(embed_batch(batch)) for batch in batches]
        try:
            results = await asyncio.gather(*tasks)
        except BaseException:
            # 一批彻底失败就取消其余：让「失败」这件事对调用方是确定且可重试的
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        # gather 保序：结果顺序与 batches 一致，因此与 texts 一致
        return [vector for batch_result in results for vector in batch_result]


embedding_service = EmbeddingService()
