"""缓存层：QueryCache（检索结果）+ EmbeddingCache（向量）。

LRUCache 内部实现为同步（纯内存操作，无 I/O）并用 threading.RLock 保护，
同时对外提供 async 包装。这样 LangChain 向量库（如 Chroma）内部的同步嵌入路径与业务异步路径
可以共享同一份缓存，避免同一段文本被重复嵌入。
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from typing import Any

from loguru import logger

from inner_rag.core.config import settings


class LRUCache:
    def __init__(self, max_size: int, ttl: int = 0) -> None:
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl  # 0 表示永不过期
        self._hits = 0
        self._misses = 0
        self._lock = threading.RLock()

    # ── 同步接口 ───────────────────────────────────────────────────────

    def get(self, key: str) -> Any | None:
        with self._lock:
            item = self._cache.get(key)
            if item is None:
                self._misses += 1
                return None
            value, ts = item
            if self._ttl > 0 and (time.monotonic() - ts) > self._ttl:
                del self._cache[key]
                self._misses += 1
                return None
            self._cache.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._cache[key] = (value, time.monotonic())
            self._cache.move_to_end(key)
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def delete(self, key: str) -> None:
        with self._lock:
            self._cache.pop(key, None)

    def clear_prefix(self, prefix: str) -> int:
        with self._lock:
            keys = [k for k in self._cache if k.startswith(prefix)]
            for key in keys:
                del self._cache[key]
            if keys:
                logger.info(f"[CACHE] cleared {len(keys)} keys with prefix={prefix}")
            return len(keys)

    def clear(self) -> int:
        with self._lock:
            size = len(self._cache)
            self._cache.clear()
            return size

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total else 0.0,
            }

    # ── 异步包装（供业务异步路径使用）────────────────────────────────

    async def aget(self, key: str) -> Any | None:
        return self.get(key)

    async def aset(self, key: str, value: Any) -> None:
        self.set(key, value)

    async def aclear_prefix(self, prefix: str) -> int:
        return self.clear_prefix(prefix)


class QueryCache:
    """检索结果缓存：key = kb_id + top_k + query。"""

    def __init__(self) -> None:
        self._cache = LRUCache(
            max_size=settings.QUERY_CACHE_MAX_SIZE,
            ttl=settings.QUERY_CACHE_TTL,
        )

    def _make_key(self, kb_id: int, query: str, k: int) -> str:
        raw = f"{kb_id}:{k}:{query.strip().lower()}"
        digest = hashlib.md5(raw.encode(), usedforsecurity=False).hexdigest()
        # 前缀包含 kb_id，便于按知识库精确失效
        return f"q:{kb_id}:{digest}"

    async def get(self, kb_id: int, query: str, k: int) -> Any | None:
        result = await self._cache.aget(self._make_key(kb_id, query, k))
        if result is not None:
            logger.debug(f"[CACHE] QueryCache HIT | kb={kb_id} query={query[:30]!r}")
        return result

    async def set(self, kb_id: int, query: str, k: int, value: Any) -> None:
        await self._cache.aset(self._make_key(kb_id, query, k), value)

    async def invalidate_kb(self, kb_id: int) -> int:
        """仅清除该知识库的检索缓存。"""
        cleared = await self._cache.aclear_prefix(f"q:{kb_id}:")
        logger.info(f"[CACHE] QueryCache invalidated kb={kb_id} keys={cleared}")
        return cleared

    async def invalidate_all(self) -> int:
        return self._cache.clear()

    def invalidate_kb_sync(self, kb_id: int) -> int:
        """同步失效接口：供同步代码路径（如删除文档）调用。"""
        cleared = self._cache.clear_prefix(f"q:{kb_id}:")
        logger.info(f"[CACHE] QueryCache invalidated kb={kb_id} keys={cleared}")
        return cleared

    def stats(self) -> dict[str, Any]:
        return self._cache.stats()


class EmbeddingCache:
    """向量缓存：key = embedding 身份（provider:model） + 文本 hash。

    带上 provider 与模型，切换后端/模型后不会命中旧向量。
    身份是动态读取的：不在 import 时固化 settings，避免配置非法（例如
    EMBEDDING_PROVIDER 写错）时连服务都启不来，而是等真正嵌入时报 503。
    """

    def __init__(self) -> None:
        self._cache = LRUCache(max_size=settings.EMBEDDING_CACHE_MAX_SIZE, ttl=0)

    def _make_key(self, text: str) -> str:
        digest = hashlib.sha256(text.encode()).hexdigest()
        return f"e:{settings.embedding_key}:{digest}"

    # 同步接口：供 LangChain 内部同步调用路径使用
    def get_sync(self, text: str) -> list[float] | None:
        return self._cache.get(self._make_key(text))

    def set_sync(self, text: str, vector: list[float]) -> None:
        self._cache.set(self._make_key(text), vector)

    # 异步接口
    async def get(self, text: str) -> list[float] | None:
        return self._cache.get(self._make_key(text))

    async def set(self, text: str, vector: list[float]) -> None:
        self._cache.set(self._make_key(text), vector)

    async def get_batch(self, texts: list[str]) -> tuple[list[list[float] | None], list[int]]:
        """批量查询，返回 (命中向量列表, 未命中下标)。"""
        vectors: list[list[float] | None] = []
        miss_indices: list[int] = []
        for i, text in enumerate(texts):
            cached = self._cache.get(self._make_key(text))
            vectors.append(cached)
            if cached is None:
                miss_indices.append(i)
        return vectors, miss_indices

    async def set_batch(self, texts: list[str], vectors: list[list[float]]) -> None:
        for text, vector in zip(texts, vectors, strict=True):
            self._cache.set(self._make_key(text), vector)

    def stats(self) -> dict[str, Any]:
        return self._cache.stats()


query_cache = QueryCache()
embedding_cache = EmbeddingCache()
