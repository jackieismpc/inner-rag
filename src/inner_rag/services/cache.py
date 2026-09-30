"""缓存层：``CacheBackend`` 契约 + 内存实现，以及两个业务缓存（检索结果 / 向量）。

Phase 7 起缓存也是插件点（``plugins.cache_backends``）：换后端只需加一个实现 + 改
``CACHE_BACKEND``，业务侧（``QueryCache`` / ``EmbeddingCache``）与调用方都不变。

契约只有四个方法，两处约定值得注意：

* **namespace 隔离**：不同业务各自一个命名空间（``query`` / ``embedding``），互不可见，
  因此失效不会误伤另一个业务；
* **失效用 fnmatch 通配**（``f"{kb_id}:*"``）：本项目的失效都是「某个知识库的全部检索结果」，
  通配语义与 Redis 的 ``SCAN MATCH`` 一致，换成远程后端时不用改调用方。

已知边界：进程内实现是单 worker 语义，多副本部署时每个副本各有一份缓存（见风险登记簿）；
换成网络后端（Redis）时应在实现内部处理连接与超时，**不要让缓存故障影响请求**。
"""

from __future__ import annotations

import fnmatch
import hashlib
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, Protocol

from loguru import logger

from inner_rag.core.config import settings
from inner_rag.plugins.registry import cache_backends

# 未显式声明限额的 namespace 用它兜底：有界总比无界安全
DEFAULT_LIMIT = (256, 0)


class LRUCache:
    """有界 LRU（可选 TTL）。内部同步 + ``RLock`` 保护。

    保持同步实现并用锁保护，是为了让「LangChain 向量库内部的同步嵌入路径」与
    「业务异步路径」能共享同一份缓存，避免同一段文本被重复嵌入。
    """

    def __init__(self, max_size: int, ttl: int = 0) -> None:
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl  # 0 表示永不过期
        self._hits = 0
        self._misses = 0
        self._lock = threading.RLock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            item = self._cache.get(key)
            if item is None:
                self._misses += 1
                return None
            value, ts = item
            # monotonic：墙钟跳变（NTP 校正）不该让缓存提前过期
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

    def clear_matching(self, pattern: str) -> int:
        """按 fnmatch 通配删除，返回删除条数。"""
        with self._lock:
            keys = [key for key in self._cache if fnmatch.fnmatchcase(key, pattern)]
            for key in keys:
                del self._cache[key]
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


class CacheBackend(Protocol):
    """缓存后端契约（五件套里的「接口」）。

    ``get`` 未命中返回 ``None``：缓存里不存 ``None``，因此不存在「命中了但值是 None」的歧义。
    """

    def get(self, namespace: str, key: str) -> Any | None: ...

    def set(self, namespace: str, key: str, value: Any, ttl: float | None = None) -> None: ...

    def invalidate(self, namespace: str, pattern: str | None = None) -> int: ...

    def stats(self, namespace: str | None = None) -> dict[str, Any]: ...


class MemoryCacheBackend:
    """进程内 LRU 后端：每个 namespace 一份 LRU，限额由构造参数决定。"""

    name = "memory"

    def __init__(self, limits: dict[str, tuple[int, int]] | None = None) -> None:
        self._limits = limits or {}
        self._caches: dict[str, LRUCache] = {}
        self._lock = threading.RLock()

    def _cache_for(self, namespace: str) -> LRUCache:
        with self._lock:
            cache = self._caches.get(namespace)
            if cache is None:
                max_size, ttl = self._limits.get(namespace, DEFAULT_LIMIT)
                cache = LRUCache(max_size=max_size, ttl=ttl)
                self._caches[namespace] = cache
            return cache

    def get(self, namespace: str, key: str) -> Any | None:
        return self._cache_for(namespace).get(key)

    def set(self, namespace: str, key: str, value: Any, ttl: float | None = None) -> None:
        # 内存后端的 TTL 由构造限额决定，不支持按条覆盖：TTL 配置一处就够，多个来源反而容易打架
        self._cache_for(namespace).set(key, value)

    def invalidate(self, namespace: str, pattern: str | None = None) -> int:
        cache = self._cache_for(namespace)
        cleared = cache.clear_matching(pattern) if pattern else cache.clear()
        if cleared:
            logger.info(
                f"[CACHE] invalidated namespace={namespace} pattern={pattern} keys={cleared}"
            )
        return cleared

    def stats(self, namespace: str | None = None) -> dict[str, Any]:
        if namespace is not None:
            return self._cache_for(namespace).stats()
        with self._lock:
            namespaces = {name: cache.stats() for name, cache in self._caches.items()}
        return {"backend": self.name, "namespaces": namespaces}


def _build_memory_cache() -> CacheBackend:
    return MemoryCacheBackend(
        limits={
            QueryCache.NAMESPACE: (settings.QUERY_CACHE_MAX_SIZE, settings.QUERY_CACHE_TTL),
            EmbeddingCache.NAMESPACE: (settings.EMBEDDING_CACHE_MAX_SIZE, 0),
        }
    )


cache_backends.register("memory", _build_memory_cache)


def build_cache_backend(name: str | None = None) -> CacheBackend:
    """按名字（默认取 ``CACHE_BACKEND``）构造缓存后端。

    配置写错直接抛可读错误，不做静默降级——悄悄换成另一个后端会让「缓存里到底有什么」
    变得不可预期（与 ``build_vector_store`` 同一原则）。
    """
    cache_backends.load_entry_points()
    backend = (name or settings.CACHE_BACKEND).strip().lower()
    factory: Callable[[], CacheBackend] | None = cache_backends.get(backend)
    if factory is None:
        msg = f"不支持的 CACHE_BACKEND: {backend!r}（可选 {'/'.join(cache_backends.names())}）"
        raise ValueError(msg)
    return factory()


class QueryCache:
    """检索结果缓存：key = kb_id + top_k + query。

    保留 async 方法是为了调用方无感：当前后端是进程内的（微秒级），换网络后端时
    调用点不用改（届时在实现内部做 I/O 与超时控制）。
    """

    NAMESPACE = "query"

    def __init__(self, backend: CacheBackend) -> None:
        self._backend = backend

    @staticmethod
    def _make_key(kb_id: int, query: str, k: int) -> str:
        raw = f"{kb_id}:{k}:{query.strip().lower()}"
        digest = hashlib.md5(raw.encode(), usedforsecurity=False).hexdigest()
        # 前缀带 kb_id：失效时可以用 `{kb_id}:*` 精确到单个知识库
        return f"{kb_id}:{digest}"

    async def get(self, kb_id: int, query: str, k: int) -> Any | None:
        result = self._backend.get(self.NAMESPACE, self._make_key(kb_id, query, k))
        if result is not None:
            logger.debug(f"[CACHE] QueryCache HIT | kb={kb_id} query={query[:30]!r}")
        return result

    async def set(self, kb_id: int, query: str, k: int, value: Any) -> None:
        self._backend.set(self.NAMESPACE, self._make_key(kb_id, query, k), value)

    async def invalidate_kb(self, kb_id: int) -> int:
        """仅清除该知识库的检索缓存。"""
        return self._backend.invalidate(self.NAMESPACE, f"{kb_id}:*")

    def invalidate_kb_sync(self, kb_id: int) -> int:
        """同步失效接口：供同步代码路径（如删除文档）调用。"""
        return self._backend.invalidate(self.NAMESPACE, f"{kb_id}:*")

    def clear(self) -> int:
        return self._backend.invalidate(self.NAMESPACE)

    def stats(self) -> dict[str, Any]:
        return self._backend.stats(self.NAMESPACE)


class EmbeddingCache:
    """向量缓存：key = embedding 身份（provider:model） + 文本 hash。

    带上 provider 与模型，切换后端/模型后不会命中旧向量。
    身份是动态读取的：不在 import 时固化 settings，避免配置非法（例如
    EMBEDDING_PROVIDER 写错）时连服务都启不来，而是等真正嵌入时报 503。
    """

    NAMESPACE = "embedding"

    def __init__(self, backend: CacheBackend) -> None:
        self._backend = backend

    def _make_key(self, text: str) -> str:
        digest = hashlib.sha256(text.encode()).hexdigest()
        return f"{settings.embedding_key}:{digest}"

    # 同步接口：供 LangChain 内部同步调用路径使用
    def get_sync(self, text: str) -> list[float] | None:
        return self._backend.get(self.NAMESPACE, self._make_key(text))

    def set_sync(self, text: str, vector: list[float]) -> None:
        self._backend.set(self.NAMESPACE, self._make_key(text), vector)

    # 异步接口
    async def get(self, text: str) -> list[float] | None:
        return self._backend.get(self.NAMESPACE, self._make_key(text))

    async def set(self, text: str, vector: list[float]) -> None:
        self._backend.set(self.NAMESPACE, self._make_key(text), vector)

    async def get_batch(self, texts: list[str]) -> tuple[list[list[float] | None], list[int]]:
        """批量查询，返回 (命中向量列表, 未命中下标)。"""
        vectors: list[list[float] | None] = []
        miss_indices: list[int] = []
        for i, text in enumerate(texts):
            cached = self._backend.get(self.NAMESPACE, self._make_key(text))
            vectors.append(cached)
            if cached is None:
                miss_indices.append(i)
        return vectors, miss_indices

    async def set_batch(self, texts: list[str], vectors: list[list[float]]) -> None:
        for text, vector in zip(texts, vectors, strict=True):
            self._backend.set(self.NAMESPACE, self._make_key(text), vector)

    def stats(self) -> dict[str, Any]:
        return self._backend.stats(self.NAMESPACE)

    def clear(self) -> int:
        """清空向量缓存（测试与运维用；向量缓存按内容寻址，正常不会被「弄脏」）。"""
        return self._backend.invalidate(self.NAMESPACE)


cache_backend = build_cache_backend()
query_cache = QueryCache(cache_backend)
embedding_cache = EmbeddingCache(cache_backend)
