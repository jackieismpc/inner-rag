"""缓存插件点测试：LRU 引擎 + ``CacheBackend`` 契约 + 两个业务缓存。

契约用例用 ``backend`` fixture 参数化：新增一个后端（例如 Redis）只需在 ``BACKEND_FACTORIES``
里加一行，同一组用例就会跑到它身上（见 `docs/architecture.md` 3.4）。
"""

from __future__ import annotations

import time

import pytest

from inner_rag.services.cache import (
    CacheBackend,
    LRUCache,
    MemoryCacheBackend,
    build_cache_backend,
    embedding_cache,
    query_cache,
)

# ── LRU 引擎（内存后端的实现细节）─────────────────────────────────────


def test_lru_eviction() -> None:
    cache = LRUCache(max_size=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.get("a")  # a 变热，b 应该先被淘汰
    cache.set("c", 3)

    assert cache.get("a") == 1
    assert cache.get("b") is None
    assert cache.get("c") == 3


def test_lru_ttl_expiry() -> None:
    cache = LRUCache(max_size=4, ttl=1)
    cache.set("k", "v")
    assert cache.get("k") == "v"
    time.sleep(1.05)
    assert cache.get("k") is None


def test_clear_matching_is_glob_scoped() -> None:
    """`1:*` 不能连带删掉 kb=11：通配必须按模式匹配，而不是前缀字符串比较。"""
    cache = LRUCache(max_size=8)
    cache.set("1:abc", 1)
    cache.set("2:def", 2)
    cache.set("11:ghi", 3)

    assert cache.clear_matching("1:*") == 1
    assert cache.get("11:ghi") == 3
    assert cache.get("2:def") == 2


def test_stats_hit_rate() -> None:
    cache = LRUCache(max_size=2)
    cache.set("a", 1)
    cache.get("a")
    cache.get("missing")
    stats = cache.stats()

    assert stats["size"] == 1
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.5


# ── CacheBackend 契约 ─────────────────────────────────────────────────

BACKEND_FACTORIES = {
    "memory": lambda: build_cache_backend("memory"),
    "memory-no-limits": MemoryCacheBackend,
}


@pytest.fixture(params=sorted(BACKEND_FACTORIES), ids=sorted(BACKEND_FACTORIES))
def backend(request: pytest.FixtureRequest) -> CacheBackend:
    return BACKEND_FACTORIES[request.param]()


def test_backend_namespaces_are_isolated(backend: CacheBackend) -> None:
    backend.set("query", "k", "检索结果")
    backend.set("embedding", "k", "向量")

    assert backend.get("query", "k") == "检索结果"
    assert backend.get("embedding", "k") == "向量"

    # 清一个 namespace 不能影响另一个
    assert backend.invalidate("query") == 1
    assert backend.get("query", "k") is None
    assert backend.get("embedding", "k") == "向量"


def test_backend_invalidate_by_pattern_and_missing_key(backend: CacheBackend) -> None:
    backend.set("query", "1:a", 1)
    backend.set("query", "2:b", 2)

    assert backend.invalidate("query", "1:*") == 1
    assert backend.get("query", "2:b") == 2
    assert backend.invalidate("query", "1:*") == 0  # 幂等：没有可删的就返回 0
    assert backend.get("query", "never-set") is None


def test_backend_stats_are_per_namespace(backend: CacheBackend) -> None:
    backend.set("query", "k", "v")
    backend.get("query", "k")
    backend.get("query", "miss")

    query_stats = backend.stats("query")
    assert query_stats["size"] == 1
    assert query_stats["hits"] == 1
    assert query_stats["misses"] == 1


def test_build_cache_backend_rejects_unknown_name() -> None:
    with pytest.raises(ValueError) as excinfo:
        build_cache_backend("redis")

    message = str(excinfo.value)
    assert "CACHE_BACKEND" in message
    assert "memory" in message  # 可选值来自注册表


# ── 业务缓存 ──────────────────────────────────────────────────────────


async def test_query_cache_is_per_kb() -> None:
    await query_cache.set(1, "相同问题", 5, [("doc", 0.9)])
    await query_cache.set(2, "相同问题", 5, [("other", 0.5)])

    assert (await query_cache.get(1, "相同问题", 5))[0][0] == "doc"
    assert (await query_cache.get(2, "相同问题", 5))[0][0] == "other"
    assert await query_cache.get(1, "相同问题", 8) is None  # k 不同不能命中

    cleared = await query_cache.invalidate_kb(1)
    assert cleared == 1
    assert await query_cache.get(1, "相同问题", 5) is None
    assert await query_cache.get(2, "相同问题", 5) is not None


async def test_query_cache_normalizes_query() -> None:
    await query_cache.set(3, "  HeLLo  ", 2, "value")
    assert await query_cache.get(3, "hello", 2) == "value"


async def test_query_cache_sync_invalidation() -> None:
    await query_cache.set(7, "问题", 5, "v")
    assert query_cache.invalidate_kb_sync(7) == 1
    assert await query_cache.get(7, "问题", 5) is None


async def test_embedding_cache_round_trip_and_batch() -> None:
    """同步接口（LangChain 内部路径）与批量接口都要能命中同一份缓存。"""
    embedding_cache.set_sync("同步文本", [1.0, 2.0])
    assert embedding_cache.get_sync("同步文本") == [1.0, 2.0]

    await embedding_cache.set_batch(["a", "b"], [[1.0], [2.0]])
    vectors, miss_indices = await embedding_cache.get_batch(["a", "b", "c"])
    assert vectors[:2] == [[1.0], [2.0]]
    assert miss_indices == [2]
