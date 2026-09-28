"""缓存层测试。"""

from __future__ import annotations

import time

from inner_rag.services.cache import LRUCache, embedding_cache, query_cache


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


def test_clear_prefix() -> None:
    cache = LRUCache(max_size=8)
    cache.set("q:1:x", 1)
    cache.set("q:2:y", 2)
    cache.set("e:1:z", 3)

    assert cache.clear_prefix("q:1:") == 1
    assert cache.get("q:2:y") == 2
    assert cache.get("e:1:z") == 3


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


def test_query_cache_sync_invalidation() -> None:
    query_cache._cache.set("q:7:abc", "v")
    assert query_cache.invalidate_kb_sync(7) == 1
    assert query_cache._cache.get("q:7:abc") is None


async def test_embedding_cache_keys_include_model_identity() -> None:
    await embedding_cache.set("文本", [1.0, 2.0])
    assert await embedding_cache.get("文本") == [1.0, 2.0]
    assert embedding_cache.stats()["size"] >= 1
