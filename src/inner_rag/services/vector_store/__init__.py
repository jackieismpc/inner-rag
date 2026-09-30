"""向量库门面：按 ``settings.VECTOR_STORE`` 选后端，业务层只依赖这里的契约。

它同时是既有的 import 入口（``from inner_rag.services.vector_store import vector_service``）：
后端实例（``vector_service``）与 ``Strategy`` / ``VectorStore`` / ``distance_to_relevance``
从包命名空间导出，所以换实现不需要改任何调用点。

后端名单由 `plugins.registry.vector_stores` 维护（与 chat / embedding / cache / queue 同一套机制）：
内置实现各自注册，第三方包可通过 entry point 组 ``inner_rag.vector_stores`` 追加，
因此这里没有「可选 zvec / chroma」这类需要跟着改的硬编码列表。
"""

from __future__ import annotations

from collections.abc import Callable

from inner_rag.core.config import settings
from inner_rag.plugins.registry import vector_stores
from inner_rag.services.vector_store.base import (
    Strategy,
    VectorStore,
    distance_to_relevance,
)

__all__ = [
    "Strategy",
    "VectorStore",
    "build_vector_store",
    "distance_to_relevance",
    "vector_service",
]


def _build_zvec() -> VectorStore:
    from inner_rag.services.vector_store.zvec_store import ZvecVectorStore

    return ZvecVectorStore()


def _build_chroma() -> VectorStore:
    from inner_rag.services.vector_store.chroma_store import ChromaVectorStore

    return ChromaVectorStore()


def _build_memory() -> VectorStore:
    from inner_rag.services.vector_store.memory_store import MemoryVectorStore

    return MemoryVectorStore()


# 注册顺序 = 文档与报错文案里的展示顺序
vector_stores.register("zvec", _build_zvec)
vector_stores.register("chroma", _build_chroma)
vector_stores.register("memory", _build_memory)


def build_vector_store(name: str | None = None) -> VectorStore:
    """按名字（默认取 ``settings.VECTOR_STORE``）构造后端实例。

    后端按需 import（见各 ``_build_*``）：用 zvec 时不必加载 chromadb（依赖很重），
    用 memory 时连 zvec 原生库都不用碰。配置写错时直接抛可读错误、并列出当前可选后端，
    不做静默降级——悄悄换后端会让「向量写进哪个库」变成不可预期的事。
    """
    vector_stores.load_entry_points()
    backend = (name or settings.VECTOR_STORE).strip().lower()
    factory: Callable[[], VectorStore] | None = vector_stores.get(backend)
    if factory is None:
        msg = f"不支持的 VECTOR_STORE: {backend!r}（可选 {'/'.join(vector_stores.names())}）"
        raise ValueError(msg)
    return factory()


vector_service = build_vector_store()
