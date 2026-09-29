"""向量库门面：按 ``settings.VECTOR_STORE`` 选后端，业务层只依赖这里的契约。

它同时是既有的 import 入口（``from inner_rag.services.vector_store import vector_service``）：
后端实例（``vector_service``）与 ``Strategy`` / ``VectorStore`` / ``distance_to_relevance``
从包命名空间导出，所以换实现不需要改任何调用点。
"""

from __future__ import annotations

from inner_rag.core.config import settings
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


def build_vector_store(name: str | None = None) -> VectorStore:
    """按名字（默认取 ``settings.VECTOR_STORE``）构造后端实例。

    后端按需 import：用 zvec 时不必加载 chromadb（依赖很重），将来 Chroma 从必装依赖降级为
    optional extra 也不需要改这里的代码。配置写错时直接抛可读错误，不做静默降级——
    悄悄换后端会让「向量写进哪个库」变成不可预期的事。
    """
    backend = (name or settings.VECTOR_STORE).strip().lower()
    if backend == "zvec":
        from inner_rag.services.vector_store.zvec_store import ZvecVectorStore

        return ZvecVectorStore()
    if backend == "chroma":
        from inner_rag.services.vector_store.chroma_store import ChromaVectorStore

        return ChromaVectorStore()
    msg = f"不支持的 VECTOR_STORE: {backend!r}（可选 zvec / chroma）"
    raise ValueError(msg)


vector_service = build_vector_store()
