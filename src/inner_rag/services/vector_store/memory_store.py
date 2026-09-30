"""内存向量库：零依赖的进程内实现。

它有两个用途：

1. **替换演练的产物**：验证「只加一个实现类 + 一次 register + 改一行配置」就能跑通
   上传 → 检索 → 问答，业务代码零改动（`services/rag.py`、路由、前端都不动）；
2. **最小可用后端**：单测与 CI 里不需要 zvec 原生库、不落盘，重启即干净。

边界（写清楚，避免被当成生产后端）：

* 数据只在进程内存里，**重启即丢**；
* 检索是纯 Python 线性扫描（O(n) × 向量维度），只适合几千分块的小库；
* 多进程部署时每个进程各持一份，**不共享**（zvec 内嵌模式同样要求单进程，原因不同）。

检索语义与 zvec / Chroma 严格对齐：距离口径同为 cosine 距离、MMR 与 hybrid 用
``base`` 里的同一份实现、阈值过滤与计数交给 ``finalize_results``。
"""

from __future__ import annotations

import heapq
from collections import Counter
from dataclasses import dataclass

from langchain_core.documents import Document

from inner_rag.services.embedding import embedding_service
from inner_rag.services.vector_store.base import (
    Strategy,
    cosine_similarity,
    distance_to_relevance,
    finalize_results,
    maximal_marginal_relevance,
    merge_hybrid,
    prepare_chunks,
    resolve_search_defaults,
)

# 分块在库内的唯一键：与 zvec 的 id 口径一致（kb_id 由外层的 bucket 表达）
ChunkKey = tuple[str, str]


@dataclass
class _StoredChunk:
    """一条分块记录：正文 + 元数据 + 向量。"""

    doc_id: str
    chunk_index: str
    document: Document
    vector: list[float]


class MemoryVectorStore:
    """进程内向量库：``{kb_id: {(doc_id, chunk_index): 分块}}``。"""

    def __init__(self) -> None:
        self._kbs: dict[int, dict[ChunkKey, _StoredChunk]] = {}

    # ── 写入 ───────────────────────────────────────────────────────────

    async def add_documents(
        self, kb_id: int, documents: list[Document], doc_id: int, filename: str
    ) -> int:
        chunks = prepare_chunks(documents, kb_id, doc_id, filename)
        if not chunks:
            return 0

        vectors = await embedding_service.aembed_documents([chunk.page_content for chunk in chunks])
        bucket = self._kbs.setdefault(kb_id, {})
        for chunk, vector in zip(chunks, vectors, strict=True):
            key: ChunkKey = (str(chunk.metadata["doc_id"]), str(chunk.metadata["chunk_index"]))
            # 用确定性 key 覆盖写：重跑入库（失败重试）不会累积重复分块
            bucket[key] = _StoredChunk(
                doc_id=key[0],
                chunk_index=key[1],
                document=chunk,
                vector=list(vector),
            )
        return len(chunks)

    # ── 检索 ───────────────────────────────────────────────────────────

    async def search(
        self,
        kb_id: int,
        query: str,
        k: int | None = None,
        strategy: Strategy = "similarity",
        score_threshold: float | None = None,
        filter_doc_ids: list[int] | None = None,
    ) -> tuple[list[tuple[Document, float | None]], int]:
        k, threshold = resolve_search_defaults(k, score_threshold)
        candidates = self._candidates(kb_id, filter_doc_ids)
        if not candidates:
            # 知识库建好了但一份文档都没入库：等价于「没有命中」，不是错误
            return [], 0

        vector = await embedding_service.aembed_query(query)
        # 距离用 cosine 距离（1 - 余弦相似度），这样后续换算与 zvec / Chroma 完全一致
        scored = [(chunk, 1.0 - cosine_similarity(vector, chunk.vector)) for chunk in candidates]

        if strategy == "mmr":
            picked = maximal_marginal_relevance(
                [(chunk.document, chunk.vector, distance) for chunk, distance in scored], k
            )
            raw: list[tuple[Document, float | None]] = [(doc, None) for doc in picked]
        elif strategy == "hybrid":
            # 与 zvec / Chroma 同策略：相似度取满 k 条，不足的部分用 MMR 去重补
            nearest = heapq.nsmallest(k, scored, key=_distance)
            raw = merge_hybrid(_as_results(nearest), self._mmr_docs(scored, k), k)
        else:
            raw = _as_results(heapq.nsmallest(k, scored, key=_distance))

        return finalize_results(raw, threshold)

    def _candidates(self, kb_id: int, filter_doc_ids: list[int] | None) -> list[_StoredChunk]:
        chunks = list(self._kbs.get(kb_id, {}).values())
        if filter_doc_ids:
            allowed = {str(doc_id) for doc_id in filter_doc_ids}
            chunks = [chunk for chunk in chunks if chunk.doc_id in allowed]
        return chunks

    @staticmethod
    def _mmr_docs(scored: list[tuple[_StoredChunk, float]], k: int) -> list[Document]:
        """MMR 候选：先取比 k 更多的近邻，再按多样性挑（与 zvec 的 ``fetch_k = k * 2`` 同量级）。"""
        pool = heapq.nsmallest(max(k * 2, k), scored, key=_distance)
        return maximal_marginal_relevance(
            [(chunk.document, chunk.vector, distance) for chunk, distance in pool], k
        )

    # ── 删除与统计 ─────────────────────────────────────────────────────

    async def delete_document(self, kb_id: int, doc_id: int) -> int:
        bucket = self._kbs.get(kb_id)
        if not bucket:
            return 0
        target = str(doc_id)
        keys = [key for key, chunk in bucket.items() if chunk.doc_id == target]
        for key in keys:
            del bucket[key]
        return len(keys)

    async def delete_kb(self, kb_id: int) -> None:
        # 幂等：从未入库过的知识库也算成功
        self._kbs.pop(kb_id, None)

    def count(self, kb_id: int) -> int:
        return len(self._kbs.get(kb_id, {}))

    def count_chunks_by_filename(self, kb_id: int) -> dict[str, int]:
        counter = Counter(
            str(chunk.document.metadata.get("filename", "?"))
            for chunk in self._kbs.get(kb_id, {}).values()
        )
        return dict(counter)

    def list_doc_ids(self, kb_id: int) -> list[str]:
        return sorted({chunk.doc_id for chunk in self._kbs.get(kb_id, {}).values()})


def _distance(item: tuple[_StoredChunk, float]) -> float:
    return item[1]


def _as_results(
    nearest: list[tuple[_StoredChunk, float]],
) -> list[tuple[Document, float | None]]:
    """距离 -> 相关度，口径与两个持久化后端一致。"""
    return [(chunk.document, distance_to_relevance(distance)) for chunk, distance in nearest]
