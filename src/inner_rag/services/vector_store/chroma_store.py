"""Chroma 适配器（兼容实现）：每知识库一个 collection，cosine + HNSW。

Phase 4 起 zvec 是默认后端，本实现保留用于：契约测试的第二套实现、已有知识库的数据读取，
以及需要「跨进程只读」这类 zvec 内嵌模式给不了的能力时的退路（切换见 ``VECTOR_STORE``）。
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Mapping
from typing import Any

import chromadb
from chromadb.api.collection_configuration import CreateCollectionConfiguration
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from loguru import logger

from inner_rag.core.config import settings
from inner_rag.services.embedding import embedding_service
from inner_rag.services.vector_store.base import (
    WRITE_BATCH_SIZE,
    Strategy,
    distance_to_relevance,
    finalize_results,
    merge_hybrid,
    prepare_chunks,
    resolve_search_defaults,
)

# 建集合时指定的距离空间（Chroma 1.x 的现代写法，等价于旧的 "hnsw:space" metadata）
COSINE_SPACE_CONFIG: CreateCollectionConfiguration = {"hnsw": {"space": "cosine"}}


class ChromaVectorStore:
    """Chroma 后端：持久化客户端 + 每知识库一个 collection。"""

    def __init__(self) -> None:
        self._client: chromadb.ClientAPI | None = None
        self._stores: dict[int, Chroma] = {}

    # ── 基础设施 ───────────────────────────────────────────────────────

    def _get_client(self) -> chromadb.ClientAPI:
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIR,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._client

    @staticmethod
    def _collection_name(kb_id: int) -> str:
        return f"kb_{kb_id}"

    def _get_store(self, kb_id: int) -> Chroma:
        if kb_id not in self._stores:
            self._stores[kb_id] = Chroma(
                client=self._get_client(),
                collection_name=self._collection_name(kb_id),
                embedding_function=embedding_service,
                collection_configuration=COSINE_SPACE_CONFIG,
                create_collection_if_not_exists=True,
            )
        return self._stores[kb_id]

    # ── 写入 ───────────────────────────────────────────────────────────

    async def add_documents(
        self, kb_id: int, documents: list[Document], doc_id: int, filename: str
    ) -> int:
        chunks = prepare_chunks(documents, kb_id, doc_id, filename)
        if not chunks:
            logger.warning(f"[VECTOR_STORE] doc_id={doc_id} 未产生任何有效分块")
            return 0

        texts = [chunk.page_content for chunk in chunks]
        started = time.perf_counter()
        # 先异步批量嵌入：结果进入 EmbeddingCache，随后 Chroma 内部再次嵌入时直接命中缓存
        await embedding_service.aembed_documents(texts)
        logger.info(
            f"[EMBED] doc_id={doc_id} | chunks={len(chunks)} | "
            f"embed_time={(time.perf_counter() - started) * 1000:.1f}ms"
        )

        store = self._get_store(kb_id)
        for start in range(0, len(chunks), WRITE_BATCH_SIZE):
            batch = chunks[start : start + WRITE_BATCH_SIZE]
            # Chroma 侧 id 用随机 uuid：重跑入库前必须先 delete_document，否则分块会累积
            ids = [uuid.uuid4().hex for _ in batch]
            await asyncio.to_thread(store.add_documents, batch, ids=ids)

        logger.info(
            f"[VECTOR_STORE] 写入 kb={kb_id} doc_id={doc_id} "
            f"filename={filename} chunks={len(chunks)}"
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
        store = self._get_store(kb_id)
        where = self._build_filter(filter_doc_ids)

        if strategy == "mmr":
            docs = await asyncio.to_thread(
                store.max_marginal_relevance_search,
                query,
                k=k,
                fetch_k=k * 3,
                filter=where,
            )
            raw: list[tuple[Document, float | None]] = [(doc, None) for doc in docs]
        elif strategy == "hybrid":
            raw = await self._hybrid_search(store, query, k, where)
        else:
            scored = await asyncio.to_thread(
                store.similarity_search_with_score, query, k=k, filter=where
            )
            raw = [(doc, distance_to_relevance(distance)) for doc, distance in scored]

        return finalize_results(raw, threshold)

    async def _hybrid_search(
        self, store: Chroma, query: str, k: int, where: dict | None
    ) -> list[tuple[Document, float | None]]:
        """相似度检索 + MMR 去重补充，兼顾精确召回与结果多样性。"""
        scored = await asyncio.to_thread(
            store.similarity_search_with_score, query, k=k, filter=where
        )
        raw: list[tuple[Document, float | None]] = [
            (doc, distance_to_relevance(distance)) for doc, distance in scored
        ]
        if len(raw) >= k:
            return raw

        mmr_docs = await asyncio.to_thread(
            store.max_marginal_relevance_search,
            query,
            k=max(1, k - len(raw)),
            fetch_k=k * 2,
            filter=where,
        )
        return merge_hybrid(raw, mmr_docs, k)

    @staticmethod
    def _build_filter(filter_doc_ids: list[int] | None) -> dict | None:
        if not filter_doc_ids:
            return None
        if len(filter_doc_ids) == 1:
            return {"doc_id": str(filter_doc_ids[0])}
        return {"doc_id": {"$in": [str(doc_id) for doc_id in filter_doc_ids]}}

    # ── 删除与统计 ─────────────────────────────────────────────────────

    async def delete_document(self, kb_id: int, doc_id: int) -> int:
        return await asyncio.to_thread(self._delete_document_sync, kb_id, doc_id)

    def _delete_document_sync(self, kb_id: int, doc_id: int) -> int:
        try:
            collection = self._get_client().get_collection(self._collection_name(kb_id))
            result = collection.get(where={"doc_id": str(doc_id)})
            ids = result.get("ids") if result else None
            if ids:
                collection.delete(ids=ids)
                logger.info(f"[VECTOR_STORE] 删除 {len(ids)} 个向量 kb={kb_id} doc_id={doc_id}")
            self._stores.pop(kb_id, None)
            return len(ids) if ids else 0
        except Exception as exc:
            logger.error(f"[VECTOR_STORE] 删除文档向量失败 kb={kb_id} doc_id={doc_id}: {exc}")
            return 0

    async def delete_kb(self, kb_id: int) -> None:
        await asyncio.to_thread(self._delete_kb_sync, kb_id)

    def _delete_kb_sync(self, kb_id: int) -> None:
        name = self._collection_name(kb_id)
        try:
            client = self._get_client()
            existing = {collection.name for collection in client.list_collections()}
            if name in existing:
                client.delete_collection(name)
                logger.info(f"[VECTOR_STORE] 删除 collection kb={kb_id}")
            else:
                # 知识库建立但从未成功入库时 collection 并不存在，属于正常情况
                logger.debug(f"[VECTOR_STORE] collection 不存在，无需删除 kb={kb_id}")
        except Exception as exc:
            logger.error(f"[VECTOR_STORE] 删除 collection 失败 kb={kb_id}: {exc}")
        finally:
            self._stores.pop(kb_id, None)

    def count(self, kb_id: int) -> int:
        try:
            return self._get_client().get_collection(self._collection_name(kb_id)).count()
        except Exception:
            # 从未入库过的知识库没有 collection，等价于 0 个分块
            return 0

    def count_chunks_by_filename(self, kb_id: int) -> dict[str, int]:
        metadatas = self._all_metadatas(kb_id)
        counts: dict[str, int] = {}
        for metadata in metadatas:
            filename = str(metadata.get("filename", "?"))
            counts[filename] = counts.get(filename, 0) + 1
        return counts

    def list_doc_ids(self, kb_id: int) -> list[str]:
        return [str(metadata.get("doc_id")) for metadata in self._all_metadatas(kb_id)]

    def iter_chunks(self, kb_id: int) -> list[Document]:
        """一次性取回全部分块（正文 + 元数据），供建倒排索引 / 对账用。"""
        try:
            collection = self._get_client().get_collection(self._collection_name(kb_id))
            result = collection.get(include=["documents", "metadatas"], limit=100000)
        except Exception:
            return []
        documents = (result or {}).get("documents") or []
        metadatas = (result or {}).get("metadatas") or []
        return [
            Document(page_content=text or "", metadata=dict(metadata or {}))
            for text, metadata in zip(documents, metadatas, strict=False)
        ]

    def _all_metadatas(self, kb_id: int) -> list[Mapping[str, Any]]:
        try:
            collection = self._get_client().get_collection(self._collection_name(kb_id))
            result = collection.get(include=["metadatas"], limit=100000)
        except Exception:
            return []
        return (result or {}).get("metadatas") or []
