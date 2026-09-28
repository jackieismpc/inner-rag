"""向量存储服务：ChromaDB 持久化，每个知识库一个独立 collection。

距离空间固定为 cosine，因此：
    距离 distance = 1 - 余弦相似度   ->   相关度 relevance = 1 - distance

注意：langchain-chroma 1.1.0 的 ``similarity_search_with_relevance_scores`` 虽然
名为 relevance，实际返回的仍是未做任何转换的原始距离，因此这里统一使用
``similarity_search_with_score`` 拿到距离后自行换算，避免出现「相关度 0%/100%」之类的错误展示。
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Literal

import chromadb
from chromadb.api.collection_configuration import CreateCollectionConfiguration
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from inner_rag.core.config import settings
from inner_rag.services.embedding import embedding_service

Strategy = Literal["similarity", "mmr", "hybrid"]

# 建集合时指定的距离空间（Chroma 1.x 的现代写法，等价于旧的 "hnsw:space" metadata）
COSINE_SPACE_CONFIG: CreateCollectionConfiguration = {"hnsw": {"space": "cosine"}}

WRITE_BATCH_SIZE = 50


class EmbeddingIdentityMismatch(RuntimeError):
    """知识库记录的 embedding 与当前配置不一致。"""


def distance_to_relevance(distance: float) -> float:
    """cosine 距离 -> [0, 1] 相关度。"""
    return max(0.0, min(1.0, 1.0 - float(distance)))


class VectorStoreService:
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

    def _collection_name(self, kb_id: int) -> str:
        return f"kb_{kb_id}"

    def get_store(self, kb_id: int) -> Chroma:
        if kb_id not in self._stores:
            self._stores[kb_id] = Chroma(
                client=self._get_client(),
                collection_name=self._collection_name(kb_id),
                embedding_function=embedding_service,
                collection_configuration=COSINE_SPACE_CONFIG,
                create_collection_if_not_exists=True,
            )
        return self._stores[kb_id]

    def _text_splitter(self) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""],
            length_function=len,
        )

    def ensure_embedding_matches(self, kb_id: int, embedding_model: str | None) -> None:
        """校验知识库的向量空间与当前 embedding 配置一致，避免静默检索到无意义的向量。"""
        if embedding_model and embedding_model != embedding_service.identity:
            msg = (
                f"知识库 kb={kb_id} 建库时使用 {embedding_model}，"
                f"当前配置为 {embedding_service.identity}；"
                "请切回原 embedding 模型，或执行 scripts/reindex_kb.py 重建索引"
            )
            raise EmbeddingIdentityMismatch(msg)

    # ── 写入 ───────────────────────────────────────────────────────────

    async def add_documents_async(
        self,
        kb_id: int,
        documents: list[Document],
        doc_id: int,
        filename: str,
    ) -> int:
        """分块 + 批量嵌入 + 写入 Chroma。返回写入的分块数。"""
        chunks = [
            chunk
            for chunk in self._text_splitter().split_documents(documents)
            if chunk.page_content.strip()
        ]
        if not chunks:
            logger.warning(f"[VECTOR_STORE] doc_id={doc_id} 未产生任何有效分块")
            return 0

        for index, chunk in enumerate(chunks):
            # Chroma 过滤要求元数据为标量，统一存字符串
            chunk.metadata.update(
                {
                    "doc_id": str(doc_id),
                    "kb_id": str(kb_id),
                    "filename": filename,
                    "chunk_index": str(index),
                }
            )

        texts = [chunk.page_content for chunk in chunks]
        started = time.perf_counter()
        # 先异步批量嵌入：结果进入 EmbeddingCache，随后 Chroma 内部再次嵌入时直接命中缓存
        await embedding_service.aembed_documents(texts)
        logger.info(
            f"[EMBED] doc_id={doc_id} | chunks={len(chunks)} | "
            f"embed_time={(time.perf_counter() - started) * 1000:.1f}ms"
        )

        store = self.get_store(kb_id)
        for start in range(0, len(chunks), WRITE_BATCH_SIZE):
            batch = chunks[start : start + WRITE_BATCH_SIZE]
            ids = [uuid.uuid4().hex for _ in batch]
            await asyncio.to_thread(store.add_documents, batch, ids=ids)

        logger.info(
            f"[VECTOR_STORE] 写入 kb={kb_id} doc_id={doc_id} "
            f"filename={filename} chunks={len(chunks)}"
        )
        return len(chunks)

    # ── 检索 ───────────────────────────────────────────────────────────

    async def similarity_search_async(
        self,
        kb_id: int,
        query: str,
        k: int | None = None,
        strategy: Strategy = "similarity",
        score_threshold: float | None = None,
        filter_doc_ids: list[int] | None = None,
    ) -> tuple[list[tuple[Document, float | None]], int]:
        """检索。返回 (结果列表, 被阈值过滤掉的数量)。

        结果元素为 (Document, relevance)，其中 relevance 为 [0, 1] 的余弦相关度；
        MMR 单独召回的条目没有分数，其 relevance 为 None（不做阈值判断）。
        """
        k = k or settings.TOP_K
        threshold = (
            settings.RETRIEVAL_SCORE_THRESHOLD if score_threshold is None else score_threshold
        )
        store = self.get_store(kb_id)
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

        filtered: list[tuple[Document, float | None]] = []
        filtered_out = 0
        for doc, score in raw:
            if score is not None and score < threshold:
                filtered_out += 1
                logger.debug(
                    f"[FILTER] relevance={score:.4f} < {threshold} "
                    f"file={doc.metadata.get('filename', '?')} "
                    f"chunk={doc.metadata.get('chunk_index', '?')}"
                )
                continue
            filtered.append((doc, score))

        # 有分数的按相关度降序，无分数的排在最后
        filtered.sort(
            key=lambda item: (item[1] is not None, item[1] if item[1] is not None else 0.0),
            reverse=True,
        )
        return filtered, filtered_out

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
        seen = {self._chunk_key(doc) for doc, _ in raw}
        for doc in mmr_docs:
            if len(raw) >= k:
                break
            key = self._chunk_key(doc)
            if key in seen:
                continue
            seen.add(key)
            raw.append((doc, None))
        return raw

    @staticmethod
    def _chunk_key(doc: Document) -> tuple[str | None, str | None]:
        """分块唯一标识，用于混合检索去重（比 id(doc) 可靠）。"""
        return doc.metadata.get("doc_id"), doc.metadata.get("chunk_index")

    @staticmethod
    def _build_filter(filter_doc_ids: list[int] | None) -> dict | None:
        if not filter_doc_ids:
            return None
        if len(filter_doc_ids) == 1:
            return {"doc_id": str(filter_doc_ids[0])}
        return {"doc_id": {"$in": [str(doc_id) for doc_id in filter_doc_ids]}}

    # ── 删除与统计 ─────────────────────────────────────────────────────

    def delete_documents(self, kb_id: int, doc_id: int) -> int:
        """删除某文档的所有向量，返回删除条数。"""
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

    def delete_kb(self, kb_id: int) -> None:
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

    def get_kb_stats(self, kb_id: int) -> dict[str, int]:
        try:
            collection = self._get_client().get_collection(self._collection_name(kb_id))
            return {"vector_count": collection.count()}
        except Exception:
            return {"vector_count": 0}

    def count_chunks_by_filename(self, kb_id: int) -> dict[str, int]:
        """统计该知识库中各文件的分块数（排查索引用）。"""
        try:
            collection = self._get_client().get_collection(self._collection_name(kb_id))
            result = collection.get(include=["metadatas"], limit=100000)
        except Exception:
            return {}

        counts: dict[str, int] = {}
        for metadata in (result or {}).get("metadatas") or []:
            filename = str(metadata.get("filename", "?"))
            counts[filename] = counts.get(filename, 0) + 1
        return counts

    def list_doc_ids(self, kb_id: int) -> list[str]:
        """列出 collection 中出现过的 doc_id（用于发现删除后的残留向量）。"""
        try:
            collection = self._get_client().get_collection(self._collection_name(kb_id))
            result = collection.get(include=["metadatas"], limit=100000)
        except Exception:
            return []
        return [str(m.get("doc_id")) for m in (result or {}).get("metadatas") or []]


vector_service = VectorStoreService()
