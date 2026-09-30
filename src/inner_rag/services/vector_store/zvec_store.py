"""zvec 适配器：内嵌（embedded）模式，每个知识库一个 collection（``ZVEC_PATH/kb_<id>/``）。

zvec 0.7.0 的实测行为决定了这里的实现方式（改动前先把对应的实测脚本跑一遍）：

* **维度必须显式给出**，而维度由 embedding 模型决定 → collection 在**首次写入**时按刚算出的
  向量维度创建，省掉一份「向量维度」配置；维度不一致由 zvec 直接报可读错误。
* **id 不允许冒号**，且 ``insert`` 撞 id 只返回错误状态码而不抛异常 → 分块 id 用短横线组装、
  写入统一用 ``upsert``，重跑入库（漏删 / 失败重试）覆盖同一分块而不是累积重复。
* **不存原文**：与 Chroma 不同，zvec 只存向量与声明的标量字段，所以分块正文要显式作为一个
  STRING 字段存下来，检索时才能还原成 ``Document``。
* **COSINE 下 ``Doc.score`` 就是 ``1 - 余弦相似度``**（即距离口径），与 Chroma 一致，
  因此共用 ``distance_to_relevance``。
* **没有 MMR API** → 用 ``base.maximal_marginal_relevance``（与内存后端共用同一份实现）。
* **写锁按 collection 目录独占，跨进程互斥**（实测：写者持有时另一个进程连只读都打不开）
  → 内嵌模式必须单进程部署，不要开 ``uvicorn --workers``；详见 ``docs/architecture.md`` 3.2。
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import cast

import zvec
from langchain_core.documents import Document
from loguru import logger

from inner_rag.core.config import settings
from inner_rag.services.embedding import embedding_service
from inner_rag.services.vector_store.base import (
    CHUNK_METADATA_FIELDS,
    MMR_FETCH_FACTOR,
    WRITE_BATCH_SIZE,
    Strategy,
    distance_to_relevance,
    finalize_results,
    maximal_marginal_relevance,
    merge_hybrid,
    prepare_chunks,
    resolve_search_defaults,
)

# 向量字段名（schema 与 query 两侧共用，改名要同时改）
VECTOR_FIELD = "embedding"
# 分块正文：zvec 不保存原文，检索结果要靠它还原 Document.page_content
CONTENT_FIELD = "content"
# 检索时要取回的字段（正文 + 白名单元数据）
_OUTPUT_FIELDS = [CONTENT_FIELD, *CHUNK_METADATA_FIELDS]


class ZvecVectorStore:
    """zvec 后端：每知识库一个 collection，句柄按 kb 缓存。"""

    def __init__(self) -> None:
        self._collections: dict[int, zvec.Collection] = {}

    # ── collection 生命周期 ────────────────────────────────────────────

    @staticmethod
    def _path(kb_id: int) -> str:
        return str(Path(settings.ZVEC_PATH) / f"kb_{kb_id}")

    def _existing(self, kb_id: int) -> zvec.Collection | None:
        """已建好的 collection；知识库从未成功入库时返回 None（磁盘上没有目录）。"""
        collection = self._collections.get(kb_id)
        if collection is not None:
            return collection
        path = self._path(kb_id)
        if not os.path.exists(path):
            return None
        collection = zvec.open(path, zvec.CollectionOption(read_only=False, enable_mmap=True))
        self._collections[kb_id] = collection
        return collection

    def _writable(self, kb_id: int, dimension: int) -> zvec.Collection:
        """写入路径：collection 不存在时用首批向量的维度创建。"""
        collection = self._existing(kb_id)
        if collection is not None:
            return collection
        collection = zvec.create_and_open(self._path(kb_id), _schema(kb_id, dimension))
        self._collections[kb_id] = collection
        return collection

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
        vectors = await embedding_service.aembed_documents(texts)
        logger.info(
            f"[EMBED] doc_id={doc_id} | chunks={len(chunks)} | "
            f"embed_time={(time.perf_counter() - started) * 1000:.1f}ms"
        )

        await asyncio.to_thread(self._write_chunks, kb_id, doc_id, chunks, vectors)
        logger.info(
            f"[VECTOR_STORE] 写入 kb={kb_id} doc_id={doc_id} "
            f"filename={filename} chunks={len(chunks)}"
        )
        return len(chunks)

    def _write_chunks(
        self, kb_id: int, doc_id: int, chunks: list[Document], vectors: list[list[float]]
    ) -> None:
        """分批 upsert → flush → optimize（全在 native 侧，调用方放进线程）。"""
        collection = self._writable(kb_id, len(vectors[0]))
        for start in range(0, len(chunks), WRITE_BATCH_SIZE):
            batch = chunks[start : start + WRITE_BATCH_SIZE]
            docs = [
                zvec.Doc(
                    id=_chunk_id(kb_id, doc_id, str(chunk.metadata["chunk_index"])),
                    vectors={VECTOR_FIELD: vectors[start + offset]},
                    fields={**chunk.metadata, CONTENT_FIELD: chunk.page_content},
                )
                for offset, chunk in enumerate(batch)
            ]
            _ensure_ok(collection.upsert(docs), f"upsert kb={kb_id} doc_id={doc_id}")
        collection.flush()
        # optimize 之后索引完备度才到 1.0：不调用也能查到（走未索引段的暴力扫描），
        # 但代价随库增长，因此在每次入库结束时补上
        collection.optimize()

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
        collection = self._existing(kb_id)
        if collection is None:
            # 知识库建好了但一份文档都没入库：等价于「没有命中」，不是错误
            return [], 0

        filter_expr = _filter_expression(filter_doc_ids)
        vector = await embedding_service.aembed_query(query)

        if strategy == "mmr":
            docs = await asyncio.to_thread(
                self._mmr, collection, vector, k, k * MMR_FETCH_FACTOR, filter_expr
            )
            raw: list[tuple[Document, float | None]] = [(doc, None) for doc in docs]
        elif strategy == "hybrid":
            raw = await asyncio.to_thread(self._hybrid, collection, vector, k, filter_expr)
        else:
            scored = await asyncio.to_thread(self._similarity, collection, vector, k, filter_expr)
            raw = [(doc, distance_to_relevance(distance)) for doc, distance in scored]

        return finalize_results(raw, threshold)

    @staticmethod
    def _similarity(
        collection: zvec.Collection, vector: list[float], k: int, filter_expr: str | None
    ) -> list[tuple[Document, float]]:
        """近邻检索原语：返回 (分块, 距离)。"""
        found = collection.query(
            queries=zvec.Query(field_name=VECTOR_FIELD, vector=vector),
            topk=k,
            filter=filter_expr,
            output_fields=_OUTPUT_FIELDS,
        )
        return [(_to_document(doc), _distance(doc)) for doc in found]

    @staticmethod
    def _mmr(
        collection: zvec.Collection,
        vector: list[float],
        k: int,
        fetch_k: int,
        filter_expr: str | None,
    ) -> list[Document]:
        """MMR 召回：zvec 没有该能力，先多取候选再自己挑（适配器内的实现细节）。"""
        found = collection.query(
            queries=zvec.Query(field_name=VECTOR_FIELD, vector=vector),
            topk=fetch_k,
            filter=filter_expr,
            include_vector=True,
            output_fields=_OUTPUT_FIELDS,
        )
        candidates = [(_to_document(doc), _stored_vector(doc), _distance(doc)) for doc in found]
        return maximal_marginal_relevance(candidates, k)

    @staticmethod
    def _hybrid(
        collection: zvec.Collection, vector: list[float], k: int, filter_expr: str | None
    ) -> list[tuple[Document, float | None]]:
        """相似度检索 + MMR 去重补充，兼顾精确召回与结果多样性（与 Chroma 实现同策略）。"""
        raw: list[tuple[Document, float | None]] = [
            (doc, distance_to_relevance(distance))
            for doc, distance in ZvecVectorStore._similarity(collection, vector, k, filter_expr)
        ]
        if len(raw) >= k:
            return raw

        docs = ZvecVectorStore._mmr(collection, vector, max(1, k - len(raw)), k * 2, filter_expr)
        return merge_hybrid(raw, docs, k)

    # ── 删除与统计 ─────────────────────────────────────────────────────

    async def delete_document(self, kb_id: int, doc_id: int) -> int:
        collection = self._existing(kb_id)
        if collection is None:
            return 0

        removed = await asyncio.to_thread(_count_document_chunks, collection, doc_id)
        if removed == 0:
            return 0
        # zvec 的 delete_by_filter 不返回删除条数，所以先数后删
        await asyncio.to_thread(collection.delete_by_filter, f'doc_id = "{doc_id}"')
        logger.info(f"[VECTOR_STORE] 删除 {removed} 个向量 kb={kb_id} doc_id={doc_id}")
        return removed

    async def delete_kb(self, kb_id: int) -> None:
        collection = self._existing(kb_id)
        if collection is None:
            # 从未入库过的知识库没有 collection，属于正常情况
            logger.debug(f"[VECTOR_STORE] collection 不存在，无需删除 kb={kb_id}")
            return
        await asyncio.to_thread(self._destroy, kb_id, collection)
        logger.info(f"[VECTOR_STORE] 删除 collection kb={kb_id}")

    def _destroy(self, kb_id: int, collection: zvec.Collection) -> None:
        """destroy 会删掉磁盘目录且必须在句柄打开时调用（关闭后再调会报错）。"""
        collection.destroy()
        self._collections.pop(kb_id, None)

    def count(self, kb_id: int) -> int:
        collection = self._existing(kb_id)
        return 0 if collection is None else int(collection.stats.doc_count)

    def count_chunks_by_filename(self, kb_id: int) -> dict[str, int]:
        collection = self._existing(kb_id)
        if collection is None:
            return {}
        counts: dict[str, int] = {}
        with collection.iter_docs(output_fields=["filename"], include_vector=False) as docs:
            for doc in docs:
                filename = str(doc.fields.get("filename", "?"))
                counts[filename] = counts.get(filename, 0) + 1
        return counts

    def list_doc_ids(self, kb_id: int) -> list[str]:
        collection = self._existing(kb_id)
        if collection is None:
            return []
        with collection.iter_docs(output_fields=["doc_id"], include_vector=False) as docs:
            return [str(doc.fields.get("doc_id")) for doc in docs]


def _schema(kb_id: int, dimension: int) -> zvec.CollectionSchema:
    """知识库 collection 的 schema。

    ``doc_id`` 带倒排索引：按文档删除（``delete_by_filter``）与 doc 级过滤都走它。
    ``page`` / ``page_start`` / ``page_end`` 都是 nullable 的：只有 PDF 才有页码，
    缺省时读回结果里没有这些键，与 Chroma 实现的「没有该字段」保持一致。
    ``page_start`` / ``page_end`` 是闭区间（1-based 物理页）：当前切分不跨页，二者相等；
    schema 必须提前声明，因为已建的 collection 改不了 schema——补字段要重建索引。
    """
    return zvec.CollectionSchema(
        name=f"kb_{kb_id}",
        fields=[
            zvec.FieldSchema("doc_id", zvec.DataType.STRING, index_param=zvec.InvertIndexParam()),
            zvec.FieldSchema("kb_id", zvec.DataType.STRING),
            zvec.FieldSchema("filename", zvec.DataType.STRING),
            zvec.FieldSchema("chunk_index", zvec.DataType.STRING),
            zvec.FieldSchema("page", zvec.DataType.INT64, nullable=True),
            zvec.FieldSchema("page_start", zvec.DataType.INT64, nullable=True),
            zvec.FieldSchema("page_end", zvec.DataType.INT64, nullable=True),
            zvec.FieldSchema(CONTENT_FIELD, zvec.DataType.STRING),
        ],
        vectors=[
            zvec.VectorSchema(
                VECTOR_FIELD,
                zvec.DataType.VECTOR_FP32,
                dimension=dimension,
                index_param=zvec.HnswIndexParam(metric_type=zvec.MetricType.COSINE),
            )
        ],
    )


def _chunk_id(kb_id: int, doc_id: int, chunk_index: str) -> str:
    """分块主键：写入幂等的关键。

    用 kb / doc / 分块序号组装（而不是随机 id），重跑入库会覆盖同一分块而不是累积重复；
    zvec 的 id 不允许冒号，因此用短横线连接。
    """
    return f"{kb_id}-{doc_id}-{chunk_index}"


def _filter_expression(filter_doc_ids: list[int] | None) -> str | None:
    """doc_id 级过滤表达式（每个知识库一个 collection，不需要 kb 条件）。

    doc_id 是我们自己生成的 int、zvec 的过滤器也没有参数化接口，因此直接拼字符串；
    检索范围由调用方（ACL 校验过）决定。
    """
    if not filter_doc_ids:
        return None
    if len(filter_doc_ids) == 1:
        return f'doc_id = "{filter_doc_ids[0]}"'
    ids = ", ".join(f'"{doc_id}"' for doc_id in filter_doc_ids)
    return f"doc_id in ({ids})"


def _to_document(doc: zvec.Doc) -> Document:
    """zvec 命中结果 -> 业务层统一使用的 Document（正文出栈，其余作为元数据）。"""
    fields = dict(doc.fields)
    return Document(page_content=fields.pop(CONTENT_FIELD), metadata=fields)


def _distance(doc: zvec.Doc) -> float:
    """命中结果的余弦距离：COSINE 度量下 zvec 的 score 就是 ``1 - 余弦相似度``。"""
    return float(cast("float", doc.score))


def _stored_vector(doc: zvec.Doc) -> list[float]:
    """回读命中向量的稠密浮点数组（仅在 ``include_vector=True`` 的 MMR 路径使用）。"""
    return [float(value) for value in cast("list[float]", doc.vectors[VECTOR_FIELD])]


def _count_document_chunks(collection: zvec.Collection, doc_id: int) -> int:
    """数出某文档现有的分块数（流式扫描，读路径不用它，只服务删除 / 重建）。"""
    with collection.iter_docs(output_fields=["doc_id"], include_vector=False) as docs:
        return sum(1 for doc in docs if doc.fields.get("doc_id") == str(doc_id))


def _ensure_ok(statuses: zvec.Status | list[zvec.Status], action: str) -> None:
    """写操作只在返回值里报错（不抛异常），失败必须显性化，否则会静默丢数据。"""
    checked = statuses if isinstance(statuses, list) else [statuses]
    failed = [status for status in checked if not status.ok()]
    if failed:
        details = "; ".join(f"code={status.code()} {status.message()}" for status in failed)
        msg = f"zvec {action} 失败: {details}"
        raise RuntimeError(msg)
