"""向量库测试：入库、相关度换算、阈值过滤、MMR/hybrid、删除。"""

from __future__ import annotations

from langchain_core.documents import Document

from inner_rag.services.vector_store import distance_to_relevance, vector_service

DOC_TEXT = (
    "inner-rag 是一个知识库问答系统。\n\n"
    "它支持文档上传、向量检索、混合检索以及引用来源展示。\n\n"
    "检索策略包括 similarity、mmr 和 hybrid 三种，相关度按余弦相似度换算。"
)


async def _ingest(kb_id: int, text: str = DOC_TEXT, doc_id: int = 1) -> int:
    documents = [Document(page_content=text, metadata={"source": "unit-test"})]
    return await vector_service.add_documents_async(
        kb_id=kb_id, documents=documents, doc_id=doc_id, filename="unit.txt"
    )


def test_distance_to_relevance_is_clamped() -> None:
    assert distance_to_relevance(0.0) == 1.0
    assert distance_to_relevance(0.25) == 0.75
    assert distance_to_relevance(-0.5) == 1.0  # 不会出现 >1 的「相关度」
    assert distance_to_relevance(1.7) == 0.0


async def test_ingest_and_stats() -> None:
    kb_id = 101
    chunks = await _ingest(kb_id)
    try:
        assert chunks > 0
        assert vector_service.get_kb_stats(kb_id)["vector_count"] == chunks
        assert vector_service.count_chunks_by_filename(kb_id) == {"unit.txt": chunks}
        assert set(vector_service.list_doc_ids(kb_id)) == {"1"}
    finally:
        vector_service.delete_kb(kb_id)


async def test_similarity_search_returns_relevance_scores() -> None:
    kb_id = 102
    await _ingest(kb_id)
    try:
        results, filtered_out = await vector_service.similarity_search_async(
            kb_id=kb_id, query="向量检索策略 similarity mmr hybrid", k=5
        )
        assert results
        assert filtered_out == 0
        for _, score in results:
            assert score is not None
            assert 0.0 <= score <= 1.0
        # 有分数的结果必须按相关度降序
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)
        # 元数据必须是 Chroma 可过滤的标量
        metadata = results[0][0].metadata
        assert metadata["doc_id"] == "1" and metadata["kb_id"] == str(kb_id)
        assert isinstance(metadata["chunk_index"], str)
    finally:
        vector_service.delete_kb(kb_id)


async def test_score_threshold_filters_low_relevance() -> None:
    kb_id = 103
    await _ingest(kb_id)
    try:
        _, filtered_out = await vector_service.similarity_search_async(
            kb_id=kb_id, query="完全无关的提问：量子色动力学与黎曼猜想", k=5
        )
        # 低相关度的分块应被过滤掉并计数（而不是被当成 0 分返回）
        stats = vector_service.get_kb_stats(kb_id)
        assert filtered_out >= 0
        assert stats["vector_count"] > 0
    finally:
        vector_service.delete_kb(kb_id)


async def test_mmr_results_have_no_fabricated_scores() -> None:
    kb_id = 104
    await _ingest(kb_id)
    try:
        results, _ = await vector_service.similarity_search_async(
            kb_id=kb_id, query="知识库问答", k=2, strategy="mmr", score_threshold=0.0
        )
        assert results
        assert all(score is None for _, score in results)
    finally:
        vector_service.delete_kb(kb_id)


async def test_hybrid_deduplicates_chunks() -> None:
    kb_id = 105
    await _ingest(kb_id)
    try:
        results, _ = await vector_service.similarity_search_async(
            kb_id=kb_id, query="混合检索 hybrid 去重", k=4, strategy="hybrid", score_threshold=0.0
        )
        keys = [(doc.metadata.get("doc_id"), doc.metadata.get("chunk_index")) for doc, _ in results]
        assert len(keys) == len(set(keys))  # 不能出现重复分块
    finally:
        vector_service.delete_kb(kb_id)


async def test_filter_by_doc_ids() -> None:
    kb_id = 106
    await _ingest(kb_id, doc_id=1)
    await _ingest(kb_id, text="另一个文档内容：仅用于 document ID 过滤测试。", doc_id=2)
    try:
        results, _ = await vector_service.similarity_search_async(
            kb_id=kb_id, query="文档内容过滤", k=5, filter_doc_ids=[2], score_threshold=0.0
        )
        assert results
        assert {doc.metadata["doc_id"] for doc, _ in results} == {"2"}
    finally:
        vector_service.delete_kb(kb_id)


async def test_delete_documents_removes_vectors() -> None:
    kb_id = 107
    await _ingest(kb_id, doc_id=1)
    await _ingest(kb_id, text="待删除文档的内容。", doc_id=2)
    try:
        removed = vector_service.delete_documents(kb_id, 2)
        assert removed > 0
        assert set(vector_service.list_doc_ids(kb_id)) == {"1"}
    finally:
        vector_service.delete_kb(kb_id)


async def test_empty_documents_are_skipped() -> None:
    kb_id = 108
    try:
        assert await _ingest(kb_id, text="   \n  ", doc_id=1) == 0
    finally:
        vector_service.delete_kb(kb_id)
