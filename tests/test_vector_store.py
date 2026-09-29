"""向量库契约测试：同一套用例跑两套后端（zvec / chroma）。

断言的是 `docs/architecture.md` 3.2 的契约（相关度口径、阈值计数、MMR 无分数、标量元数据、
删除后的可见性），不是某个后端的实现细节——两套实现的分块 id、排序细节允许不同。
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from inner_rag.core.config import settings
from inner_rag.services.vector_store import VectorStore, build_vector_store, distance_to_relevance

KB_ID = 101

DOC_TEXT = (
    "inner-rag 是一个知识库问答系统。\n\n"
    "它支持文档上传、向量检索、混合检索以及引用来源展示。\n\n"
    "检索策略包括 similarity、mmr 和 hybrid 三种，相关度按余弦相似度换算。"
)


@pytest.fixture
def vector_paths(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """每个用例一套独立目录：契约测试不能被上一个用例的向量数据影响。"""
    monkeypatch.setattr(settings, "ZVEC_PATH", str(tmp_path / "zvec"))
    monkeypatch.setattr(settings, "CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))


@pytest.fixture(params=["zvec", "chroma"])
def store(vector_paths: None, request: pytest.FixtureRequest) -> VectorStore:
    """参数化后端：两套实现必须通过同一份契约。"""
    return build_vector_store(request.param)


@pytest.fixture
def zvec_store(vector_paths: None) -> VectorStore:
    """只跑 zvec 的用例（幂等写入是它的实现特性）。"""
    return build_vector_store("zvec")


async def _ingest(store: VectorStore, text: str = DOC_TEXT, doc_id: int = 1) -> int:
    documents = [Document(page_content=text, metadata={"source": "unit-test", "page": 1})]
    return await store.add_documents(
        kb_id=KB_ID, documents=documents, doc_id=doc_id, filename="unit.txt"
    )


def test_distance_to_relevance_is_clamped() -> None:
    assert distance_to_relevance(0.0) == 1.0
    assert distance_to_relevance(0.25) == 0.75
    assert distance_to_relevance(-0.5) == 1.0  # 不会出现 >1 的「相关度」
    assert distance_to_relevance(1.7) == 0.0


async def test_ingest_and_stats(store: VectorStore) -> None:
    chunks = await _ingest(store)
    try:
        assert chunks > 0
        assert store.count(KB_ID) == chunks
        assert store.count_chunks_by_filename(KB_ID) == {"unit.txt": chunks}
        assert set(store.list_doc_ids(KB_ID)) == {"1"}
    finally:
        await store.delete_kb(KB_ID)


async def test_similarity_search_returns_relevance_scores(store: VectorStore) -> None:
    await _ingest(store)
    try:
        results, filtered_out = await store.search(
            kb_id=KB_ID, query="向量检索策略 similarity mmr hybrid", k=5
        )
        assert results
        assert filtered_out == 0
        for _, score in results:
            assert score is not None
            assert 0.0 <= score <= 1.0
        # 有分数的结果必须按相关度降序
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)
        # 元数据必须是可过滤的标量，且正文来自原文
        doc, _ = results[0]
        assert doc.page_content.strip()
        assert doc.metadata["doc_id"] == "1" and doc.metadata["kb_id"] == str(KB_ID)
        assert isinstance(doc.metadata["chunk_index"], str)
        assert doc.metadata["filename"] == "unit.txt"
        assert doc.metadata["page"] == 1
    finally:
        await store.delete_kb(KB_ID)


async def test_score_threshold_filters_low_relevance(store: VectorStore) -> None:
    await _ingest(store)
    try:
        results, filtered_out = await store.search(
            kb_id=KB_ID, query="完全无关的提问：量子色动力学与黎曼猜想", k=5, score_threshold=0.99
        )
        # 低相关度的分块应被过滤掉并计数（而不是被当成 0 分返回）
        assert filtered_out > 0
        assert all(score >= 0.99 for _, score in results)
        assert filtered_out + len(results) == store.count(KB_ID)
    finally:
        await store.delete_kb(KB_ID)


async def test_mmr_results_have_no_fabricated_scores(store: VectorStore) -> None:
    await _ingest(store)
    try:
        results, _ = await store.search(
            kb_id=KB_ID, query="知识库问答", k=2, strategy="mmr", score_threshold=0.0
        )
        assert results
        assert all(score is None for _, score in results)
    finally:
        await store.delete_kb(KB_ID)


async def test_hybrid_deduplicates_chunks(store: VectorStore) -> None:
    await _ingest(store)
    try:
        results, _ = await store.search(
            kb_id=KB_ID, query="混合检索 hybrid 去重", k=4, strategy="hybrid", score_threshold=0.0
        )
        keys = [(doc.metadata.get("doc_id"), doc.metadata.get("chunk_index")) for doc, _ in results]
        assert len(keys) == len(set(keys))  # 不能出现重复分块
    finally:
        await store.delete_kb(KB_ID)


async def test_filter_by_doc_ids(store: VectorStore) -> None:
    await _ingest(store, doc_id=1)
    await _ingest(store, text="另一个文档内容：仅用于 document ID 过滤测试。", doc_id=2)
    try:
        results, _ = await store.search(
            kb_id=KB_ID, query="文档内容过滤", k=5, filter_doc_ids=[2], score_threshold=0.0
        )
        assert results
        assert {doc.metadata["doc_id"] for doc, _ in results} == {"2"}
    finally:
        await store.delete_kb(KB_ID)


async def test_delete_document_removes_vectors(store: VectorStore) -> None:
    first_chunks = await _ingest(store, doc_id=1)
    second_chunks = await _ingest(store, text="待删除文档的内容。", doc_id=2)
    try:
        removed = await store.delete_document(KB_ID, 2)
        assert removed == second_chunks
        assert set(store.list_doc_ids(KB_ID)) == {"1"}
        assert store.count(KB_ID) == first_chunks
    finally:
        await store.delete_kb(KB_ID)


async def test_delete_kb_drops_all_vectors(store: VectorStore) -> None:
    await _ingest(store)
    await store.delete_kb(KB_ID)
    assert store.count(KB_ID) == 0
    assert await store.search(kb_id=KB_ID, query="任意查询") == ([], 0)


async def test_empty_documents_are_skipped(store: VectorStore) -> None:
    try:
        assert await _ingest(store, text="   \n  ") == 0
        assert store.count(KB_ID) == 0
    finally:
        await store.delete_kb(KB_ID)


async def test_zvec_reingest_does_not_accumulate(zvec_store: VectorStore) -> None:
    """同一文档重复入库必须覆盖同一批分块（zvec 侧 id 可重复 → upsert 幂等）。

    Chroma 侧用随机 id，重复入库会累积，所以那条路径靠「重新入库前先 delete_document」保证；
    zvec 不需要依赖调用方的纪律，但两条路径都要让 count 与数据库的分块数对得上。
    """
    chunks = await _ingest(zvec_store)
    try:
        assert await _ingest(zvec_store) == chunks
        assert zvec_store.count(KB_ID) == chunks
        assert zvec_store.count_chunks_by_filename(KB_ID) == {"unit.txt": chunks}
    finally:
        await zvec_store.delete_kb(KB_ID)
