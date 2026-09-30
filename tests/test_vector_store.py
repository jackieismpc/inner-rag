"""向量库契约测试：同一套用例跑三套后端（zvec / chroma / memory）。

断言的是 `docs/architecture.md` 3.2 的契约（相关度口径、阈值计数、MMR 无分数、标量元数据、
删除后的可见性、**写入幂等与断点续跑**），不是某个后端的实现细节——各实现的分块 id、
排序细节允许不同。

`memory` 是 Phase 7 的替换演练产物（`memory_store.py`）：它只加了一个实现类 + 一次
`vector_stores.register`，业务代码零改动，却直接通过了下面全部契约用例。

「断点续跑」为什么按契约测而不是按实现测：入库是长任务（全库 1.1 万分块），
任何一次 429 / 超时 / 进程被杀都会中断它，所以「重跑等于续跑」是**写入路径的基本要求**，
不是某个后端的优化。用例通过在第 N 批嵌入上注入异常来制造真实中断，
然后断言「不重算、不遗漏、不重复」。
"""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.documents import Document

from inner_rag.core.config import settings
from inner_rag.services.embedding import embedding_service
from inner_rag.services.vector_store import VectorStore, build_vector_store, distance_to_relevance
from inner_rag.services.vector_store.base import (
    embed_batches_in_order,
    normalized_chunk_key,
    prepare_chunks,
)

KB_ID = 101

DOC_TEXT = (
    "inner-rag 是一个知识库问答系统。\n\n"
    "它支持文档上传、向量检索、混合检索以及引用来源展示。\n\n"
    "检索策略包括 similarity、mmr 和 hybrid 三种，相关度按余弦相似度换算。"
)


def _paged_documents(pages: int = 50) -> list[Document]:
    """造多页文档：切分器**不跨 Document**，所以每页各自成块，页数 ≈ 分块数。

    默认 50 页 > ``EMBED_BATCH_SIZE``(20) 的 2 倍，保证嵌入至少分 3 批，
    这样「在第 2 批上注入异常」才有意义（只分 1 批的话中断等价于全成功）。
    """
    return [
        Document(
            page_content=f"第 {index} 页评测语料。" + f"这一段属于第 {index} 页的内容。" * 12,
            metadata={"source": "unit-test", "page": index},
        )
        for index in range(1, pages + 1)
    ]


@pytest.fixture
def vector_paths(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """每个用例一套独立目录：契约测试不能被上一个用例的向量数据影响。"""
    monkeypatch.setattr(settings, "ZVEC_PATH", str(tmp_path / "zvec"))
    monkeypatch.setattr(settings, "CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))


@pytest.fixture(params=["zvec", "chroma", "memory"])
def store(vector_paths: None, request: pytest.FixtureRequest) -> VectorStore:
    """参数化后端：三套实现必须通过同一份契约。"""
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
    """同一文档重复入库既不累积也不重算（zvec 侧 id 可重复 → upsert 幂等）。

    Chroma 侧用随机 id，重复入库会累积，所以那条路径靠「重新入库前先 delete_document」保证；
    zvec 不需要依赖调用方的纪律，但两条路径都要让 count 与数据库的分块数对得上。
    返回值语义是「**本次新写入**的分块数」——第二次入库全部命中已有分块，因此必须是 0。
    """
    chunks = await _ingest(zvec_store)
    try:
        assert await _ingest(zvec_store) == 0
        assert zvec_store.count(KB_ID) == chunks
        assert zvec_store.count_chunks_by_filename(KB_ID) == {"unit.txt": chunks}
    finally:
        await zvec_store.delete_kb(KB_ID)


def test_unknown_backend_error_lists_available_ones() -> None:
    """配置写错时要能照着实提示改，而不是只说「不支持」。"""
    with pytest.raises(ValueError) as excinfo:
        build_vector_store("no-such-store")
    message = str(excinfo.value)
    assert "no-such-store" in message
    for name in ("zvec", "chroma", "memory"):
        assert name in message


async def test_memory_reingest_does_not_accumulate() -> None:
    """memory 用确定性分块 key（doc_id + chunk_index），重跑入库跳过已有分块而不是累积。"""
    store = build_vector_store("memory")
    chunks = await _ingest(store)
    try:
        assert await _ingest(store) == 0
        assert store.count(KB_ID) == chunks
    finally:
        await store.delete_kb(KB_ID)


# ── 断点续跑（入库路径的基本要求，三后端共用一份契约）────────────────────


def test_normalized_chunk_key_stringifies_chunk_index() -> None:
    """进度比较的判据必须跨后端一致：chunk_index 在元数据里可能是 int 也可能是 str。

    zvec 的 schema 把它声明成 STRING，内存后端存的是切片下标（int）。
    不统一字符串化的话，「已入库」会被判成「没有」，续跑时白跑一整份文档。
    """
    int_keyed = Document(page_content="x", metadata={"doc_id": 7, "chunk_index": 3})
    str_keyed = Document(page_content="x", metadata={"doc_id": "7", "chunk_index": "3"})
    assert normalized_chunk_key(int_keyed) == ("7", "3")
    assert normalized_chunk_key(str_keyed) == ("7", "3")


async def test_stored_chunk_keys_reports_only_that_document(store: VectorStore) -> None:
    """进度查询要按文档隔离：不然续跑 A 会以为 B 的分块已经写过。"""
    await _ingest(store, doc_id=1)
    first = await _ingest(store, text="第二个文档的正文。", doc_id=2)
    try:
        keys = store.stored_chunk_keys(KB_ID, 1)
        assert keys == {("1", str(index)) for index in range(len(keys))}
        assert store.stored_chunk_keys(KB_ID, 2) == {("2", str(index)) for index in range(first)}
        # 没入库过的文档没有进度，而不是抛异常（续跑会从「一份都没写」开始）
        assert store.stored_chunk_keys(KB_ID, 999) == set()
    finally:
        await store.delete_kb(KB_ID)


async def test_stored_chunk_keys_is_empty_for_unknown_kb(store: VectorStore) -> None:
    """知识库从未成功入库时没有 collection，进度查询返回空集而不是报错。"""
    assert store.stored_chunk_keys(KB_ID, 1) == set()


@pytest.mark.parametrize("fail_on_call", [2, 3])
async def test_resume_after_interruption_fills_the_gap(
    store: VectorStore, monkeypatch: pytest.MonkeyPatch, fail_on_call: int
) -> None:
    """嵌入中途失败 → 重跑必须**不重算、不遗漏、不重复**。

    制造的中断是真实的：让第 ``fail_on_call`` 次批量嵌入抛异常，此时前面的批次已经
    flush 到库里（逐批落库），后面的批次被取消。重跑时按 (doc_id, chunk_index) 跳过
    已入库的部分，只补缺口。
    """
    documents = _paged_documents()
    total = len(prepare_chunks(documents, KB_ID, 1, "unit.txt"))
    # 前置条件：至少要有 fail_on_call 批，否则「在第 N 批失败」根本没有第 N 批
    assert total > settings.EMBED_BATCH_SIZE * (fail_on_call - 1)

    original = embedding_service.aembed_documents
    calls = {"count": 0}

    async def flaky(texts: list[str]) -> list[list[float]]:
        calls["count"] += 1
        if calls["count"] == fail_on_call:
            raise RuntimeError("模拟嵌入中断（429 / 超时 / 进程被杀）")
        return await original(texts)

    monkeypatch.setattr(embedding_service, "aembed_documents", flaky)
    with pytest.raises(RuntimeError, match="模拟嵌入中断"):
        await store.add_documents(kb_id=KB_ID, documents=documents, doc_id=1, filename="unit.txt")

    partial = store.count(KB_ID)
    try:
        # 中断前的批次必须已经落库，否则「续跑」就没有起点
        assert 0 < partial < total

        monkeypatch.setattr(embedding_service, "aembed_documents", original)
        written = await store.add_documents(
            kb_id=KB_ID, documents=documents, doc_id=1, filename="unit.txt"
        )
        assert written == total - partial  # 只补缺口，没重算
        assert store.count(KB_ID) == total  # 没遗漏
        assert store.stored_chunk_keys(KB_ID, 1) == {
            ("1", str(index)) for index in range(total)
        }  # 没重复
        assert store.count_chunks_by_filename(KB_ID) == {"unit.txt": total}

        # 全部入库后再跑一次：应当「无事可做」，返回 0 且总数不变
        assert (
            await store.add_documents(
                kb_id=KB_ID, documents=documents, doc_id=1, filename="unit.txt"
            )
            == 0
        )
        assert store.count(KB_ID) == total
    finally:
        await store.delete_kb(KB_ID)


async def test_interruption_cancels_remaining_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    """失败必须「确定且可重试」：不能让未完成的批次继续在后台烧配额。

    否则调用方以为已经失败并重试，实际两轮请求叠在一起——既打满额度，
    又让「库里到底写了多少」无法解释。

    这里直接在 ``embed_batches_in_order`` 层面测（取消逻辑归它管，与后端无关）：
    第 0 批「慢但会成功」、第 1 批立刻失败、第 2 批慢到「没被取消就一定会跑完」。
    时差是 20ms 对 5s，所以「第 2 批是否跑完」是确定性的，不看调度运气。
    """
    documents = [Document(page_content=f"c{index}", metadata={}) for index in range(6)]
    monkeypatch.setattr(settings, "EMBED_BATCH_SIZE", 2)
    started: list[str] = []
    survivors: list[str] = []

    async def scripted(texts: list[str]) -> list[list[float]]:
        head = texts[0]
        started.append(head)
        if head == "c0":
            await asyncio.sleep(0.02)  # 让「按序 yield」有机会先交出第 0 批
            return [[0.0]] * len(texts)
        if head == "c2":
            raise RuntimeError("第 1 批失败")
        await asyncio.sleep(5)  # 第 2 批：没被取消就会在这里跑完
        survivors.append(head)
        return [[0.0]] * len(texts)

    monkeypatch.setattr(embedding_service, "aembed_documents", scripted)

    delivered: list[str] = []
    with pytest.raises(RuntimeError, match="第 1 批失败"):
        async for batch, _ in embed_batches_in_order(documents):
            delivered.append(batch[0].page_content)

    assert delivered == ["c0"]  # 失败前已交出的批次确实落了库
    assert "c4" in started  # 第 2 批真的起来了（否则这条断言证明不了取消生效）
    assert survivors == []  # 它被取消了，没跑到底


async def test_memory_backend_instances_are_isolated() -> None:
    """内存后端的边界：数据挂在实例上，两个实例互不可见（写在这里以免被误当成 bug）。"""
    first = build_vector_store("memory")
    second = build_vector_store("memory")
    await _ingest(first)
    try:
        assert first.count(KB_ID) > 0
        assert second.count(KB_ID) == 0
        assert await second.search(kb_id=KB_ID, query="任意查询") == ([], 0)
    finally:
        await first.delete_kb(KB_ID)
