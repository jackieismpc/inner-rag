"""精排插件点测试：契约（只改顺序）、内置实现的重排逻辑、以及接进检索层的时机。

这一层最容易出的错不是「排得不好」，而是**把结果弄丢**：精排只要增删一条、改一次分，
阈值语义与前端展示就会跟着漂。所以用例的重点是契约本身，其次才是排序质量。
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.documents import Document

from inner_rag.core.config import settings
from inner_rag.plugins.registry import rerankers
from inner_rag.services import rerank, retrieval
from inner_rag.services import vector_store as vector_store_package
from inner_rag.services.lexical import lexical_index
from inner_rag.services.rerank import (
    LexicalReranker,
    LLMReranker,
    NoopReranker,
    candidate_idf,
    parse_order,
)
from inner_rag.services.vector_store.memory_store import MemoryVectorStore

KB_ID = 7778


def _doc(content: str, chunk_index: int, doc_id: int = 1) -> Document:
    return Document(
        page_content=content,
        metadata={"doc_id": str(doc_id), "chunk_index": str(chunk_index), "page": 1},
    )


def _result(*contents: str) -> list[tuple[Document, float | None]]:
    return [(_doc(content, index), 1.0 - index * 0.1) for index, content in enumerate(contents)]


# ── 契约 ──────────────────────────────────────────────────────────────


async def test_noop_returns_the_very_same_list() -> None:
    results = _result("甲", "乙")

    assert await NoopReranker().rerank(KB_ID, "任意", results) is results


@pytest.mark.parametrize("reranker_impl", [NoopReranker(), LexicalReranker()])
async def test_rerank_preserves_length_and_scores(reranker_impl: Any) -> None:
    """契约：只改顺序。条数与每条的分值必须逐项保持。"""
    results = _result("诺诺的真名是陈墨瞳。", "完全无关的一句话。", "诺诺在雨里站着。")

    reordered = await reranker_impl.rerank(KB_ID, "诺诺的真名是什么？", results)

    assert len(reordered) == len(results)
    assert {id(doc) for doc, _ in reordered} == {id(doc) for doc, _ in results}
    assert sorted(score for _, score in reordered) == sorted(score for _, score in results)


# ── 候选集内 IDF ──────────────────────────────────────────────────────


def test_candidate_idf_rewards_terms_present_in_few_candidates() -> None:
    """本批里人人皆有的查询词没有区分度，只在少数候选里出现的才有。"""
    query = {"诺诺", "真名"}
    documents = [{"诺诺", "真名"}, {"诺诺"}, {"诺诺"}, {"诺诺"}]

    weights = candidate_idf(query, documents)

    assert weights["真名"] > weights["诺诺"]


def test_candidate_idf_is_small_when_every_candidate_has_the_term() -> None:
    """本批里人人皆有的词几乎没有区分度（权重趋近 0，而不是负数或无穷）。"""
    weights = candidate_idf({"诺诺"}, [{"诺诺"}, {"诺诺"}])

    assert 0.0 < weights["诺诺"] < 1.0


# ── lexical 精排 ──────────────────────────────────────────────────────


async def test_lexical_reranker_promotes_the_candidate_covering_all_terms() -> None:
    results = _result("诺诺在雨里站着。", "他啊了一声。", "诺诺的真名是陈墨瞳。")

    reordered = await LexicalReranker().rerank(KB_ID, "诺诺的真名是什么？", results)

    assert reordered[0][0].page_content == "诺诺的真名是陈墨瞳。"


async def test_lexical_reranker_is_stable_on_ties() -> None:
    """覆盖率相同时保持粗排顺序——精排不该把「本来就对」的题打乱。"""
    results = _result("诺诺来了。", "诺诺走了。")

    reordered = await LexicalReranker().rerank(KB_ID, "诺诺", results)

    assert [doc.page_content for doc, _ in reordered] == ["诺诺来了。", "诺诺走了。"]


@pytest.mark.parametrize(
    ("query", "contents"),
    [
        ("诺诺", ["只有一个候选"]),  # 单条
        ("!!!", ["甲", "乙"]),  # 查询切不出词元
    ],
)
async def test_lexical_reranker_passes_through_when_nothing_to_do(
    query: str, contents: list[str]
) -> None:
    results = _result(*contents)

    reordered = await LexicalReranker().rerank(KB_ID, query, results)

    assert [doc.page_content for doc, _ in reordered] == [doc.page_content for doc, _ in results]


# ── llm 精排：解析与兜底 ──────────────────────────────────────────────


def test_parse_order_accepts_a_plain_number_list() -> None:
    assert parse_order("2,1,3", 3) == [1, 0, 2]


def test_parse_order_drops_duplicates_and_out_of_range() -> None:
    """模型多给、重复、越界都是常态：必须去重并丢弃，不能让它污染结果。"""
    assert parse_order("[3] 1. 3, 9, 0, 2", 3) == [2, 0, 1]


def test_parse_order_returns_empty_when_there_is_no_number() -> None:
    assert parse_order("抱歉，我无法完成这个任务。", 3) == []


async def test_llm_reranker_reorders_and_keeps_every_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = _result("甲", "乙", "丙")

    async def fake_ask(_query: str, _candidates: Any) -> list[int]:
        return [2, 0]  # 漏掉了下标 1

    monkeypatch.setattr(LLMReranker, "_ask", staticmethod(fake_ask))

    reordered = await LLMReranker().rerank(KB_ID, "任意", results)

    assert [doc.page_content for doc, _ in reordered] == ["丙", "甲", "乙"]


async def test_llm_reranker_falls_back_to_original_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = _result("甲", "乙")

    async def boom(_query: str, _candidates: Any) -> list[int]:
        msg = "上游 500"
        raise RuntimeError(msg)

    monkeypatch.setattr(LLMReranker, "_ask", staticmethod(boom))

    assert await LLMReranker().rerank(KB_ID, "任意", results) is results


async def test_llm_reranker_falls_back_when_output_is_unparseable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = _result("甲", "乙")

    async def empty(_query: str, _candidates: Any) -> list[int]:
        return []

    monkeypatch.setattr(LLMReranker, "_ask", staticmethod(empty))

    assert await LLMReranker().rerank(KB_ID, "任意", results) is results


async def test_llm_reranker_keeps_the_tail_beyond_top_n(monkeypatch: pytest.MonkeyPatch) -> None:
    """``RERANK_LLM_TOP_N`` 之外的候选不送模型，但也绝不能丢。"""
    monkeypatch.setattr(settings, "RERANK_LLM_TOP_N", 2)
    results = _result("甲", "乙", "丙", "丁")

    async def fake_ask(_query: str, _candidates: Any) -> list[int]:
        return [1, 0]

    monkeypatch.setattr(LLMReranker, "_ask", staticmethod(fake_ask))

    reordered = await LLMReranker().rerank(KB_ID, "任意", results)

    assert [doc.page_content for doc, _ in reordered] == ["乙", "甲", "丙", "丁"]


# ── 注册表 ────────────────────────────────────────────────────────────


def test_registry_exposes_every_builtin() -> None:
    assert set(rerankers.names()) == {"none", "lexical", "llm"}
    assert rerankers.settings_key == "RERANK_BACKEND"


def test_build_reranker_rejects_unknown_name_with_available_list() -> None:
    with pytest.raises(ValueError, match="不支持的 RERANK_BACKEND"):
        rerank.build_reranker("cross-encoder-from-nowhere")


def test_default_backend_is_disabled() -> None:
    assert settings.RERANK_BACKEND == "none"
    assert rerank.is_enabled() is False


# ── 接进检索层的时机 ──────────────────────────────────────────────────


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> MemoryVectorStore:
    backend = MemoryVectorStore()
    monkeypatch.setattr(vector_store_package, "vector_service", backend)
    monkeypatch.setattr(retrieval, "vector_service", backend)
    lexical_index.invalidate()
    yield backend
    lexical_index.invalidate()


class _SpyReranker:
    """假精排器：整体反转顺序，用于确认「精排在阈值过滤之后」「条数与分数不变」。"""

    name = "spy"

    def __init__(self) -> None:
        self.calls: list[tuple[int, str, int]] = []
        self.inputs: list[list[str]] = []
        self.input_scores: list[list[float | None]] = []

    async def rerank(
        self, kb_id: int, query: str, results: list[tuple[Document, float | None]]
    ) -> list[tuple[Document, float | None]]:
        self.calls.append((kb_id, query, len(results)))
        self.inputs.append([doc.page_content for doc, _ in results])
        self.input_scores.append([score for _, score in results])
        return list(reversed(results))


async def _ingest(backend: MemoryVectorStore, texts: list[str]) -> None:
    documents = [
        Document(page_content=text, metadata={"page": index + 1})
        for index, text in enumerate(texts)
    ]
    await backend.add_documents(kb_id=KB_ID, documents=documents, doc_id=1, filename="unit.txt")


async def test_rerank_not_called_when_disabled(
    store: MemoryVectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _ingest(store, ["诺诺的真名是陈墨瞳。"])
    spy = _SpyReranker()
    monkeypatch.setattr(rerank, "reranker", spy)

    await retrieval.search(KB_ID, "诺诺", k=2, strategy="similarity", score_threshold=0.0)

    assert spy.calls == []


async def test_rerank_runs_after_threshold_filtering(
    store: MemoryVectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """精排看到的必须是**已经过阈值**的候选集，否则它会把被滤掉的分块又摆回来。"""
    await _ingest(store, ["诺诺的真名是陈墨瞳。", "完全无关的一句话。"])
    spy = _SpyReranker()
    monkeypatch.setattr(rerank, "reranker", spy)

    results, filtered_out = await retrieval.search(
        KB_ID, "诺诺的真名是陈墨瞳", k=8, strategy="similarity", score_threshold=0.99
    )

    # 阈值 0.99 把两条都挡掉 → 精排没有候选可排，因此不该被调用
    assert results == []
    assert filtered_out == 2
    assert spy.calls == []


async def test_rerank_reorders_what_enters_context(
    store: MemoryVectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "HYBRID_SPARSE_WEIGHT", 0.0)
    await _ingest(store, ["诺诺的真名是陈墨瞳。", "诺诺在雨里站着。", "他啊了一声。"])
    spy = _SpyReranker()
    monkeypatch.setattr(rerank, "reranker", spy)

    results, _filtered_out = await retrieval.search(
        KB_ID, "诺诺", k=3, strategy="similarity", score_threshold=0.0
    )

    assert spy.calls == [(KB_ID, "诺诺", 3)]
    # 精排拿到的就是粗排顺序，输出是它的反转 —— 顺序确实被改了
    assert [doc.page_content for doc, _ in results] == list(reversed(spy.inputs[0]))
    # 分数逐项不变：精排只改顺序、不改分，阈值语义因此不受它影响
    assert [score for _, score in results] == list(reversed(spy.input_scores[0]))
    assert len(results) == 3


async def test_rerank_tolerates_a_misbehaving_implementation(
    store: MemoryVectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """实现违约（增删条目）时必须退回粗排顺序，而不是把结果弄丢。"""
    await _ingest(store, ["诺诺的真名是陈墨瞳。", "诺诺在雨里站着。"])

    class _DroppingReranker:
        name = "dropping"

        async def rerank(self, kb_id: int, query: str, results: Any) -> Any:
            return list(results)[:1]

    monkeypatch.setattr(rerank, "reranker", _DroppingReranker())

    results, _filtered_out = await retrieval.search(
        KB_ID, "诺诺", k=8, strategy="similarity", score_threshold=0.0
    )

    assert len(results) == 2
