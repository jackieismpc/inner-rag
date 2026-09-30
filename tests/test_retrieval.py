"""检索组合层测试：融合规则、阈值语义、以及「哪几种策略会调用词面检索」。

这一层是 Phase 8 的核心，它要同时满足两个容易互相破坏的性质：

1. **融合的分数要继续可比较**：词面分先归一到 ``[0, 1]``，再乘 ``HYBRID_SPARSE_WEIGHT``，
   因此 ``RETRIEVAL_SCORE_THRESHOLD`` 对两路召回是同一把尺子；
2. **阈值必须在融合之后生效**：先过滤再融合，会把「向量分低、词面完全匹配」的候选提前丢掉——
   而补上这一路正是融合的目的（Phase 8.1 实测失败的两道题都是这个形状）。

``fuse`` 是纯函数，用单测钉死规则；``search`` 用假后端 + 假索引验证「路径选择」与「阈值时机」，
不依赖任何真实向量库或网络。
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.documents import Document

from inner_rag.core.config import settings
from inner_rag.services import retrieval
from inner_rag.services import vector_store as vector_store_package
from inner_rag.services.lexical import lexical_index
from inner_rag.services.retrieval import fuse
from inner_rag.services.vector_store import base as vector_base
from inner_rag.services.vector_store.memory_store import MemoryVectorStore

KB_ID = 7777


def _doc(content: str, chunk_index: int, doc_id: int = 1) -> Document:
    return Document(
        page_content=content,
        metadata={"doc_id": str(doc_id), "chunk_index": str(chunk_index), "page": 1},
    )


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> MemoryVectorStore:
    """干净的 memory 后端，并让检索层与词面索引都指向它。

    索引层是在调用时 import ``vector_service`` 的（避免循环 import），检索层则在 import
    时就绑定了名字——两处都要换，漏一处会退化成「用真后端检索、用假后端建索引」。
    """
    backend = MemoryVectorStore()
    monkeypatch.setattr(vector_store_package, "vector_service", backend)
    monkeypatch.setattr(retrieval, "vector_service", backend)
    lexical_index.invalidate()
    yield backend
    lexical_index.invalidate()


async def _ingest(backend: MemoryVectorStore, texts: list[str]) -> None:
    documents = [
        Document(page_content=text, metadata={"page": index + 1})
        for index, text in enumerate(texts)
    ]
    await backend.add_documents(kb_id=KB_ID, documents=documents, doc_id=1, filename="unit.txt")


def _spy_lexical(monkeypatch: pytest.MonkeyPatch, hits: list[tuple[Document, float]]) -> list[Any]:
    """替换词面检索，记录调用参数并返回固定命中（原始 BM25 分）。"""
    calls: list[Any] = []

    def fake_search(
        kb_id: int, query: str, k: int, filter_doc_ids: list[int] | None = None
    ) -> list[tuple[Document, float]]:
        calls.append((kb_id, query, k, filter_doc_ids))
        return list(hits)

    monkeypatch.setattr(lexical_index, "search", fake_search)
    return calls


# ── 融合规则（纯函数） ────────────────────────────────────────────────


def test_fuse_dedupes_by_chunk_and_keeps_higher_score() -> None:
    shared = _doc("同一分块", 0)

    fused = fuse([(shared, 0.2)], [(shared, 0.5)], k=5)

    assert len(fused) == 1
    assert fused[0][1] == 0.5


def test_fuse_keeps_dense_score_when_it_is_higher() -> None:
    shared = _doc("同一分块", 0)

    fused = fuse([(shared, 0.9)], [(shared, 0.3)], k=5)

    assert len(fused) == 1
    assert fused[0][1] == 0.9


def test_fuse_ignores_index_difference_between_doc_ids() -> None:
    """去重键是 (doc_id, chunk_index)：不同文档的相同 chunk_index 不能互相覆盖。"""
    first = _doc("甲", 0, doc_id=1)
    second = _doc("乙", 0, doc_id=2)

    fused = fuse([(first, 0.3)], [(second, 0.4)], k=5)

    assert len(fused) == 2


def test_fuse_orders_by_score_descending_with_none_last() -> None:
    scored_low = _doc("低", 0)
    scored_high = _doc("高", 1)
    no_score = _doc("MMR 补充项", 2)
    sparse = _doc("词面", 3)

    fused = fuse([(scored_low, 0.2), (no_score, None), (scored_high, 0.8)], [(sparse, 0.5)], k=5)

    assert [doc.page_content for doc, _ in fused] == ["高", "词面", "低", "MMR 补充项"]
    assert fused[-1][1] is None


def test_fuse_prefers_scored_entry_over_mmr_duplicate_on_dense_side() -> None:
    """向量侧同一个分块既在相似度结果里又在 MMR 补充里：保留有分数的那条。"""
    chunk = _doc("分块", 0)

    fused = fuse([(chunk, None), (chunk, 0.42)], [], k=5)

    assert len(fused) == 1
    assert fused[0][1] == 0.42


def test_fuse_gives_none_score_a_sparse_value_when_merged() -> None:
    """MMR 补充项若同时被词面命中，应升级为有分数的条目（否则会被阈值漏判）。"""
    chunk = _doc("分块", 0)

    fused = fuse([(chunk, None)], [(chunk, 0.6)], k=5)

    assert fused[0][1] == 0.6


def test_fuse_truncates_to_k() -> None:
    dense = [(_doc(f"d{index}", index), 0.9 - index * 0.1) for index in range(4)]

    fused = fuse(dense, [], k=2)

    assert len(fused) == 2


def test_fuse_passthrough_without_sparse_hits() -> None:
    dense = [(_doc("a", 0), 0.1), (_doc("b", 1), 0.7)]

    assert [doc.page_content for doc, _ in fuse(dense, [], k=5)] == ["b", "a"]


# ── 阈值在融合之后生效 ────────────────────────────────────────────────


def test_lexical_only_hit_can_clear_the_threshold() -> None:
    """Phase 8.1 的核心断言：向量分低于阈值、词面完全匹配的分块，必须能被救回来。"""
    dense_hit = _doc("向量捞到的弱相关页", 0)
    lexical_hit = _doc("诺诺的真名是陈墨瞳。", 1)

    fused = fuse([(dense_hit, 0.16)], [(lexical_hit, 0.5)], k=2)
    results, filtered_out = vector_base.finalize_results(fused, 0.3)

    assert [doc.page_content for doc, _ in results] == ["诺诺的真名是陈墨瞳。"]
    assert filtered_out == 1


def test_threshold_filters_weak_lexical_hits_too() -> None:
    """归一化让词面最高分恒为 1.0 × 权重；其余命中按比例衰减，同样受阈值约束。"""
    best = _doc("最匹配", 0)
    weak = _doc("沾了点边", 1)

    normalized = [(best, 1.0 * 0.5), (weak, 0.25 * 0.5)]
    results, filtered_out = vector_base.finalize_results(normalized, 0.3)

    assert [doc.page_content for doc, _ in results] == ["最匹配"]
    assert filtered_out == 1


# ── 策略选择：谁会用词面检索 ──────────────────────────────────────────


async def test_similarity_never_calls_lexical(
    store: MemoryVectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _ingest(store, ["诺诺的真名是陈墨瞳。", "完全无关的一句话。"])

    def boom(*_args: Any, **_kwargs: Any) -> list[tuple[Document, float]]:
        msg = "similarity 是纯向量策略，不该调用词面检索"
        raise AssertionError(msg)

    monkeypatch.setattr(lexical_index, "search", boom)

    results, filtered_out = await retrieval.search(
        KB_ID, "诺诺的真名是什么", k=2, strategy="similarity", score_threshold=0.0
    )

    assert results  # 纯向量不能因为词面缺席就查不到
    assert filtered_out == 0


async def test_hybrid_weight_zero_falls_back_to_dense_only(
    store: MemoryVectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _ingest(store, ["诺诺的真名是陈墨瞳。", "完全无关的一句话。"])
    monkeypatch.setattr(settings, "HYBRID_SPARSE_WEIGHT", 0.0)
    calls = _spy_lexical(monkeypatch, [])

    await retrieval.search(KB_ID, "诺诺", k=2, strategy="hybrid", score_threshold=0.0)

    assert calls == []  # 权重为 0 = 显式关闭词面一路


async def test_hybrid_consults_lexical_with_query_and_k(
    store: MemoryVectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _ingest(store, ["诺诺的真名是陈墨瞳。", "完全无关的一句话。"])
    monkeypatch.setattr(settings, "HYBRID_SPARSE_WEIGHT", 0.5)
    calls = _spy_lexical(monkeypatch, [])

    await retrieval.search(KB_ID, "诺诺的真名", k=3, strategy="hybrid", score_threshold=0.0)

    assert calls == [(KB_ID, "诺诺的真名", 3, None)]


async def test_dense_side_always_bypasses_threshold(
    store: MemoryVectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """阈值必须只在融合后生效：传给向量库的阈值恒为 0，否则词面救不回来。"""
    await _ingest(store, ["诺诺的真名是陈墨瞳。"])
    captured: dict[str, Any] = {}
    real_search = store.search

    async def spy(**kwargs: Any) -> tuple[list[tuple[Document, float | None]], int]:
        captured.update(kwargs)
        return await real_search(**kwargs)

    monkeypatch.setattr(store, "search", spy)

    await retrieval.search(KB_ID, "诺诺", k=2, strategy="hybrid", score_threshold=0.9)

    assert captured["score_threshold"] == 0.0


# ── 前置门：两路都要有实质证据才启用词面 ──────────────────────────────


async def test_sparse_skipped_when_neither_arm_has_evidence(
    store: MemoryVectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """负样本题的真实形状：「巴黎在哪个国家？」因香槟品牌「巴黎之花」被词面命中。

    稠密侧一条都没过阈值，词面最强匹配也只有 13.6（弱）→ 两边都没证据 → 不该启用词面，
    否则 context 非空、模型放弃拒答（实测拒答正确率 100% → 0%）。
    """
    monkeypatch.setattr(settings, "HYBRID_SPARSE_WEIGHT", 0.6)
    dense_doc, sparse_doc = _doc("向量命中的弱相关页", 0), _doc("巴黎之花美丽时光", 1)

    async def fake_dense(**_kwargs: Any) -> tuple[list[tuple[Document, float | None]], int]:
        return [(dense_doc, 0.2)], 0  # 0.2 < 0.3：稠密侧没有过阈值的候选

    monkeypatch.setattr(store, "search", fake_dense)
    calls = _spy_lexical(monkeypatch, [(sparse_doc, 13.63)])  # 实测值

    results, filtered_out = await retrieval.search(
        KB_ID, "巴黎在哪个国家？", k=2, strategy="hybrid", score_threshold=0.3
    )

    assert calls != []  # 查了，但没有采信
    assert results == []
    assert filtered_out == 1


async def test_sparse_used_when_lexical_match_is_strong(
    store: MemoryVectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """要救的题的真实形状：「诺诺的真名」两路都没过阈值，但词面最强匹配 24.8（强）。

    稠密侧救不了它（真匹配页只有 14.0 分、排在第 8），所以判据必须落在
    「词面最强匹配」这个 query 级信号上（详见 docs/evaluation.md 4.5）。
    """
    monkeypatch.setattr(settings, "HYBRID_SPARSE_WEIGHT", 0.6)
    dense_doc, sparse_doc = _doc("向量命中的弱相关页", 0), _doc("诺诺的真名是陈墨瞳。", 1)

    async def fake_dense(**_kwargs: Any) -> tuple[list[tuple[Document, float | None]], int]:
        return [(dense_doc, 0.2)], 0  # 稠密侧没有过阈值候选

    monkeypatch.setattr(store, "search", fake_dense)
    _spy_lexical(monkeypatch, [(sparse_doc, 24.83)])  # 实测值，高于门槛 20

    results, _filtered_out = await retrieval.search(
        KB_ID, "诺诺的真名是什么？", k=2, strategy="hybrid", score_threshold=0.3
    )

    assert [doc.page_content for doc, _ in results] == ["诺诺的真名是陈墨瞳。"]


async def test_sparse_used_when_dense_has_candidate_above_threshold(
    store: MemoryVectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """稠密侧有过阈值候选时即使词面很弱也启用（「或」的左半边）。"""
    monkeypatch.setattr(settings, "HYBRID_SPARSE_WEIGHT", 0.6)
    dense_doc, sparse_doc = _doc("向量命中的相关页", 0), _doc("同页的另一段", 1)

    async def fake_dense(**_kwargs: Any) -> tuple[list[tuple[Document, float | None]], int]:
        return [(dense_doc, 0.434)], 0  # 高于 0.3

    monkeypatch.setattr(store, "search", fake_dense)
    calls = _spy_lexical(monkeypatch, [(sparse_doc, 9.3)])  # sakura-who 的实测值，低于门槛

    results, filtered_out = await retrieval.search(
        KB_ID, "任意问题", k=2, strategy="hybrid", score_threshold=0.3
    )

    assert calls != []
    assert [doc.page_content for doc, _ in results] == ["同页的另一段", "向量命中的相关页"]
    assert filtered_out == 0


async def test_sparse_gate_threshold_is_configurable(
    store: MemoryVectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """门槛是配置项：调低到 10 时上面那个巧合匹配就会被采信（可观测、可回退）。"""
    monkeypatch.setattr(settings, "HYBRID_SPARSE_WEIGHT", 0.6)
    monkeypatch.setattr(settings, "HYBRID_MIN_SPARSE_SCORE", 10.0)

    async def fake_dense(**_kwargs: Any) -> tuple[list[tuple[Document, float | None]], int]:
        return [(_doc("弱相关页", 0), 0.2)], 0

    monkeypatch.setattr(store, "search", fake_dense)
    _spy_lexical(monkeypatch, [(_doc("巴黎之花美丽时光", 1), 13.63)])

    results, _filtered_out = await retrieval.search(
        KB_ID, "巴黎在哪个国家？", k=2, strategy="hybrid", score_threshold=0.3
    )

    assert [doc.page_content for doc, _ in results] == ["巴黎之花美丽时光"]


async def test_sparse_gate_counts_mmr_none_scores_as_no_support(
    store: MemoryVectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MMR 补充项是 ``score=None``，按契约不过阈值 —— 不能被当成「稠密侧有过阈值候选」。"""
    monkeypatch.setattr(settings, "HYBRID_SPARSE_WEIGHT", 0.6)

    async def fake_dense(**_kwargs: Any) -> tuple[list[tuple[Document, float | None]], int]:
        return [(_doc("MMR 补充项", 0), None)], 0

    monkeypatch.setattr(store, "search", fake_dense)
    calls = _spy_lexical(monkeypatch, [(_doc("弱词面命中", 1), 5.0)])  # 5.0 < 门槛 20

    results, _filtered_out = await retrieval.search(
        KB_ID, "任意问题", k=2, strategy="hybrid", score_threshold=0.3
    )

    assert calls != []  # 词面被查询了
    # 但只有 MMR 那条 score=None 留下（过阈值判定对它不生效），弱词面命中没有被采信
    assert [doc.page_content for doc, _ in results] == ["MMR 补充项"]


async def test_empty_kb_returns_empty_result(store: MemoryVectorStore) -> None:
    results, filtered_out = await retrieval.search(KB_ID, "任意问题", k=3, score_threshold=0.3)

    assert results == []
    assert filtered_out == 0


# ── 融合后的端到端行为 ────────────────────────────────────────────────


async def test_search_fuses_dense_and_sparse_scores(
    store: MemoryVectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "HYBRID_SPARSE_WEIGHT", 0.5)
    dense_doc, sparse_doc = _doc("向量命中", 0), _doc("词面命中", 1)

    async def fake_dense(**_kwargs: Any) -> tuple[list[tuple[Document, float | None]], int]:
        return [(dense_doc, 0.4)], 0

    monkeypatch.setattr(store, "search", fake_dense)
    _spy_lexical(monkeypatch, [(sparse_doc, 4.0)])

    results, filtered_out = await retrieval.search(
        KB_ID, "任意问题", k=2, strategy="hybrid", score_threshold=0.3
    )

    # 词面最高分归一化到 1.0 × 0.5 = 0.5，排在有分数的向量命中（0.4）之前
    assert [doc.page_content for doc, _ in results] == ["词面命中", "向量命中"]
    assert filtered_out == 0


async def test_search_filters_weak_lexical_hits(
    store: MemoryVectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "HYBRID_SPARSE_WEIGHT", 0.5)
    dense_doc, best, weak = _doc("向量命中", 0), _doc("最匹配", 1), _doc("沾了点边", 2)

    async def fake_dense(**_kwargs: Any) -> tuple[list[tuple[Document, float | None]], int]:
        # 0.35 过阈值：稠密侧「有支持」，词面这一路才启用（见前置门用例）
        return [(dense_doc, 0.35)], 0

    monkeypatch.setattr(store, "search", fake_dense)
    _spy_lexical(monkeypatch, [(best, 4.0), (weak, 1.0)])

    results, filtered_out = await retrieval.search(
        KB_ID, "任意问题", k=3, strategy="hybrid", score_threshold=0.3
    )

    # 4.0 → 1.0 × 0.5 = 0.50（保留）；1.0 → 0.25 × 0.5 = 0.125（滤掉）；向量 0.35 保留
    assert [doc.page_content for doc, _ in results] == ["最匹配", "向量命中"]
    assert filtered_out == 1


async def test_search_falls_back_to_dense_count_when_sparse_is_empty(
    store: MemoryVectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """词面没命中时不应改变向量侧的过滤计数（走的是同一条 finalize 路径）。"""
    await _ingest(store, ["诺诺的真名是陈墨瞳。", "完全无关的一句话。"])
    _spy_lexical(monkeypatch, [])

    results, filtered_out = await retrieval.search(
        KB_ID, "完全不相干的问题", k=8, strategy="hybrid", score_threshold=0.99
    )

    assert results == []
    assert filtered_out == 2  # 库里只有 2 个分块，阈值 0.99 全部挡掉


async def test_search_uses_configured_defaults(
    store: MemoryVectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _ingest(store, ["诺诺的真名是陈墨瞳。"])
    monkeypatch.setattr(settings, "TOP_K", 5)
    monkeypatch.setattr(settings, "RETRIEVAL_SCORE_THRESHOLD", 0.0)
    calls = _spy_lexical(monkeypatch, [])

    await retrieval.search(KB_ID, "诺诺")

    # k 与阈值只在 resolve_search_defaults 解析一处，词面侧拿到的 k 也必须是同一个
    assert calls == [(KB_ID, "诺诺", 5, None)]
