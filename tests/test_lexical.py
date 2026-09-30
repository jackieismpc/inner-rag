"""词面检索（BM25）契约测试。

测的是 ``docs/architecture.md`` 里定下来的语义，不是实现细节：

* 分词对专名有效（中文 bigram）、对英文数字按词小写；
* BM25 把「字面匹配」的分块排到前面，且分数**无上界**；
* ``normalize_scores`` 把无上界的分数压到 ``[0, 1]``（否则融合后阈值失去意义）；
* 索引按知识库缓存，只在 ``invalidate`` 之后重建——缓存是这里唯一的状态，
  错了会变成「入库了却检索不到」，属于最难查的一类 bug。

索引只依赖后端的 ``iter_chunks``，所以用假后端替换即可，不需要真的向量库。
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from inner_rag.services import vector_store as vector_store_package
from inner_rag.services.lexical import LexicalIndex, normalize_scores, tokenize
from inner_rag.services.lexical import lexical_index as global_index

KB_ID = 4242


def _doc(content: str, chunk_index: int, doc_id: int = 1) -> Document:
    return Document(
        page_content=content,
        metadata={"doc_id": str(doc_id), "chunk_index": str(chunk_index), "page": 1},
    )


class _FakeStore:
    """只实现 ``iter_chunks``：词面索引对后端的要求就这一个方法。"""

    def __init__(self, chunks: list[Document] | None = None) -> None:
        self.chunks = list(chunks or [])

    def iter_chunks(self, kb_id: int) -> list[Document]:
        return list(self.chunks) if kb_id == KB_ID else []


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> tuple[LexicalIndex, _FakeStore]:
    """返回 (干净索引, 假后端)；用例往里塞分块即可。

    ``LexicalIndex._get`` 在调用时才 import ``vector_service``（避免与向量库包互相 import），
    因此替换包命名空间上的这个名字就能生效。
    """
    store = _FakeStore()
    monkeypatch.setattr(vector_store_package, "vector_service", store)
    return LexicalIndex(), store


# ── 分词 ──────────────────────────────────────────────────────────────


def test_tokenize_lowercases_latin_words() -> None:
    assert tokenize("Hello BM25 WORLD") == ["hello", "bm25", "world"]


def test_tokenize_keeps_single_cjk_chars_and_bigrams() -> None:
    # 单字必须保留：query 只有一个汉字时（如「龙」）bigram 为空，只有单字能匹配
    assert tokenize("陈墨瞳") == ["陈", "墨", "瞳", "陈墨", "墨瞳"]


def test_tokenize_mixes_latin_and_cjk() -> None:
    assert tokenize("inner-rag 向量库") == ["inner", "rag", "向", "量", "库", "向量", "量库"]


def test_tokenize_ignores_punctuation_and_empty_input() -> None:
    assert tokenize("，。！？——") == []
    assert tokenize("") == []


# ── BM25 打分 ─────────────────────────────────────────────────────────


def test_search_ranks_literal_match_first(env: tuple[LexicalIndex, _FakeStore]) -> None:
    index, store = env
    store.chunks = [_doc("完全无关的一句话。", 0), _doc("诺诺的真名是陈墨瞳。", 1)]

    hits = index.search(KB_ID, "诺诺的真名是什么", k=2)

    assert [doc.metadata["chunk_index"] for doc, _ in hits] == ["1", "0"]
    assert hits[0][1] > hits[1][1]


def test_search_scores_are_unbounded(env: tuple[LexicalIndex, _FakeStore]) -> None:
    """原始 BM25 分可以大于 1——这正是必须先归一再融合的原因。"""
    index, store = env
    store.chunks = [_doc("绘梨衣的言灵是审判。", 0)]

    ((_, score),) = index.search(KB_ID, "绘梨衣的言灵", k=1)

    assert score > 1.0


def test_search_returns_empty_for_unmatched_query(env: tuple[LexicalIndex, _FakeStore]) -> None:
    index, store = env
    store.chunks = [_doc("完全无关的一句话。", 0)]

    assert index.search(KB_ID, "zzzz", k=5) == []


def test_search_returns_empty_for_empty_kb(env: tuple[LexicalIndex, _FakeStore]) -> None:
    index, _store = env

    assert index.search(KB_ID, "任意问题", k=5) == []


def test_search_honours_filter_doc_ids(env: tuple[LexicalIndex, _FakeStore]) -> None:
    index, store = env
    store.chunks = [
        _doc("诺诺的真名是陈墨瞳。", 0, doc_id=1),
        _doc("诺诺的真名是陈墨瞳。", 1, doc_id=2),
    ]

    hits = index.search(KB_ID, "诺诺的真名", k=5, filter_doc_ids=[2])

    assert [doc.metadata["chunk_index"] for doc, _ in hits] == ["1"]


# ── 索引缓存与失效 ────────────────────────────────────────────────────


def test_size_is_none_before_first_search(env: tuple[LexicalIndex, _FakeStore]) -> None:
    index, _store = env

    assert index.size(KB_ID) is None


def test_index_is_built_once_and_cached(env: tuple[LexicalIndex, _FakeStore]) -> None:
    index, store = env
    store.chunks = [_doc("诺诺的真名是陈墨瞳。", 0)]
    index.search(KB_ID, "诺诺", k=1)
    assert index.size(KB_ID) == 1

    # 后端内容变了但没通知失效：缓存必须保持原样，否则等于每次查询都重建索引
    store.chunks = [_doc("内容一。", 0), _doc("内容二。", 1)]

    assert index.size(KB_ID) == 1


def test_invalidate_single_kb_then_rebuild(env: tuple[LexicalIndex, _FakeStore]) -> None:
    index, store = env
    store.chunks = [_doc("诺诺的真名是陈墨瞳。", 0)]
    index.search(KB_ID, "诺诺", k=1)

    index.invalidate(KB_ID)
    assert index.size(KB_ID) is None

    store.chunks = [_doc("内容一。", 0), _doc("内容二。", 1)]
    index.search(KB_ID, "内容", k=1)

    assert index.size(KB_ID) == 2


def test_invalidate_without_kb_id_clears_everything(
    env: tuple[LexicalIndex, _FakeStore],
) -> None:
    index, store = env
    store.chunks = [_doc("诺诺的真名是陈墨瞳。", 0)]
    index.search(KB_ID, "诺诺", k=1)
    assert index.size(KB_ID) == 1

    index.invalidate()

    assert index.size(KB_ID) is None


def test_invalidate_unknown_kb_is_a_noop(env: tuple[LexicalIndex, _FakeStore]) -> None:
    index, _store = env

    index.invalidate(987654)  # 不存在的库：不能抛异常


# ── 归一化 ────────────────────────────────────────────────────────────


def test_normalize_scores_maps_max_to_one() -> None:
    hits = [(_doc("a", 0), 4.0), (_doc("b", 1), 2.0), (_doc("c", 2), 1.0)]

    normalized = normalize_scores(hits)

    assert [score for _doc_, score in normalized] == [1.0, 0.5, 0.25]


def test_normalize_scores_handles_all_zero() -> None:
    hits = [(_doc("a", 0), 0.0), (_doc("b", 1), 0.0)]

    assert [score for _doc_, score in normalize_scores(hits)] == [0.0, 0.0]


def test_normalize_scores_handles_empty() -> None:
    assert normalize_scores([]) == []


# ── 全局单例 ──────────────────────────────────────────────────────────


def test_global_index_is_the_shared_instance() -> None:
    """``lexical_index`` 是全进程共享的那一份（入库 / 删除 / 清缓存处调的就是它）。"""
    assert isinstance(global_index, LexicalIndex)
