"""查询改写插件点测试：契约（首条恒为原查询）、内置改写器的行为、以及多查询融合。

改写是这个项目里**最容易好心办坏事**的一层：多一路查询必然多一份成本与噪声，
所以契约必须钉死「原查询永远在第一位」——最坏情况只是多跑一次召回，而不是更差。
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.documents import Document

from inner_rag.core.config import settings
from inner_rag.plugins.registry import query_rewriters
from inner_rag.services import query_rewrite, retrieval
from inner_rag.services import vector_store as vector_store_package
from inner_rag.services.lexical import lexical_index
from inner_rag.services.query_rewrite import (
    AliasQueryRewriter,
    KeywordsQueryRewriter,
    LLMQueryRewriter,
    NoopQueryRewriter,
    parse_aliases,
    strip_stopwords,
)
from inner_rag.services.vector_store.memory_store import MemoryVectorStore

KB_ID = 7779


def _doc(content: str, chunk_index: int, doc_id: int = 1) -> Document:
    return Document(
        page_content=content,
        metadata={"doc_id": str(doc_id), "chunk_index": str(chunk_index), "page": 1},
    )


# ── 基础工具 ──────────────────────────────────────────────────────────


def test_strip_stopwords_removes_interrogative_frame() -> None:
    # 停用词替换成空格而不是直接删除：保留词边界，日志里也更容易看出被抹掉了什么
    assert strip_stopwords("诺诺的真名是什么？") == "诺诺 真名"


def test_strip_stopwords_keeps_latin_entity() -> None:
    assert strip_stopwords("Sakura是谁？") == "Sakura"


def test_strip_stopwords_returns_empty_for_pure_stopwords() -> None:
    assert strip_stopwords("是什么？") == ""


def test_parse_aliases_reads_pairs() -> None:
    assert parse_aliases("Sakura=路明非,绘梨衣=上杉绘梨衣") == {
        "Sakura": "路明非",
        "绘梨衣": "上杉绘梨衣",
    }


def test_parse_aliases_skips_malformed_entries() -> None:
    assert parse_aliases("没有等号,Sakura=路明非,=值,名=") == {"Sakura": "路明非"}


def test_parse_aliases_handles_empty_config() -> None:
    assert parse_aliases("") == {}


# ── 契约：首条恒为原查询 ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "rewriter",
    [NoopQueryRewriter(), AliasQueryRewriter(), KeywordsQueryRewriter()],
)
async def test_rewriters_never_drop_the_original_query(rewriter: Any) -> None:
    queries = await rewriter.rewrite(KB_ID, "诺诺的真名是什么？")

    assert queries[0] == "诺诺的真名是什么？"


async def test_noop_returns_exactly_one_query() -> None:
    assert await NoopQueryRewriter().rewrite(KB_ID, "任意问题") == ["任意问题"]


# ── alias ─────────────────────────────────────────────────────────────


async def test_alias_rewriter_adds_a_parallel_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "QUERY_ALIASES", "Sakura=路明非")

    queries = await AliasQueryRewriter().rewrite(KB_ID, "Sakura是谁？")

    assert queries == ["Sakura是谁？", "路明非是谁？"]


async def test_alias_rewriter_is_case_insensitive_for_latin_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "QUERY_ALIASES", "sakura=路明非")

    queries = await AliasQueryRewriter().rewrite(KB_ID, "SAKURA 的言灵")

    assert queries[1] == "路明非 的言灵"


async def test_alias_rewriter_is_inert_without_a_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "QUERY_ALIASES", "Sakura=路明非")

    assert await AliasQueryRewriter().rewrite(KB_ID, "恺撒是谁？") == ["恺撒是谁？"]


async def test_alias_rewriter_is_inert_without_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "QUERY_ALIASES", "")

    assert await AliasQueryRewriter().rewrite(KB_ID, "Sakura是谁？") == ["Sakura是谁？"]


# ── keywords ──────────────────────────────────────────────────────────


async def test_keywords_rewriter_adds_a_content_word_query() -> None:
    queries = await KeywordsQueryRewriter().rewrite(KB_ID, "上杉绘梨衣的言灵是什么？")

    assert queries == ["上杉绘梨衣的言灵是什么？", "上杉绘梨衣 言灵"]


async def test_keywords_rewriter_is_inert_when_nothing_is_removed() -> None:
    assert await KeywordsQueryRewriter().rewrite(KB_ID, "上杉绘梨衣") == ["上杉绘梨衣"]


# ── llm ───────────────────────────────────────────────────────────────


async def test_llm_rewriter_parses_lines_and_dedupes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeMessage:
        content = "1. 路明非的花名是什么\n- 谁被叫作 Sakura\n路明非的花名是什么\n"

    class _FakeLLM:
        async def ainvoke(self, _prompt: str) -> _FakeMessage:
            return _FakeMessage()

    monkeypatch.setattr(LLMQueryRewriter, "_get_llm", lambda _self: _FakeLLM())

    queries = await LLMQueryRewriter().rewrite(KB_ID, "Sakura是谁？")

    assert queries == ["Sakura是谁？", "路明非的花名是什么", "谁被叫作 Sakura"]


async def test_llm_rewriter_falls_back_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class _BoomLLM:
        async def ainvoke(self, _prompt: str) -> Any:
            msg = "上游 500"
            raise RuntimeError(msg)

    monkeypatch.setattr(LLMQueryRewriter, "_get_llm", lambda _self: _BoomLLM())

    assert await LLMQueryRewriter().rewrite(KB_ID, "Sakura是谁？") == ["Sakura是谁？"]


# ── 模块级入口：去重 / 保序 / 截断 ────────────────────────────────────


async def test_module_rewrite_truncates_to_configured_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Many:
        name = "many"

        async def rewrite(self, _kb_id: int, query: str) -> list[str]:
            return [query, "v1", "v2", "v3", "v4"]

    monkeypatch.setattr(query_rewrite, "query_rewriter", _Many())
    monkeypatch.setattr(settings, "QUERY_REWRITE_MAX_QUERIES", 3)

    assert await query_rewrite.rewrite(KB_ID, "原问题") == ["原问题", "v1", "v2"]


async def test_module_rewrite_drops_duplicates_and_blank_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Messy:
        name = "messy"

        async def rewrite(self, _kb_id: int, query: str) -> list[str]:
            return ["  ", "v1", query, "v1", "  v2  "]

    monkeypatch.setattr(query_rewrite, "query_rewriter", _Messy())

    assert await query_rewrite.rewrite(KB_ID, "原问题") == ["原问题", "v1", "v2"]


async def test_module_rewrite_puts_original_first_even_if_rewriter_forgets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """实现违约（把原查询丢了）时由入口兜住：最坏也只是多跑一路，不会更差。"""

    class _Forgetful:
        name = "forgetful"

        async def rewrite(self, _kb_id: int, _query: str) -> list[str]:
            return ["只有变体"]

    monkeypatch.setattr(query_rewrite, "query_rewriter", _Forgetful())

    assert await query_rewrite.rewrite(KB_ID, "原问题") == ["原问题", "只有变体"]


async def test_module_rewrite_falls_back_when_everything_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Empty:
        name = "empty"

        async def rewrite(self, _kb_id: int, _query: str) -> list[str]:
            return []

    monkeypatch.setattr(query_rewrite, "query_rewriter", _Empty())

    assert await query_rewrite.rewrite(KB_ID, "原问题") == ["原问题"]


# ── 注册表 ────────────────────────────────────────────────────────────


def test_registry_exposes_every_builtin() -> None:
    assert set(query_rewriters.names()) == {"none", "alias", "keywords", "llm"}
    assert query_rewriters.settings_key == "QUERY_REWRITE_BACKEND"


def test_build_query_rewriter_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="不支持的 QUERY_REWRITE_BACKEND"):
        query_rewrite.build_query_rewriter("hyde-from-nowhere")


def test_default_backend_is_noop() -> None:
    assert settings.QUERY_REWRITE_BACKEND == "none"
    assert query_rewrite.query_rewriter.name == "none"


# ── 接进检索层：多查询共享同一份融合与阈值 ────────────────────────────


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> MemoryVectorStore:
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


async def test_multi_query_runs_one_search_per_variant(
    store: MemoryVectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _ingest(store, ["路明非的花名是 Sakura。", "诺诺的真名是陈墨瞳。"])
    monkeypatch.setattr(settings, "QUERY_ALIASES", "Sakura=路明非")
    monkeypatch.setattr(settings, "QUERY_REWRITE_BACKEND", "alias")
    monkeypatch.setattr(query_rewrite, "query_rewriter", AliasQueryRewriter())

    seen: list[str] = []
    real_search = store.search

    async def spy(**kwargs: Any) -> Any:
        seen.append(kwargs["query"])
        return await real_search(**kwargs)

    monkeypatch.setattr(store, "search", spy)

    results, _filtered_out = await retrieval.search(
        KB_ID, "Sakura", k=3, strategy="similarity", score_threshold=0.0
    )

    assert seen == ["Sakura", "路明非"]
    assert results  # 变体命中不丢：两路结果在同一个融合里


async def test_single_query_keeps_the_old_single_call_behaviour(
    store: MemoryVectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """默认（none）下检索行为与不做改写的版本完全一致——一次查询、一次向量检索。"""
    await _ingest(store, ["诺诺的真名是陈墨瞳。"])
    monkeypatch.setattr(query_rewrite, "query_rewriter", NoopQueryRewriter())

    calls: list[str] = []
    real_search = store.search

    async def spy(**kwargs: Any) -> Any:
        calls.append(kwargs["query"])
        return await real_search(**kwargs)

    monkeypatch.setattr(store, "search", spy)

    await retrieval.search(KB_ID, "诺诺", k=2, strategy="similarity", score_threshold=0.0)

    assert calls == ["诺诺"]
