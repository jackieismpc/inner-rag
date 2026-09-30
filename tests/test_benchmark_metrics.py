"""benchmark 包的离线单测：指标算法、评测集 schema 校验与锚点校验。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmark import dataset as ds
from benchmark import metrics


def test_percentile_handles_empty_single_and_interpolation() -> None:
    assert metrics.percentile([], 0.5) == 0.0
    assert metrics.percentile([7.0], 0.95) == 7.0
    assert metrics.percentile([0.0, 10.0], 0.5) == 5.0
    assert metrics.percentile([1.0, 2.0, 3.0, 4.0], 1.0) == 4.0


def test_pages_hit_and_reciprocal_rank() -> None:
    assert metrics.pages_hit([12, 40], [40]) is True
    assert metrics.pages_hit([12, 40], [99]) is False
    assert metrics.pages_hit([12, 40], []) is False
    assert metrics.reciprocal_rank([12, 40, 41], [40, 41]) == 0.5
    assert metrics.reciprocal_rank([12], [40]) == 0.0


def test_page_hit_rate_counts_all_expected_pages() -> None:
    assert metrics.page_hit_rate([7, 9, 11], [7, 9]) == 1.0
    assert metrics.page_hit_rate([7], [7, 9]) == 0.5
    assert metrics.page_hit_rate([7], []) == 0.0


def test_keyword_coverage_requires_every_keyword_for_pass() -> None:
    assert metrics.keyword_coverage("Sakura 是路明非的花名", ["路明非", "花名"]) == 1.0
    assert metrics.keyword_coverage("Sakura 是路明非", ["路明非", "花名"]) == 0.5
    assert metrics.keyword_coverage("任意回答", []) == 1.0


def test_forbidden_and_refusal_detection() -> None:
    assert metrics.has_forbidden("是上杉绘梨衣", ["上杉绘梨衣"]) is True
    assert metrics.has_forbidden("是路明非", ["上杉绘梨衣"]) is False
    assert metrics.is_refusal("文档中没有找到相关信息。") is True
    assert metrics.is_refusal("Sakura 是路明非。") is False


def test_citation_precision_penalizes_extra_citations() -> None:
    assert metrics.citation_precision([5904, 6355], [5904, 6355]) == 1.0
    assert metrics.citation_precision([5904, 1], [5904, 6355]) == 0.5
    assert metrics.citation_precision([], [5904]) == 0.0


def test_citation_hit_is_per_question_not_per_span() -> None:
    """引用命中只看「有没有引对」，因此不受引用条数影响——这正是它与精度互补的原因。"""
    assert metrics.citation_hit([5904, 1, 2, 3, 4], [5904]) is True  # 引对一条即命中
    assert metrics.citation_hit([1, 2, 3, 4, 5], [5904]) is False
    assert metrics.citation_hit([], [5904]) is False
    assert metrics.citation_hit([5904], []) is False  # 负样本没有期望页，不算命中


def test_citation_hit_contrasts_with_precision_on_degenerate_citations() -> None:
    """退化引用的两个极端：精度会把它们拉向 0% / 100%，而命中只看对错。

    改动前后引用条数从 4.1 条变 5.0 条，精度因此不可比（真实踩到过的坑，
    见 docs/evaluation.md 4.4）；下面这两个断言就是那次误判的最小复现。
    """
    # 只引了一条，恰好是对的：精度 100%，但只有一条来源
    assert metrics.citation_precision([123], [123]) == 1.0
    assert metrics.citation_hit([123], [123]) is True

    # 一条都没引：精度 0%
    assert metrics.citation_precision([], [123]) == 0.0
    assert metrics.citation_hit([], [123]) is False


def test_summarize_separates_positives_from_negatives() -> None:
    results = [
        {
            "expect_refusal": False,
            "hit": True,
            "rr": 1.0,
            "page_hit": 1.0,
            "retrieval_ms": 10.0,
            "keyword_coverage": 1.0,
            "citation_precision": 1.0,
            "citation_hit": True,
            "forbidden_hit": False,
            "refusal": None,
            "total_ms": 100.0,
        },
        {
            "expect_refusal": False,
            "hit": False,
            "rr": 0.0,
            "page_hit": 0.0,
            "retrieval_ms": 30.0,
            "keyword_coverage": 0.5,
            "citation_precision": 0.0,
            "citation_hit": False,
            "forbidden_hit": True,
            "refusal": None,
            "total_ms": 200.0,
        },
        {
            "expect_refusal": True,
            "hit": False,
            "rr": 0.0,
            "page_hit": 0.0,
            "retrieval_ms": 20.0,
            "keyword_coverage": None,
            "citation_precision": None,
            "citation_hit": None,
            "forbidden_hit": None,
            "refusal": True,
            "total_ms": None,
        },
    ]
    summary = metrics.summarize(results)

    assert summary["items"] == 3
    assert summary["positives"] == 2
    assert summary["negatives"] == 1
    assert summary["recall_at_k"] == pytest.approx(0.5)
    assert summary["mrr"] == pytest.approx(0.5)
    assert summary["page_hit_rate"] == pytest.approx(0.5)
    assert summary["keyword_pass_rate"] == pytest.approx(0.5)
    assert summary["keyword_coverage"] == pytest.approx(0.75)
    assert summary["forbidden_rate"] == pytest.approx(0.5)
    assert summary["citation_precision"] == pytest.approx(0.5)
    assert summary["citation_hit_rate"] == pytest.approx(0.5)
    assert summary["refusal_accuracy"] == pytest.approx(1.0)
    assert summary["retrieval_p50_ms"] == pytest.approx(20.0)
    assert summary["retrieval_p95_ms"] == pytest.approx(29.0)
    assert summary["total_p50_ms"] == pytest.approx(150.0)
    assert summary["total_p95_ms"] == pytest.approx(195.0)


def test_summarize_returns_none_for_metrics_without_data() -> None:
    """老结果 JSON 没有 citation_hit 字段：聚合要返回 None，而不是当成 0 分。"""
    summary = metrics.summarize(
        [
            {
                "expect_refusal": False,
                "hit": True,
                "rr": 1.0,
                "page_hit": 1.0,
                "retrieval_ms": 10.0,
                "keyword_coverage": None,
                "forbidden_hit": None,
                "refusal": None,
                "total_ms": None,
            }
        ]
    )

    assert summary["citation_hit_rate"] is None
    assert summary["citation_precision"] is None
    assert summary["keyword_pass_rate"] is None
    assert summary["refusal_accuracy"] is None
    assert summary["total_p50_ms"] is None


def _valid_item(item_id: str = "x") -> dict:
    return {
        "id": item_id,
        "question": "问题是「Sakura 是谁」吗",
        "category": "fact",
        "expected_answer": "路明非",
        "answer_keywords": ["路明非"],
        "expect_refusal": False,
        "citations": [{"page": 1, "quote": "引用片段"}],
    }


def _write_jsonl(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "eval.jsonl"
    body = "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
    path.write_text(body + "\n", encoding="utf-8")
    return path


def test_load_eval_set_rejects_missing_field(tmp_path: Path) -> None:
    item = _valid_item()
    del item["answer_keywords"]
    with pytest.raises(ValueError, match="缺少字段"):
        ds.load_eval_set(_write_jsonl(tmp_path, [item]))


def test_load_eval_set_rejects_duplicate_ids(tmp_path: Path) -> None:
    records = [_valid_item("dup"), _valid_item("dup")]
    with pytest.raises(ValueError, match="重复 id"):
        ds.load_eval_set(_write_jsonl(tmp_path, records))


def test_negative_item_must_not_carry_citations(tmp_path: Path) -> None:
    item = _valid_item("neg")
    item.update({"expect_refusal": True, "answer_keywords": []})
    with pytest.raises(ValueError, match="负样本"):
        ds.load_eval_set(_write_jsonl(tmp_path, [item]))


def test_positive_item_requires_citations(tmp_path: Path) -> None:
    item = _valid_item()
    item["citations"] = []
    with pytest.raises(ValueError, match="必须有 citations"):
        ds.load_eval_set(_write_jsonl(tmp_path, [item]))


def test_validate_flags_item_without_fixture() -> None:
    problems = ds.validate([_valid_item("orphan")], [], pdf_path=None)
    assert any("没有对应的离线 fixture" in problem for problem in problems)


def test_load_fixtures_rejects_overlong_text(tmp_path: Path) -> None:
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir()
    (fixtures_dir / "long.txt").write_text("字" * 200, encoding="utf-8")
    entry = {"fixture_id": "long", "file": "long.txt", "page": 1, "eval_ids": ["x"]}
    (fixtures_dir / "index.jsonl").write_text(
        json.dumps(entry, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="超过 200 字"):
        ds.load_fixtures(fixtures_dir)


def test_repo_eval_set_matches_fixtures() -> None:
    """仓库里的评测集与 fixture 必须自洽（每条正样本都有短片段）。"""
    items = ds.load_eval_set()
    fixtures = ds.load_fixtures()

    assert len({item["id"] for item in items}) == len(items)
    covered = {eval_id for fixture in fixtures for eval_id in fixture["eval_ids"]}
    for item in items:
        if not item["expect_refusal"]:
            assert item["id"] in covered, f"{item['id']} 没有对应的离线 fixture"
    assert all(len(fixture["text"]) < 200 for fixture in fixtures)
    assert all(problem.startswith("[skip]") for problem in ds.validate(items, fixtures, None))


@pytest.mark.skipif(not ds.DEFAULT_PDF.exists(), reason="本地无 龙族.pdf，跳过原文锚点校验")
def test_eval_anchors_match_source_pdf() -> None:
    """评测集与 fixture 的每个片段都必须能在 PDF 对应页原文里找到。"""
    items = ds.load_eval_set()
    fixtures = ds.load_fixtures()
    assert ds.validate(items, fixtures) == []


def _load_gen_fixtures() -> Any:
    """按路径加载 scripts/gen_eval_fixtures.py（scripts/ 不是包，不能直接 import）。"""
    import importlib.util

    path = ds.REPO_ROOT / "scripts" / "gen_eval_fixtures.py"
    spec = importlib.util.spec_from_file_location("gen_eval_fixtures", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_excerpt_fits_under_the_limit_on_a_long_page() -> None:
    """窗口上限要留余量，否则长页永远裁不出片段。

    旧实现把窗口扩到正好 199 字，紧接着又被 `len(text) >= 199` 否掉——源书几乎没有短于
    199 字的页，所以它等价于「这个脚本裁不出任何片段」，只是失败长相很像「这几条题定位不到」。
    """
    module = _load_gen_fixtures()
    page = "前" * 500 + "关键证据在这里" + "后" * 500

    excerpt, reason = module.excerpt_around(page, "关键证据在这里", 199)

    assert reason == ""
    assert excerpt is not None
    assert len(excerpt) < 200
    assert "关键证据在这里" in excerpt


def test_excerpt_stays_a_substring_of_the_page() -> None:
    """裁出来的必须是页文本的**原样子串**——锚点校验就是这么比的。"""
    module = _load_gen_fixtures()
    page = "开头一句。\n\n    引用片段在这一行里   \n结尾一句。"

    excerpt, _ = module.excerpt_around(page, "引用片段在这一行里", 199)

    assert excerpt is not None
    assert excerpt in page


def test_excerpt_separates_the_two_failure_reasons() -> None:
    """「引用不在这页」与「引用本身超长」要分开报：前者是题目/页码问题，后者只是上限问题。"""
    module = _load_gen_fixtures()

    missing, missing_reason = module.excerpt_around("这是一页正文", "别处的句子", 199)
    assert missing is None
    assert "不在这一页" in missing_reason

    long_quote = "长" * 250
    too_long, too_long_reason = module.excerpt_around("正文" + long_quote + "正文", long_quote, 199)
    assert too_long is None
    assert "上限" in too_long_reason


@pytest.mark.skipif(not ds.DEFAULT_PDF.exists(), reason="本地无 龙族.pdf，跳过提取器一致性校验")
def test_page_text_is_the_same_as_what_ingest_parsed(tmp_path: Path) -> None:
    """锚点校验读到的文本必须与**入库解析出来的文本**同源。

    这里曾经是两套提取器：校验与裁 fixture 用 pymupdf 的 `get_text()`，而入库解析
    （`inner_rag.services.parser` 的 PDF 分支）用 pypdf 的 `extract_text()`。实测源书 p9
    两边给出 222 字 / 184 字且断句不同，于是「引用片段必须出现在该页原文里」这条校验
    比的是另一份文本——正确的题被判成锚点错误。这条用例用**真实 parser 解析单页**来钉住一致性。
    """
    from pypdf import PdfReader, PdfWriter

    from inner_rag.services.parser import DocumentParser

    page = 9
    source = PdfReader(str(ds.DEFAULT_PDF))
    writer = PdfWriter()
    writer.add_page(source.pages[page - 1])
    single_page = tmp_path / "single.pdf"
    with single_page.open("wb") as handle:
        writer.write(handle)

    documents, _ = DocumentParser().parse(str(single_page))
    ingested = documents[0].page_content

    assert ingested  # 解析出了东西，否则下面那句只是「空 == 空」
    assert ingested == ds.PageText(ds.DEFAULT_PDF)(page)


# ── 「答案不在库里」的题要被摘掉 ────────────────────────────────────────
#
# 评测集照全库出题，而小库只收了源书的一部分页。不筛的话，小库上 90% 的题会以
# 「召回为空」记 0 分，整套指标量的是拒答策略而不是检索质量。筛选依据是库的**真实内容**
# （分块元数据里的 page），不是建库脚本里的页窗口常量——后者会随窗口调整而过期。


def test_filter_by_evidence_skips_incomplete_items(monkeypatch: pytest.MonkeyPatch) -> None:
    from benchmark import run_bench

    monkeypatch.setattr(run_bench, "_pages_in_store", lambda _kb_id: {10, 11, 20})
    items = [
        {"id": "full", "citations": [{"page": 10}, {"page": 11}]},
        {"id": "partial", "citations": [{"page": 10}, {"page": 999}]},
        {"id": "absent", "citations": [{"page": 999}]},
        {"id": "refusal", "citations": [], "expect_refusal": True},
    ]

    kept, absent, partial = run_bench._filter_by_evidence(items, 1, "skip")

    assert [item["id"] for item in kept] == ["full", "refusal"]
    assert absent == ["absent"]
    assert partial == ["partial"]


def test_filter_by_evidence_can_turn_absent_items_into_refusals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """小库上「答案根本不在库里」的题，正好是天然的拒答场景，别浪费。

    改判后它们进 negatives（`expect_refusal=True`），于是 `refusal_accuracy` 的分母里
    包含了这批真实分布的问题；同时必须**复制**而不是原地改，免得污染调用方的 items。
    """
    from benchmark import run_bench

    monkeypatch.setattr(run_bench, "_pages_in_store", lambda _kb_id: {10})
    items = [
        {"id": "absent", "citations": [{"page": 999}], "expect_refusal": False},
        {"id": "partial", "citations": [{"page": 10}, {"page": 999}], "expect_refusal": False},
        {"id": "full", "citations": [{"page": 10}], "expect_refusal": False},
    ]

    kept, absent, partial = run_bench._filter_by_evidence(items, 1, "refuse")

    assert [(item["id"], item["expect_refusal"]) for item in kept] == [
        ("absent", True),
        ("full", False),
    ]
    assert absent == ["absent"] and partial == ["partial"]
    # 原对象没被改（partial 被跳过、absent 被改判，原列表都保持原样）
    assert [item["expect_refusal"] for item in items] == [False, False, False]


def test_filter_by_evidence_keeps_everything_when_the_kb_is_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """全库上不该动任何题——否则「筛掉难例把分数做上去」就成了一条捷径。"""
    from benchmark import run_bench

    monkeypatch.setattr(run_bench, "_pages_in_store", lambda _kb_id: set(range(1, 12_000)))
    items = [
        {"id": "a", "citations": [{"page": 9}]},
        {"id": "b", "citations": [{"page": 11063}]},
        {"id": "neg", "citations": [], "expect_refusal": True},
    ]

    for mode in ("skip", "refuse"):
        kept, absent, partial = run_bench._filter_by_evidence(items, 7, mode)
        assert len(kept) == 3, f"{mode} 模式在全库上跳了题"
        assert absent == [] and partial == []
