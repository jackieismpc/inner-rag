"""benchmark 包的离线单测：指标算法、评测集 schema 校验与锚点校验。"""

from __future__ import annotations

import json
from pathlib import Path

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
    assert summary["refusal_accuracy"] == pytest.approx(1.0)
    assert summary["retrieval_p50_ms"] == pytest.approx(20.0)
    assert summary["retrieval_p95_ms"] == pytest.approx(29.0)
    assert summary["total_p50_ms"] == pytest.approx(150.0)
    assert summary["total_p95_ms"] == pytest.approx(195.0)


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
