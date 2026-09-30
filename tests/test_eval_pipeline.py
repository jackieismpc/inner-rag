"""Phase 6 评测链路的离线用例（L1：不联网、不需要 PDF、不调 judge 模型）。

覆盖的是「评测工具自己算得对不对」——脚本写错比模型答错更致命，因为它会让
所有结论都不可信（见 docs/evaluation.md 第 8 节）：

* 页区间命中判定（Recall / MRR / 引用精度）；
* 分块元数据确实带上 page_start / page_end；
* 建库脚本的页窗口与 manifest 字段；
* judge 输出解析（外部模型的输出不保证干净）与失败归因。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
from langchain_core.documents import Document

from benchmark import langsmith_sync, metrics
from inner_rag.services.vector_store.base import CHUNK_METADATA_FIELDS, page_span, prepare_chunks

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_script(name: str):
    """按文件路径加载 scripts/ 下的脚本（它们不是包的一部分，不能 import 模块名）。"""
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_script_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


build_eval_kb = _load_script("build_eval_kb")
eval_answer = _load_script("eval_answer")


# ── 页区间命中判定 ──────────────────────────────────────────────────────


def test_spans_hit_uses_closed_interval() -> None:
    spans = [(100, 105), (200, 210)]
    assert metrics.spans_hit(spans, [103]) is True
    assert metrics.spans_hit(spans, [99]) is False
    assert metrics.spans_hit(spans, []) is False
    assert metrics.spans_hit([], [103]) is False


def test_spans_mrr_and_page_hit_rate() -> None:
    spans = [(10, 20), (30, 40), (50, 60)]
    assert metrics.spans_reciprocal_rank(spans, [35]) == pytest.approx(0.5)
    assert metrics.spans_reciprocal_rank(spans, [99]) == 0.0
    # 三个期望页里命中两个（35 在第二个区间、55 在第三个）
    assert metrics.spans_page_hit_rate(spans, [5, 35, 55]) == pytest.approx(2 / 3)


def test_single_page_helpers_still_agree_with_spans() -> None:
    """旧的单点口径必须与区间口径一致，否则历史报告的数字无法比较。"""
    pages = [1, 5, 9]
    expected = [5, 7]
    assert metrics.pages_hit(pages, expected) == metrics.spans_hit(
        metrics.spans_from_pages(pages), expected
    )
    assert metrics.reciprocal_rank(pages, expected) == metrics.spans_reciprocal_rank(
        metrics.spans_from_pages(pages), expected
    )
    assert metrics.citation_precision(pages, expected) == metrics.spans_citation_precision(
        metrics.spans_from_pages(pages), expected
    )


def test_citation_precision_penalizes_extra_citations() -> None:
    expected = [5904]
    assert metrics.spans_citation_precision([(5904, 5904)], expected) == 1.0
    # 引了 4 条只有 1 条对：多引无关页要显性拉低精度
    assert metrics.spans_citation_precision(
        [(5904, 5904), (1, 1), (2, 2), (3, 3)], expected
    ) == pytest.approx(0.25)
    assert metrics.spans_citation_precision([], expected) == 0.0


# ── 分块元数据 ──────────────────────────────────────────────────────────


def test_prepare_chunks_writes_page_span() -> None:
    documents = [
        Document(page_content="第一页内容。" * 20, metadata={"source": "x.pdf", "page": 7}),
        Document(page_content="无页码内容。", metadata={"source": "x.txt"}),
    ]
    chunks = prepare_chunks(documents, kb_id=3, doc_id=11, filename="x.pdf")
    assert chunks, "至少应产出一个分块"

    paged = [chunk for chunk in chunks if chunk.metadata.get("page") == 7]
    assert paged, "带页码的文档应保留 page"
    for chunk in paged:
        assert chunk.metadata["page_start"] == 7
        assert chunk.metadata["page_end"] == 7
        assert page_span(chunk) == (7, 7)
        # 元数据是白名单重建：解析器的 source 不该进向量库
        assert "source" not in chunk.metadata
        for field in CHUNK_METADATA_FIELDS:
            assert field in chunk.metadata

    unpaged = [chunk for chunk in chunks if "page" not in chunk.metadata]
    for chunk in unpaged:
        assert page_span(chunk) is None


def test_page_span_falls_back_to_single_page() -> None:
    """旧索引只有 page：区间必须退化成单点，而不是读不到页码。"""
    document = Document(page_content="x", metadata={"page": 42})
    assert page_span(document) == (42, 42)
    assert page_span(Document(page_content="x", metadata={})) is None


# ── 建库脚本（纯函数部分） ──────────────────────────────────────────────


def test_resolve_windows_small_covers_all_anchor_pages() -> None:
    """小库必须覆盖评测集里的每一个锚点页，否则该题必然「检索失败」而被误判为模型差。"""
    from benchmark import dataset as ds

    items = ds.load_eval_set()
    windows = build_eval_kb.resolve_windows("small")
    covered = set(build_eval_kb.window_pages(windows))
    anchors = {page for item in items for page in ds.expected_pages(item)}
    assert anchors, "评测集至少要有一个锚点页"
    missing = sorted(anchors - covered)
    assert not missing, f"小库页窗口漏掉锚点页：{missing}"


def test_window_pages_validates_and_dedups() -> None:
    assert build_eval_kb.window_pages([(5, 7), (7, 8)]) == [5, 6, 7, 8]
    with pytest.raises(ValueError, match="start > end"):
        build_eval_kb.window_pages([(9, 3)])


def test_resolve_windows_full_needs_total_pages() -> None:
    assert build_eval_kb.resolve_windows("full", 11138) == [(1, 11138)]
    with pytest.raises(ValueError):
        build_eval_kb.resolve_windows("full")
    with pytest.raises(ValueError, match="未知 profile"):
        build_eval_kb.resolve_windows("tiny")


def test_build_manifest_has_comparable_fields() -> None:
    """manifest 字段不齐，两次评测的数字就没法比（换 embedding 后尤其危险）。"""
    manifest = build_eval_kb.build_manifest(
        kb_id=1,
        name="dragon_king_small",
        profile="small",
        source=REPO_ROOT / "data" / "uploads" / "龙族.pdf",
        windows=[(1, 2)],
        pages=2,
        chunks=2,
        embedding_identity="openrouter:test",
        duration_s=1.23,
        config={
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "embedding_max_input_chars": 400,
            "app_version": "0.3.0",
            "git_commit": "abc1234",
        },
    )
    for field in (
        "kb_id",
        "name",
        "profile",
        "source_sha256",
        "page_windows",
        "pages",
        "chunks",
        "chunk_size",
        "chunk_overlap",
        "embedding_identity",
        "embedding_max_input_chars",
        "built_at",
        "duration_s",
    ):
        assert field in manifest, f"manifest 缺字段 {field}"
    assert manifest["page_windows"] == [[1, 2]]
    assert manifest["duration_s"] == 1.2


# ── judge 解析与归因 ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected_correct"),
    [
        ('{"correct": 1, "faithfulness": 1, "reason": "对"}', 1),
        ('```json\n{"correct": 0, "faithfulness": 0, "reason": "错"}\n```', 0),
        ('前缀废话 {"correct": true, "faithfulness": "是", "reason": ""} 后缀', 1),
        ("完全不是 JSON", None),
        ("", None),
    ],
)
def test_parse_judge_response_tolerates_messy_output(
    raw: str, expected_correct: int | None
) -> None:
    verdict = eval_answer.parse_judge_response(raw)
    assert verdict["correct"] == expected_correct
    if expected_correct is None:
        assert "judge" in verdict["reason"]


def test_estimate_cost_usd_without_price_is_zero() -> None:
    """没有价格表就不许报费用：编一个单价出来比不报更糟。"""
    assert eval_answer.estimate_cost_usd(1000, 1000, 0.0, 0.0) == 0.0
    assert eval_answer.estimate_cost_usd(1_000_000, 0, 0.5, 2.0) == pytest.approx(0.5)


def test_attribute_failure_distinguishes_retrieval_and_generation() -> None:
    assert (
        eval_answer.attribute_failure({"expect_refusal": False, "hit": False, "filtered_out": 0})
        == "检索失败（证据未召回）"
    )
    assert "阈值过严" in eval_answer.attribute_failure(
        {"expect_refusal": False, "hit": False, "filtered_out": 3}
    )
    assert "生成失败" in eval_answer.attribute_failure(
        {
            "expect_refusal": False,
            "hit": True,
            "judge_correct": 0,
            "keyword_coverage": 0.0,
            "citation_precision": 1.0,
        }
    )
    assert "引用错误" in eval_answer.attribute_failure(
        {
            "expect_refusal": False,
            "hit": True,
            "judge_correct": 0,
            "keyword_coverage": 0.0,
            "citation_precision": 0.0,
        }
    )
    assert (
        eval_answer.attribute_failure({"expect_refusal": True, "refusal": False})
        == "拒答失败（负样本未拒绝）"
    )


def test_summarize_includes_judge_and_cost_dimensions() -> None:
    results = [
        {
            "expect_refusal": False,
            "hit": True,
            "rr": 1.0,
            "page_hit": 1.0,
            "retrieval_ms": 10.0,
            "keyword_coverage": 1.0,
            "forbidden_hit": False,
            "refusal": False,
            "citation_precision": 1.0,
            "total_ms": 20.0,
            "judge_correct": 1,
            "faithfulness": 1.0,
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "cost_usd": 0.001,
        },
        {
            "expect_refusal": True,
            "hit": False,
            "rr": 0.0,
            "page_hit": 0.0,
            "retrieval_ms": 12.0,
            "keyword_coverage": 1.0,
            "forbidden_hit": False,
            "refusal": True,
            "citation_precision": 0.0,
            "total_ms": 15.0,
            "judge_correct": None,
            "faithfulness": None,
            "prompt_tokens": 80,
            "completion_tokens": 20,
            "cost_usd": 0.001,
        },
    ]
    summary = metrics.summarize(results)
    assert summary["judge_accuracy"] == 1.0
    assert summary["faithfulness"] == 1.0
    assert summary["refusal_accuracy"] == 1.0
    assert summary["false_refusal_rate"] == 0.0
    assert summary["prompt_tokens"] == 180
    assert summary["cost_usd"] == pytest.approx(0.002)


# ── LangSmith 联动（可选，关闭时必须能完整跑完） ────────────────────────


class _FakeClient:
    """记录调用序列的假客户端：验证「回写什么」，不需要真实 Key。"""

    def __init__(self, *, fail_on_feedback: bool = False) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._fail_on_feedback = fail_on_feedback

    def has_dataset(self, dataset_name: str) -> bool:
        self.calls.append(("has_dataset", {"dataset_name": dataset_name}))
        return False

    def create_dataset(self, dataset_name: str, description: str) -> dict:
        self.calls.append(("create_dataset", {"dataset_name": dataset_name}))
        return {"id": "ds-1"}

    def create_example(self, **kwargs: Any) -> dict:
        self.calls.append(("create_example", kwargs))
        return {"id": "ex-1"}

    def create_feedback(self, **kwargs: Any) -> dict:
        if self._fail_on_feedback:
            msg = "boom"
            raise RuntimeError(msg)
        self.calls.append(("create_feedback", kwargs))
        return {"id": "fb-1"}


def test_sync_dataset_writes_eval_id_as_external_key() -> None:
    client = _FakeClient()
    items = [
        {"id": "sakura-who", "question": "Sakura是谁？", "citations": [{"page": 5904}]},
        {"id": "negative-geography", "question": "巴黎在哪个国家？", "citations": []},
    ]
    name = langsmith_sync.sync_dataset(client, items, "inner-rag-dragon-v1")
    assert name == "inner-rag-dragon-v1"
    kinds = [kind for kind, _ in client.calls]
    assert kinds.count("create_dataset") == 1
    assert kinds.count("create_example") == 2
    # 评测集 id 必须落到 metadata：否则两次实验无法按题对齐
    example = client.calls[2][1]
    assert example["metadata"]["eval_id"] == "sakura-who"
    assert example["metadata"]["anchor_pages"] == [5904]


def test_push_feedback_only_for_traced_items_and_counts() -> None:
    client = _FakeClient()
    results = [
        {
            "id": "sakura-who",
            "trace_id": "trace-1",
            "judge_correct": 1,
            "citation_precision": 0.5,
            "faithfulness": 1.0,
        },
        {"id": "no-trace", "judge_correct": 1, "citation_precision": 1.0},
    ]
    written = langsmith_sync.push_feedback(client, results)
    # 3 个分数各一条 = 3；没有 trace_id 的那条不写（写了也对不上具体调用）
    assert written == 3
    assert all(call[1].get("trace_id") == "trace-1" for call in client.calls)


def test_langsmith_disabled_is_noop_not_exception() -> None:
    """DoD：关闭 LangSmith 时评测必须能完整跑完，这里验证「不抛、不写」。"""
    assert langsmith_sync.sync_dataset(None, [{"id": "x", "question": "q"}], "ds") is None
    assert langsmith_sync.push_feedback(None, [{"id": "x", "trace_id": "t"}]) == 0


def test_push_feedback_swallows_backend_errors() -> None:
    client = _FakeClient(fail_on_feedback=True)
    results = [{"id": "x", "trace_id": "t", "judge_correct": 1}]
    assert langsmith_sync.push_feedback(client, results) == 0


def test_select_pages_keeps_source_page_numbers() -> None:
    """页码必须是源 PDF 的物理页：改成子 PDF 的局部页号会让引用与锚点全对不上。"""
    documents = [
        Document(page_content="a", metadata={"page": 5904}),
        Document(page_content="b", metadata={"page": 6079}),
        Document(page_content="c", metadata={"page": 9000}),
    ]
    selected = build_eval_kb.select_pages(documents, {5904, 6079})
    assert [doc.metadata["page"] for doc in selected] == [5904, 6079]
    with pytest.raises(ValueError, match="一页都没命中"):
        build_eval_kb.select_pages(documents, {1})
