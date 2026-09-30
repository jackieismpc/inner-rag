"""基准测试指标：纯函数实现，不依赖 inner_rag，便于单测与复用。

判定口径：

* **检索侧**以「页区间」为单位：命中 = 召回分块的页区间 ``(page_start, page_end)`` 与期望锚点页
  有交集。当前切分不跨页，区间退化为单点，但判定逻辑按区间写（见 ``base.page_span``）；
* **回答侧**以关键词、拒答行为与引用页码为单位；LLM-as-judge 正确性与忠实度在 Phase 6 追加。

所有指标都是「越大越好」，取值 [0, 1]（毫秒类指标除外）。
"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import mean

# 判断「模型是否在拒答」的标记词（与 rag.py 的兜底文案保持一致）
REFUSAL_MARKERS: tuple[str, ...] = (
    "未找到",
    "没有找到",
    "没有相关",
    "无相关",
    "未提及",
    "不包含",
    "无法回答",
    "无法确定",
    "没有提及",
)


def percentile(values: Sequence[float], p: float) -> float:
    """线性插值分位数（p 取 0~1）；空列表返回 0.0。"""
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def _span_covers(span: tuple[int, int], page: int) -> bool:
    """闭区间 [start, end] 是否覆盖该页。"""
    return span[0] <= page <= span[1]


def spans_hit(retrieved_spans: Sequence[tuple[int, int]], expected_pages: Sequence[int]) -> bool:
    """期望锚点页是否落在任一召回分块的页区间里（Recall 的逐题判定）。"""
    if not expected_pages:
        return False
    wanted = set(expected_pages)
    return any(_span_covers(span, page) for span in retrieved_spans for page in wanted)


def spans_reciprocal_rank(
    retrieved_spans: Sequence[tuple[int, int]], expected_pages: Sequence[int]
) -> float:
    """第一条命中结果的倒数排名；未命中为 0。"""
    wanted = set(expected_pages)
    if not wanted:
        return 0.0
    for index, span in enumerate(retrieved_spans):
        if any(_span_covers(span, page) for page in wanted):
            return 1.0 / (index + 1)
    return 0.0


def spans_page_hit_rate(
    retrieved_spans: Sequence[tuple[int, int]], expected_pages: Sequence[int]
) -> float:
    """命中的期望页数 / 期望页总数（多锚点题反映证据是否被拆散）。"""
    if not expected_pages:
        return 0.0
    wanted = set(expected_pages)
    covered = {page for span in retrieved_spans for page in wanted if _span_covers(span, page)}
    return len(covered) / len(wanted)


def spans_from_pages(pages: Sequence[int]) -> list[tuple[int, int]]:
    """只有单点页号（旧结果 / 旧索引）时的兼容转换：每个页号退化成单点区间。"""
    return [(int(page), int(page)) for page in pages]


def pages_hit(retrieved_pages: Sequence[int], expected_pages: Sequence[int]) -> bool:
    """期望锚点页是否出现在召回页码里（单点口径，等价于区间退化的 ``spans_hit``）。"""
    return spans_hit(spans_from_pages(retrieved_pages), expected_pages)


def reciprocal_rank(retrieved_pages: Sequence[int], expected_pages: Sequence[int]) -> float:
    """单点口径的 MRR。"""
    return spans_reciprocal_rank(spans_from_pages(retrieved_pages), expected_pages)


def page_hit_rate(retrieved_pages: Sequence[int], expected_pages: Sequence[int]) -> float:
    """单点口径的页命中率。"""
    return spans_page_hit_rate(spans_from_pages(retrieved_pages), expected_pages)


def keyword_coverage(answer: str, keywords: Sequence[str]) -> float:
    """关键词命中率；无关键词时视为 1.0（例如负样本）。"""
    if not keywords:
        return 1.0
    return sum(1 for kw in keywords if kw in answer) / len(keywords)


def has_forbidden(answer: str, forbidden: Sequence[str]) -> bool:
    """回答里是否出现「明确错误」的关键词。"""
    return any(word in answer for word in forbidden)


def is_refusal(answer: str) -> bool:
    """回答是否表达了「知识库里没有相关信息」。"""
    return any(marker in answer for marker in REFUSAL_MARKERS)


def spans_citation_precision(
    cited_spans: Sequence[tuple[int, int]], expected_pages: Sequence[int]
) -> float:
    """引用精度：命中期望页的引用区间数 / 引用的区间总数；没有引用则为 0。

    分母用「引用条数」而不是「引用页数」：多引无关页会立刻拉低精度，
    与「少引但准」区分开（引用展示走 sources，见 docs/evaluation.md 4.2）。
    """
    spans = list(dict.fromkeys(cited_spans))  # 同一区间被多次引用只算一次
    if not spans:
        return 0.0
    wanted = set(expected_pages)
    if not wanted:
        return 0.0
    hits = sum(1 for span in spans if any(_span_covers(span, page) for page in wanted))
    return hits / len(spans)


def citation_precision(cited_pages: Sequence[int], expected_pages: Sequence[int]) -> float:
    """单点口径的引用精度。"""
    return spans_citation_precision(spans_from_pages(cited_pages), expected_pages)


def spans_citation_hit(
    cited_spans: Sequence[tuple[int, int]], expected_pages: Sequence[int]
) -> bool:
    """引用命中：引用的来源里**至少有一条**覆盖期望页。

    为什么除了精度还要有这个：精度的分母是「本题引用了多少条来源」，而引用的条数在不同
    配置下会变——一道题一条都没召回时精度记 0.0（不是 0/5），只召回一条且恰好对时精度是
    1/1 = 100%。实测改动前后引用条数从平均 4.1 条变成 5.0 条，两个精度值因此**不可比**
    （见 docs/evaluation.md 4.4）。引用命中是「这道题有没有引对」的 0/1 判定，与 Recall@k
    同口径，可以跨配置比较，也是 DoD 里「引用命中率 ≥ 0.8」对应的量。
    """
    wanted = set(expected_pages)
    if not wanted:
        return False
    spans = list(dict.fromkeys(cited_spans))
    return any(any(_span_covers(span, page) for page in wanted) for span in spans)


def citation_hit(cited_pages: Sequence[int], expected_pages: Sequence[int]) -> bool:
    """单点口径的引用命中。"""
    return spans_citation_hit(spans_from_pages(cited_pages), expected_pages)


def _mean(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def threshold_curve(results: Sequence[dict], thresholds: Sequence[float]) -> list[dict]:
    """从「未过滤的一次召回」推导各阈值下的检索指标。

    为什么能这么做：阈值过滤发生在 Top-k **之后**（`finalize_results` 先滤后排），
    所以一次 `threshold=0` 的召回就包含全部信息——对任一阈值 t，只需丢掉 score < t 的条目
    再重算命中判定，结果与真的按 t 检索完全一致，不必为每个候选阈值各跑一遍（省时也省钱）。

    前置条件：结果里要有 `scores` 与 `expected_pages`（run_bench 会写）；
    老结果 JSON 缺字段时返回空表，而不是算出一堆 0 分。
    """
    usable = [r for r in results if r.get("scores") and r.get("expected_pages") is not None]
    if not usable:
        return []

    curve: list[dict] = []
    for threshold in thresholds:
        recall: list[float] = []
        rr: list[float] = []
        page_hit: list[float] = []
        empty_items = 0
        kept_total = 0

        for result in usable:
            spans = [
                span
                for span, score in zip(result["retrieved_spans"], result["scores"], strict=False)
                if score is None or score >= threshold
            ]
            kept_total += len(spans)
            if not spans:
                empty_items += 1
            if result["expect_refusal"]:
                continue
            pages = expected_pages_of(result)
            recall.append(1.0 if spans_hit(spans, pages) else 0.0)
            rr.append(spans_reciprocal_rank(spans, pages))
            page_hit.append(spans_page_hit_rate(spans, pages))

        curve.append(
            {
                "threshold": round(float(threshold), 4),
                "recall_at_k": _mean(recall),
                "mrr": _mean(rr),
                "page_hit_rate": _mean(page_hit),
                "empty_items": empty_items,
                "kept_avg": round(kept_total / len(usable), 2),
            }
        )
    return curve


def expected_pages_of(result: dict) -> list[int]:
    """从逐题结果里取期望锚点页（`run_bench` 写入的 `expected_pages`）。缺失返回空列表。"""
    return list(result.get("expected_pages") or [])


def summarize(results: Sequence[dict]) -> dict:
    """把逐题结果聚合成一行可比较的指标（缺失的维度返回 None）。"""
    positives = [r for r in results if not r["expect_refusal"]]
    negatives = [r for r in results if r["expect_refusal"]]

    retrieval_ms = [r["retrieval_ms"] for r in results if r.get("retrieval_ms") is not None]
    total_ms = [r["total_ms"] for r in results if r.get("total_ms") is not None]

    keywords = [r["keyword_coverage"] for r in positives if r.get("keyword_coverage") is not None]
    citations = [
        r["citation_precision"] for r in positives if r.get("citation_precision") is not None
    ]
    citation_hits = [r["citation_hit"] for r in positives if r.get("citation_hit") is not None]
    forbidden = [r["forbidden_hit"] for r in results if r.get("forbidden_hit") is not None]
    refusals = [r["refusal"] for r in negatives if r.get("refusal") is not None]
    # 误拒率：该答却拒答（正样本被判为拒答的比例），与拒答正确率一起看才有意义
    false_refusals = [r["refusal"] for r in positives if r.get("refusal") is not None]
    judges = [r["judge_correct"] for r in positives if r.get("judge_correct") is not None]
    faithful = [r["faithfulness"] for r in positives if r.get("faithfulness") is not None]
    prompt_tokens = [r["prompt_tokens"] for r in results if r.get("prompt_tokens") is not None]
    completion_tokens = [
        r["completion_tokens"] for r in results if r.get("completion_tokens") is not None
    ]
    costs = [r["cost_usd"] for r in results if r.get("cost_usd") is not None]

    return {
        "items": len(results),
        "positives": len(positives),
        "negatives": len(negatives),
        "recall_at_k": _mean([1.0 if r["hit"] else 0.0 for r in positives]),
        "mrr": _mean([r["rr"] for r in positives]),
        "page_hit_rate": _mean([r["page_hit"] for r in positives]),
        "keyword_pass_rate": _mean([1.0 if (k or 0) >= 1.0 else 0.0 for k in keywords])
        if keywords
        else None,
        "keyword_coverage": _mean(keywords) if keywords else None,
        "forbidden_rate": _mean([1.0 if f else 0.0 for f in forbidden]) if forbidden else None,
        "citation_precision": _mean(citations) if citations else None,
        # 与 citation_precision 同源但口径可比：分母恒为「题数」，不随引用条数漂移
        "citation_hit_rate": _mean([1.0 if h else 0.0 for h in citation_hits])
        if citation_hits
        else None,
        "refusal_accuracy": _mean([1.0 if r else 0.0 for r in refusals]) if refusals else None,
        "false_refusal_rate": _mean([1.0 if r else 0.0 for r in false_refusals])
        if false_refusals
        else None,
        "judge_accuracy": _mean([1.0 if j else 0.0 for j in judges]) if judges else None,
        "faithfulness": _mean(faithful) if faithful else None,
        "prompt_tokens": sum(prompt_tokens) if prompt_tokens else None,
        "completion_tokens": sum(completion_tokens) if completion_tokens else None,
        "cost_usd": round(sum(costs), 6) if costs else None,
        "retrieval_p50_ms": percentile(retrieval_ms, 0.5),
        "retrieval_p95_ms": percentile(retrieval_ms, 0.95),
        "total_p50_ms": percentile(total_ms, 0.5) if total_ms else None,
        "total_p95_ms": percentile(total_ms, 0.95) if total_ms else None,
    }
