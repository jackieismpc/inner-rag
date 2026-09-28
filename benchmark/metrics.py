"""基准测试指标：纯函数实现，不依赖 inner_rag，便于单测与复用。

判定口径：

* **检索侧**以「页码」为单位：命中 = 召回分块的页码集合与期望锚点页集合有交集。
* **回答侧**以关键词、拒答行为与引用页码为单位；LLM-as-judge 与忠实度在 Phase 4 追加。

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


def pages_hit(retrieved_pages: Sequence[int], expected_pages: Sequence[int]) -> bool:
    """期望锚点页是否出现在召回页码里（Recall 的逐题判定）。"""
    if not expected_pages:
        return False
    return bool(set(retrieved_pages) & set(expected_pages))


def reciprocal_rank(retrieved_pages: Sequence[int], expected_pages: Sequence[int]) -> float:
    """第一条命中结果的倒数排名；未命中为 0。"""
    wanted = set(expected_pages)
    if not wanted:
        return 0.0
    for index, page in enumerate(retrieved_pages):
        if page in wanted:
            return 1.0 / (index + 1)
    return 0.0


def page_hit_rate(retrieved_pages: Sequence[int], expected_pages: Sequence[int]) -> float:
    """命中的期望页数 / 期望页总数（多锚点题反映证据是否被拆散）。"""
    if not expected_pages:
        return 0.0
    return len(set(retrieved_pages) & set(expected_pages)) / len(set(expected_pages))


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


def citation_precision(cited_pages: Sequence[int], expected_pages: Sequence[int]) -> float:
    """引用精度：引用到的期望页 / 引用的全部页；没有引用则为 0。"""
    if not cited_pages:
        return 0.0
    if not expected_pages:
        return 0.0
    hits = len(set(cited_pages) & set(expected_pages))
    return hits / len(set(cited_pages))


def _mean(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


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
    forbidden = [r["forbidden_hit"] for r in results if r.get("forbidden_hit") is not None]
    refusals = [r["refusal"] for r in negatives if r.get("refusal") is not None]

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
        "refusal_accuracy": _mean([1.0 if r else 0.0 for r in refusals]) if refusals else None,
        "retrieval_p50_ms": percentile(retrieval_ms, 0.5),
        "retrieval_p95_ms": percentile(retrieval_ms, 0.95),
        "total_p50_ms": percentile(total_ms, 0.5) if total_ms else None,
        "total_p95_ms": percentile(total_ms, 0.95) if total_ms else None,
    }
