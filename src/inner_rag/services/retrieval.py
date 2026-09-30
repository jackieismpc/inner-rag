"""检索组合层：把「向量召回」和「词面召回」融合成一路结果。

为什么要有这一层：`vector_service.search` 只回答「向量库怎么查」，而「这次查询到底该用哪几路
召回、怎么合成一个分数」是业务决策，不该压进每个后端适配器里。放进组合层后：

* 后端依旧是「一个向量库」的实现（换向量库不影响召回策略）；
* 问答、基准、探针脚本共用同一条检索路径——评测测的就是线上跑的那条路，
  否则「评测涨了、线上没变」这类偏差永远查不出来。

`strategy` 的三种取值（对外语义）：

* ``similarity``：纯向量。
* ``mmr``：纯向量 + 多样性（MMR）。
* ``hybrid``：向量 ∪ 词面（BM25）。**这是「混合检索」的正解**——向量负责语义泛化，
  词面负责专名与精确措辞；实测里失败的题都是「问题关键词字面就在语料里、向量却没召回」。

融合规则（``HYBRID_SPARSE_WEIGHT``，0 = 退回纯向量 + MMR）：

* **前置门**：两路里至少要有一路拿出实质证据才启用词面——稠密侧有过阈值的候选，
  或词面最强匹配 ≥ ``HYBRID_MIN_SPARSE_SCORE``。词面是补充，不能在「库里没有相关内容」
  时替稠密侧翻案：巧合的字面匹配会把拒答能力直接摧毁（见下方 ``dense_support``）；
* 同一分块取两侧分数的较大者：``max(向量相关度, w × 归一化 BM25)``。
  两路都命中说明它既语义相关又词面匹配，理应有更高的排位；
* BM25 先归一到 ``[0, 1]``（见 `lexical.normalize_scores`），否则它无上界的分值会把
  ``RETRIEVAL_SCORE_THRESHOLD`` 的语义冲掉；代价是**词面第一名恒为 ``w``**，
  只要词面命中一条就必然过阈值——这也是前置门不可省的原因；
* MMR 补充项（``score=None``）按契约不过阈值过滤，融合后仍保持 ``None``。
"""

from __future__ import annotations

from langchain_core.documents import Document

from inner_rag.core.config import settings
from inner_rag.core.observability import tracer
from inner_rag.services.lexical import lexical_index, normalize_scores
from inner_rag.services.vector_store import vector_service
from inner_rag.services.vector_store.base import (
    Strategy,
    chunk_key,
    finalize_results,
    resolve_search_defaults,
)

RetrievalResult = list[tuple[Document, float | None]]


def _sort_key(item: tuple[Document, float | None]) -> tuple[bool, float]:
    """与 `base.finalize_results` 同一口径：有分数的按分数降序，无分数（MMR）排最后。"""
    score = item[1]
    return (score is not None, score if score is not None else 0.0)


def fuse(dense: RetrievalResult, sparse: list[tuple[Document, float]], k: int) -> RetrievalResult:
    """向量结果与词面结果按分块去重融合，取前 k 条（纯函数，便于单测）。"""
    merged: dict[tuple[str | None, str | None], tuple[Document, float | None]] = {}

    for doc, score in dense:
        # 同一分块可能在 dense 里出现两次（相似度 + MMR 补充），保留有分数的那条
        key = chunk_key(doc)
        existing = merged.get(key)
        if existing is None or (existing[1] is None and score is not None):
            merged[key] = (doc, score)

    for doc, score in sparse:
        key = chunk_key(doc)
        existing = merged.get(key)
        if existing is None:
            merged[key] = (doc, score)
        elif existing[1] is None or score > existing[1]:
            # 词面更匹配（或原本没有分数）时以词面分数为准，避免同一个分块留下两条
            merged[key] = (existing[0], max(score, existing[1] or 0.0))

    ranked = sorted(merged.values(), key=_sort_key, reverse=True)
    return ranked[:k]


async def search(
    kb_id: int,
    query: str,
    k: int | None = None,
    strategy: Strategy = "hybrid",
    score_threshold: float | None = None,
    filter_doc_ids: list[int] | None = None,
) -> tuple[RetrievalResult, int]:
    """检索入口：返回 ``(结果, 被阈值滤掉的条数)``，语义与 ``vector_service.search`` 一致。"""
    k, threshold = resolve_search_defaults(k, score_threshold)

    async with tracer.span(
        "vector.search",
        kb_id=kb_id,
        strategy=strategy,
        k=k,
        lexical=bool(strategy == "hybrid" and settings.HYBRID_SPARSE_WEIGHT > 0),
    ) as span:
        dense, filtered_out = await vector_service.search(
            kb_id=kb_id,
            query=query,
            k=k,
            strategy=strategy,
            # 阈值在融合之后统一生效：先按阈值过滤再融合，会把「向量分数低但词面完全匹配」
            # 的候选提前丢掉，而补上这一路正是融合的目的。
            score_threshold=0.0,
            filter_doc_ids=filter_doc_ids,
        )
        span.set(dense_hits=len(dense))

        # 词面这一路是**补充**，不是独立的召回源：两路里至少要有一路拿出实质证据才启用。
        #
        # 为什么必须加这道门：BM25 是字面匹配，分不清同形不同义。负样本题「巴黎在哪个国家？」
        # 因为正文里有一款香槟叫「巴黎之花美丽时光」而被命中，而归一化让词面第一名恒为
        # 1.0 × 权重——这条巧合匹配必然过阈值、进了 context，模型随即放弃拒答
        # （实测拒答正确率 100% → 0%）。
        #
        # 判据是「或」：稠密侧有过阈值的候选 **或** 词面最强匹配够强。两者都不可用时，
        # 最合理的解释是库里确实没有相关内容，此时不该由词面单方面放行。
        #
        # 为什么按「词面最强匹配」而不是「目标页的分数」设门槛：恰好被救回来的那道题
        # （「诺诺的真名」）目标页只有 14.0 分，而巧合匹配（香槟品牌）有 13.6 分——按页分
        # 几乎分不开；但**整个 query 的最强匹配**是 24.8 vs 13.6，差 1.8 倍，这才是稳的判据。
        # 稠密侧在这两题上都一条没过阈值（所以单靠稠密侧也分不开，见 docs/evaluation.md 4.5）。
        dense_support = any(score is not None and score >= threshold for _doc, score in dense)
        span.set(dense_support=dense_support)

        sparse: list[tuple[Document, float]] = []
        if strategy == "hybrid" and settings.HYBRID_SPARSE_WEIGHT > 0:
            async with tracer.span("lexical.search", kb_id=kb_id) as lexical_span:
                hits = lexical_index.search(kb_id, query, k, filter_doc_ids)
                strongest = hits[0][1] if hits else 0.0
                enabled = dense_support or strongest >= settings.HYBRID_MIN_SPARSE_SCORE
                lexical_span.set(strongest=round(strongest, 3), enabled=enabled)
                if enabled:
                    sparse = [
                        (doc, score * settings.HYBRID_SPARSE_WEIGHT)
                        for doc, score in normalize_scores(hits)
                    ]
            lexical_span.set(hits=len(sparse))

        if not sparse:
            results, filtered_out = finalize_results(dense, threshold)
            span.set(hits=len(results), filtered_out=filtered_out, sparse_hits=0)
            return results, filtered_out

        # dense 侧的 filtered_out 在融合后不再有意义：融合可能把被滤掉的分块重新带回，
        # 因此过滤计数一律以融合后的结果为准（重跑一次 finalize_results 即得）。
        fused = fuse(dense, sparse, k)
        results, filtered_out = finalize_results(fused, threshold)
        span.set(hits=len(results), filtered_out=filtered_out, sparse_hits=len(sparse))
        return results, filtered_out
