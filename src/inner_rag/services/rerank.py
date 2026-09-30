"""Rerank 插件点：对粗排结果做最后一轮精排。

为什么要有这一层：`services/retrieval.py` 的融合分只有一个维度（向量相关度 ∪ 归一化 BM25），
它决定「谁进 Top-k」；但**谁进 context** 是另一件事——`RERANK_TOP_K`（默认 5）小于
`TOP_K`（默认 8），所以 Top-k 里排在 6–8 位的候选会被整条丢掉。

小库实测（kb=1，hybrid，k=8）显示这一段恰好是失分点：三道题的期望页卡在 6–8 位，
且它们与第 5 名的分差只有 **0.003–0.04**（见 `docs/evaluation.md` 4.6）——粗排在这么窄的
区间里排序本来就不可靠。粗排分数融合的是「两条召回通道的置信度」，而精排可以直接看
「候选与查询的贴合程度」，这是两种不同的证据。

契约（每个实现都要满足）：

* **只改顺序，不得增删，也不得改分**。分数代表粗排相关度，既用于 `RETRIEVAL_SCORE_THRESHOLD`
  判定，也用于前端展示；让 rerank 改分会让「同一个阈值」在不同配置下含义不同，
  两次评测也就无法对照（本项目的评测纪律见 `docs/evaluation.md` 第 7 节）；
* **失败必须退回原顺序**：抛异常、解析不出、模型返回垃圾时一律原样返回。
  精排是「有则更好」的一层，不能因为它把结果弄丢；
* 默认 ``none``（不启用）。rerank 必然改变排序，必须先在评测集上证明有提升再打开
  （回归 >2pp 不允许合入）。

已知边界：内置实现只看**本次查询的候选集**，不引入外部模型；需要真正的 cross-encoder 时，
注册一个 entry point 即可（组名 ``inner_rag.rerankers``，见 `docs/architecture.md` 第 5 节）。
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from typing import Protocol

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from loguru import logger

from inner_rag.core.config import settings
from inner_rag.plugins.registry import rerankers
from inner_rag.services.lexical import tokenize

RetrievalResult = list[tuple[Document, float | None]]

# 关掉精排的实现名。写成一个常量，是为了让「是否启用」只在一处判断（见 `is_enabled`）。
NOOP_NAME = "none"

_NUMBER = re.compile(r"\d+")


class Reranker(Protocol):
    """精排契约：输入已按粗排分数降序的候选，输出同样条数、同样分数的另一个顺序。"""

    name: str

    async def rerank(self, kb_id: int, query: str, results: RetrievalResult) -> RetrievalResult: ...


class NoopReranker:
    """默认实现：不精排（保持粗排顺序）。"""

    name = NOOP_NAME

    async def rerank(self, kb_id: int, query: str, results: RetrievalResult) -> RetrievalResult:
        return results


def candidate_idf(query_terms: set[str], document_terms: list[set[str]]) -> dict[str, float]:
    """在**候选集内**估计 IDF：只在少数候选里出现的查询词更有区分度。

    为什么不复用全库 IDF：BM25 那一档衡量的量的是「这条分块对整库有多独特」，
    但精排面对的问题是「这条分块相对**同批竞争者**有多贴合查询」。一个查询词如果在
    8 条候选里人人都有，它对排序毫无信息量，可全库 IDF 反而可能很高（比如全库只有
    20 页提到它）。候选集内估计正好把这种「本批无区分度」的词压到接近 0。
    """
    size = len(document_terms)
    weights: dict[str, float] = {}
    for term in query_terms:
        document_frequency = sum(1 for terms in document_terms if term in terms)
        weights[term] = math.log(
            1.0 + (size - document_frequency + 0.5) / (document_frequency + 0.5)
        )
    return weights


class LexicalReranker:
    """按「查询词元在候选集内的稀有度加权覆盖率」重排。

    与词面召回（BM25）是两条不同的证据：BM25 比的是**分块之间的绝对匹配强度**
    （受长度归一与词频饱和影响，长分块天然吃亏），本重排比的是**查询被覆盖了多少**——
    一条把查询里所有实词都覆盖到的短分块，会比只反复出现同一个词的长分块排得更前。
    """

    name = "lexical"

    async def rerank(self, kb_id: int, query: str, results: RetrievalResult) -> RetrievalResult:
        if len(results) <= 1:
            return results
        query_terms = set(tokenize(query))
        if not query_terms:
            return results

        document_terms = [set(tokenize(doc.page_content)) for doc, _ in results]
        weights = candidate_idf(query_terms, document_terms)
        total = sum(weights.values())
        if total <= 0.0:
            return results

        scored = [
            (index, sum(weights[term] for term in query_terms & terms) / total)
            for index, terms in enumerate(document_terms)
        ]
        # 稳定排序：覆盖率相同时保持粗排顺序，避免精排把「本来就对」的题打乱
        order = sorted(scored, key=lambda item: (-item[1], item[0]))
        return [results[index] for index, _ in order]


class LLMReranker:
    """provider 侧 listwise 精排：把候选编号交给 chat 模型，要它返回排序后的编号序列。

    模型输出不可靠——可能漏号、重复、给不存在的编号、夹带解释文字。因此**必须解析 + 兜底**：
    正则抽数字 → 去重 → 丢弃越界 → 缺的按原顺序补到尾部，保证输出集合与输入完全一致
    （契约要求不得增删）。任何异常一律退回粗排顺序。
    """

    name = "llm"

    def _get_llm(self) -> BaseChatModel:
        # 延迟 import：providers 与 plugins 之间本就有惰性注册的约定（见 plugins/registry.py）
        from inner_rag.providers import get_chat_model

        return get_chat_model()

    async def rerank(self, kb_id: int, query: str, results: RetrievalResult) -> RetrievalResult:
        top_n = max(2, settings.RERANK_LLM_TOP_N)
        candidates = results[:top_n]
        tail = results[top_n:]
        if len(candidates) <= 1:
            return results

        try:
            order = await self._ask(query, candidates)
        except Exception as exc:
            logger.warning(f"[RERANK] llm 精排失败，退回粗排顺序: {exc}")
            return results
        if not order:
            return results

        chosen = set(order)
        reordered = [candidates[index] for index in order]
        reordered.extend(
            candidates[index] for index in range(len(candidates)) if index not in chosen
        )
        return reordered + tail

    async def _ask(self, query: str, candidates: RetrievalResult) -> list[int]:
        snippet = settings.RERANK_LLM_SNIPPET_CHARS
        listing = "\n".join(
            f"[{index + 1}] {doc.page_content[:snippet]}"
            for index, (doc, _score) in enumerate(candidates)
        )
        prompt = (
            "你是检索结果重排器。给定用户问题与若干候选片段，"
            "按与问题的相关程度从高到低给出片段编号。\n"
            f"只输出编号，用英文逗号分隔，必须包含全部 {len(candidates)} 个编号，"
            "每个编号只出现一次，不要输出任何解释。\n\n"
            f"问题：{query}\n\n候选片段：\n{listing}"
        )
        message = await self._get_llm().ainvoke(prompt)
        return parse_order(str(message.content), len(candidates))


def parse_order(raw: str, size: int) -> list[int]:
    """从模型输出里抽出 0-based 排序下标：去重、丢弃越界，无法解析时返回空列表。"""
    order: list[int] = []
    seen: set[int] = set()
    for token in _NUMBER.findall(raw):
        index = int(token) - 1
        if 0 <= index < size and index not in seen:
            seen.add(index)
            order.append(index)
    return order


rerankers.register(NOOP_NAME, NoopReranker)
rerankers.register("lexical", LexicalReranker)
rerankers.register("llm", LLMReranker)


def build_reranker(name: str | None = None) -> Reranker:
    """按名字（默认取 ``RERANK_BACKEND``）构造精排器；未知名字抛可读错误。

    不做静默降级：把「排序被谁改过」悄悄换掉，会让评测数字无法解释
    （与 ``build_vector_store`` / ``build_cache_backend`` 同一原则）。
    """
    rerankers.load_entry_points()
    backend = (name or settings.RERANK_BACKEND).strip().lower()
    factory: Callable[[], Reranker] | None = rerankers.get(backend)
    if factory is None:
        msg = f"不支持的 RERANK_BACKEND: {backend!r}（可选 {'/'.join(rerankers.names())}）"
        raise ValueError(msg)
    return factory()


reranker = build_reranker()


def is_enabled() -> bool:
    """``none`` 视为未启用（默认）；其余实现都会真的改变排序。"""
    return reranker.name != NOOP_NAME
