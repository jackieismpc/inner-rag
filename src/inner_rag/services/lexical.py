"""词面检索（BM25）：补稠密检索的短板——专名与生僻词。

为什么需要它：Phase 6/8 的小库实测里，失败的题都是同一个形状——问题里的关键词在语料里
**字面就存在**，但向量召回完全没把它捞进候选集（答案页不在 Top-8 里）：

* 「诺诺的真名是什么？」→ 目标页写着「陈墨瞳」，embedding（512 token 免费模型）没能把这两个
  绰号与真名拉到一起；Top-8 全是 score 0.16–0.29 的无关页；
* 「上杉绘梨衣的言灵是什么？」→ 目标页同时含「绘梨衣」与「言灵」，同样没被召回。

两题都是**词面可匹配**的，所以这不是「需要更好的 embedding」，而是「缺一路稀疏检索」。
向量负责语义泛化，词面负责专名与精确措辞，两者融合（`HYBRID_SPARSE_WEIGHT`）才是完整的召回。

分词为什么是字 bigram 而不是分词器：中文分词要么引第三方词典（重、且对小说专名反而切错），
要么自己维护词表。字 bigram（「陈墨瞳」→ 陈墨/墨瞳）对专名足够，零依赖、确定性、可离线测；
英文与数字按词切并小写化。

索引按知识库缓存在进程内，由 `services/vector_store` 的 `iter_chunks` 构建一次；
入库 / 删文档 / 删库时调用 `invalidate(kb_id)` 丢弃（与检索缓存同一时机）。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from langchain_core.documents import Document
from loguru import logger

# BM25 的两个标准参数：k1 控制词频饱和，b 控制长度归一化强度
BM25_K1 = 1.5
BM25_B = 0.75

# 英文 / 数字按词切；中文与其它表意文字按字 bigram。两段互补，不重叠。
_LATIN = re.compile(r"[a-z0-9]+")
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]")

MIN_QUERY_TERMS = 1


def tokenize(text: str) -> list[str]:
    """把文本切成检索用的词元：拉丁词（小写）+ 中日韩字 bigram。

    单字也保留一份：query 是单个汉字（如「龙」）时 bigram 为空，只有单字能匹配。
    """
    lowered = text.lower()
    tokens = _LATIN.findall(lowered)

    characters = _CJK.findall(lowered)
    tokens.extend(characters)
    tokens.extend(characters[index] + characters[index + 1] for index in range(len(characters) - 1))
    return tokens


@dataclass
class _IndexedChunk:
    doc: Document
    # 词元 -> 词频。用 Counter 而不是 list：BM25 要词频，重建时也省一次统计。
    term_frequencies: Counter[str] = field(default_factory=Counter)
    length: int = 0


class _KbIndex:
    """一个知识库的倒排索引（进程内，由 `LexicalIndex` 管理生命周期）。"""

    def __init__(self, chunks: list[Document]) -> None:
        self._chunks = [
            _IndexedChunk(doc=doc, term_frequencies=Counter(tokenize(doc.page_content)), length=0)
            for doc in chunks
        ]
        self._postings: dict[str, list[int]] = {}
        for position, chunk in enumerate(self._chunks):
            chunk.length = sum(chunk.term_frequencies.values())
            for term in chunk.term_frequencies:
                self._postings.setdefault(term, []).append(position)
        self._avg_length = (
            sum(chunk.length for chunk in self._chunks) / len(self._chunks) if self._chunks else 0.0
        )

    def __len__(self) -> int:
        return len(self._chunks)

    def search(
        self, query: str, k: int, filter_doc_ids: list[int] | None = None
    ) -> list[tuple[Document, float]]:
        """BM25 打分，返回按分数降序的前 k 条（原始分，未归一化）。"""
        terms = tokenize(query)
        if not terms or not self._chunks:
            return []

        allowed = {str(doc_id) for doc_id in filter_doc_ids} if filter_doc_ids else None
        scores: dict[int, float] = {}
        total = len(self._chunks)
        for term, query_frequency in Counter(terms).items():
            positions = self._postings.get(term)
            if not positions:
                continue
            # IDF：term 越罕见越有区分度（+0.5 平滑，避免文档数少时出现负数）
            idf = math.log(1.0 + (total - len(positions) + 0.5) / (len(positions) + 0.5))
            if idf <= 0.0:
                continue
            for position in positions:
                chunk = self._chunks[position]
                if allowed is not None and str(chunk.doc.metadata.get("doc_id")) not in allowed:
                    continue
                frequency = chunk.term_frequencies[term]
                length_ratio = chunk.length / self._avg_length if self._avg_length else 1.0
                saturation = (
                    frequency
                    * (BM25_K1 + 1)
                    / (frequency + BM25_K1 * (1 - BM25_B + BM25_B * length_ratio))
                )
                # query 里同一个词出现两次，说明它更重要
                scores[position] = scores.get(position, 0.0) + idf * saturation * query_frequency

        if not scores:
            return []
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [(self._chunks[position].doc, score) for position, score in ranked[:k]]


class LexicalIndex:
    """按知识库缓存的词面索引。

    缓存的必要性：建索引要把整库分块读出来（zvec 只能逐条扫），每次查询都重建等于把
    扫描成本放进检索热路径；索引只读、且由 `invalidate` 在入库 / 删除时丢弃，
    因此不存在「读到旧内容」的窗口。
    """

    def __init__(self) -> None:
        self._indexes: dict[int, _KbIndex] = {}

    def invalidate(self, kb_id: int | None = None) -> None:
        """丢弃某库（或全部）的索引：内容变了就必须重建。"""
        if kb_id is None:
            self._indexes.clear()
        else:
            self._indexes.pop(kb_id, None)

    def _get(self, kb_id: int) -> _KbIndex:
        cached = self._indexes.get(kb_id)
        if cached is not None:
            return cached

        from inner_rag.services.vector_store import vector_service

        chunks = vector_service.iter_chunks(kb_id)
        index = _KbIndex(chunks)
        self._indexes[kb_id] = index
        logger.debug(f"[LEXICAL] 建索引 kb={kb_id} chunks={len(index)}")
        return index

    def search(
        self, kb_id: int, query: str, k: int, filter_doc_ids: list[int] | None = None
    ) -> list[tuple[Document, float]]:
        """词面检索，返回 (分块, BM25 原始分) 降序。库为空时返回空列表。"""
        if len(tokenize(query)) < MIN_QUERY_TERMS:
            return []
        return self._get(kb_id).search(query, k, filter_doc_ids)

    def size(self, kb_id: int) -> int | None:
        """已缓存的索引规模（未建索引返回 None）——用于诊断与测试。"""
        index = self._indexes.get(kb_id)
        return len(index) if index is not None else None


lexical_index = LexicalIndex()


def normalize_scores(hits: list[tuple[Document, float]]) -> list[tuple[Document, float]]:
    """把 BM25 原始分线性归一到 ``[0, 1]``（除以本次最高分）。

    为什么必须归一：融合后的分数要继续参与 `RETRIEVAL_SCORE_THRESHOLD` 过滤，而 BM25 的分值
    是无上界的（取决于语料长度与 IDF），直接与 ``[0, 1]`` 的向量相关度混在一起，
    阈值就失去意义了。归一后「本次词面最匹配的一条 = 1.0」，语义与向量侧一致。
    """
    if not hits:
        return []
    best = max(score for _doc, score in hits)
    if best <= 0.0:
        return [(doc, 0.0) for doc, _ in hits]
    return [(doc, score / best) for doc, score in hits]
