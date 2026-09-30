"""向量库适配器的共同契约与共用实现。

对外语义只在这一处定义，所有后端（zvec / chroma）共用：

* **相关度口径**：对外一律是 ``[0, 1]`` 的余弦相关度，越大越相关（cosine 距离 ``d`` → 相关度
  ``1 - d``）。两个后端都自己换算，因为 Chroma 的 ``similarity_search_with_relevance_scores``
  名字里带 relevance、返回的却是没转换过的原始距离，直接用会出现「相关度 0% / 100%」这种错误展示；
* **MMR 无分数**：``strategy="mmr"`` 的条目 ``score=None``，不参与阈值过滤，排序时排在有分数之后；
* **阈值过滤计数**：被 ``score_threshold`` 滤掉的条数要返回，供「空召回率」这类指标使用；
* **元数据是标量**：``doc_id`` / ``kb_id`` / ``chunk_index`` 存字符串、``page`` 与页区间存整数；
  解析器额外附带的 ``source`` / ``sheet`` / ``ocr`` 不写进向量库——下游只消费
  ``filename`` / ``page`` / ``page_start`` / ``page_end`` / ``doc_id`` / ``chunk_index``
  （见 ``services/rag.py`` 与 ``chunk_key``）；
* **页码是区间**：除单页 ``page`` 外还写 ``page_start`` / ``page_end``（闭区间，1-based 物理页）。
  当前解析器逐页产出 ``Document``、切分也不跨页（实测：龙族 PDF 平均 211 字符/页 < ``CHUNK_SIZE``），
  所以区间**当前恒等**；保留区间是为了让引用核对与评测命中判定按区间写，
  将来调整分块策略（合并相邻页）时只改 ``page_span`` 一处即可，不必再改契约。

新增后端时：写入前调用 ``prepare_chunks``、返回前调用 ``finalize_results``、
检索默认值走 ``resolve_search_defaults``，即可与现有后端保持同一份语义。
"""

from __future__ import annotations

from typing import Literal, Protocol

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from inner_rag.core.config import settings

Strategy = Literal["similarity", "mmr", "hybrid"]

# 写入分批大小：控制单次 native 调用的内存占用与超时粒度
WRITE_BATCH_SIZE = 50

# 写入向量库的分块元数据白名单。解析器还会附带 source / sheet / ocr 等字段，
# 但下游（context 组装、混合检索去重）只消费这些；向量库 schema 也只声明这些字段。
CHUNK_METADATA_FIELDS = (
    "doc_id",
    "kb_id",
    "filename",
    "chunk_index",
    "page",
    "page_start",
    "page_end",
)

# MMR 的相关性 / 多样性权重，与 LangChain 默认一致
MMR_LAMBDA = 0.5
# MMR 候选池倍数：先多召回再挑，否则多样性没有选择空间
MMR_FETCH_FACTOR = 3


def distance_to_relevance(distance: float) -> float:
    """cosine 距离 -> [0, 1] 相关度。"""
    return max(0.0, min(1.0, 1.0 - float(distance)))


def resolve_search_defaults(k: int | None, score_threshold: float | None) -> tuple[int, float]:
    """检索默认值只解析一处：``k`` 默认 ``TOP_K``，阈值默认 ``RETRIEVAL_SCORE_THRESHOLD``。"""
    if k is None:
        k = settings.TOP_K
    if score_threshold is None:
        score_threshold = settings.RETRIEVAL_SCORE_THRESHOLD
    return k, score_threshold


def finalize_results(
    raw: list[tuple[Document, float | None]], threshold: float
) -> tuple[list[tuple[Document, float | None]], int]:
    """阈值过滤 + 相关度降序，返回 (结果, 被阈值滤掉的条数)。

    无分数的条目（MMR 单独召回的分块）不做阈值判断，排序时排在有分数的之后。
    """
    filtered: list[tuple[Document, float | None]] = []
    filtered_out = 0
    for doc, score in raw:
        if score is not None and score < threshold:
            filtered_out += 1
            logger.debug(
                f"[FILTER] relevance={score:.4f} < {threshold} "
                f"file={doc.metadata.get('filename', '?')} "
                f"chunk={doc.metadata.get('chunk_index', '?')}"
            )
            continue
        filtered.append((doc, score))

    filtered.sort(
        key=lambda item: (item[1] is not None, item[1] if item[1] is not None else 0.0),
        reverse=True,
    )
    return filtered, filtered_out


def chunk_key(doc: Document) -> tuple[str | None, str | None]:
    """分块唯一标识，用于混合检索去重（比 id(doc) 可靠）。"""
    return doc.metadata.get("doc_id"), doc.metadata.get("chunk_index")


def page_span(doc: Document) -> tuple[int, int] | None:
    """分块覆盖的页区间 ``(start, end)``：闭区间、1-based 物理页；没有页码信息返回 None。

    区间优先取上游已经写好的 ``page_start`` / ``page_end``，缺失时退回单页 ``page``
    （旧索引与「解析器逐页产出」的当前实现都走这一支）。
    """
    metadata = doc.metadata
    start = metadata.get("page_start")
    end = metadata.get("page_end")
    if start is None or end is None:
        page = metadata.get("page")
        if page is None:
            return None
        try:
            page_int = int(page)
        except (TypeError, ValueError):
            return None
        return page_int, page_int
    return int(start), int(end)


def prepare_chunks(
    documents: list[Document], kb_id: int, doc_id: int, filename: str
) -> list[Document]:
    """切分文档并写入分块元数据，返回可入库的分块（空分块已剔除）。

    元数据是**白名单式重建**：解析器的原文元数据（``source`` 路径、``sheet``、``ocr`` 标记）
    不进向量库，只保留 ``CHUNK_METADATA_FIELDS``，保证两个后端的 schema 与返回值一致。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""],
        length_function=len,
    )
    chunks = [chunk for chunk in splitter.split_documents(documents) if chunk.page_content.strip()]
    for index, chunk in enumerate(chunks):
        metadata: dict[str, str | int] = {
            "doc_id": str(doc_id),
            "kb_id": str(kb_id),
            "filename": filename,
            "chunk_index": str(index),
        }
        span = page_span(chunk)
        if span is not None:
            # page 保留单点：旧索引与现有消费方（前端展示、chunk_key）都在用它
            metadata["page"] = span[0]
            metadata["page_start"] = span[0]
            metadata["page_end"] = span[1]
        chunk.metadata = metadata
    return chunks


class VectorStore(Protocol):
    """向量库契约：业务层只依赖这组方法，后端差异（zvec / Chroma）不泄漏到上层。

    实现者必须遵守 ``prepare_chunks`` / ``finalize_results`` 的语义（见模块 docstring）。
    """

    async def add_documents(
        self, kb_id: int, documents: list[Document], doc_id: int, filename: str
    ) -> int:
        """分块 + 嵌入 + 写入，返回写入的分块数（没有有效分块时返回 0）。"""
        ...

    async def search(
        self,
        kb_id: int,
        query: str,
        k: int | None = None,
        strategy: Strategy = "similarity",
        score_threshold: float | None = None,
        filter_doc_ids: list[int] | None = None,
    ) -> tuple[list[tuple[Document, float | None]], int]:
        """检索，返回 (结果列表, 被阈值滤掉的条数)；元素为 (分块, 相关度 | None)。"""
        ...

    async def delete_document(self, kb_id: int, doc_id: int) -> int:
        """删除某文档的全部分块，返回删除条数。"""
        ...

    async def delete_kb(self, kb_id: int) -> None:
        """删除整个知识库的向量（幂等：从未入库过的知识库也算成功）。"""
        ...

    def count(self, kb_id: int) -> int:
        """知识库当前的分块数（用于出参与对账；未知的知识库返回 0）。"""
        ...

    def count_chunks_by_filename(self, kb_id: int) -> dict[str, int]:
        """各文件的分块数（排查索引用）。"""
        ...

    def list_doc_ids(self, kb_id: int) -> list[str]:
        """库里出现过的 doc_id（用于发现删除后的残留向量）。"""
        ...
