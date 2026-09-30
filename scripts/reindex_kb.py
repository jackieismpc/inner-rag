#!/usr/bin/env python
"""重建整个知识库的向量索引。

用法:
    uv run scripts/reindex_kb.py <kb_id> [--keep-vectors]

换过 embedding 模型后必须重建，否则向量空间与查询向量不一致（检索结果会是噪声）。
"""

from __future__ import annotations

import argparse
import asyncio

from loguru import logger

from inner_rag.core.database import SessionLocal
from inner_rag.models import DocStatus
from inner_rag.repositories import Repositories, build_repositories
from inner_rag.services.cache import query_cache
from inner_rag.services.document import DocumentProcessingError, doc_service
from inner_rag.services.vector_store import vector_service


def _reset_all(repos: Repositories, kb_id: int) -> list[int]:
    """把整个库的文档打回 PENDING，返回 doc_id 列表。"""
    return repos.docs.reset_kb_for_reprocess(kb_id)


def _report(repos: Repositories, kb_id: int) -> list[str]:
    """打印每个文档的最终状态，返回失败文件名列表。"""
    rows = repos.docs.list_by_kb(kb_id)
    for row in rows:
        logger.info(f"  {row.status.value:10s} chunks={row.chunk_count:4d}  {row.filename}")
    return [row.filename for row in rows if row.status == DocStatus.FAILED]


async def reindex(kb_id: int, keep_vectors: bool = False) -> int:
    if not keep_vectors:
        await vector_service.delete_kb(kb_id)
        logger.info(f"已清空知识库 {kb_id} 的旧向量")

    with SessionLocal() as db:
        doc_ids = _reset_all(build_repositories(db), kb_id)

    for index, doc_id in enumerate(doc_ids, start=1):
        logger.info(f"[{index}/{len(doc_ids)}] 重建 doc_id={doc_id}")
        try:
            await doc_service.process_document(doc_id)
        except DocumentProcessingError as exc:
            # 单个文档失败不该中断整库重建（失败原因已写进文档状态），跑完再统一汇报
            logger.warning(f"  doc_id={doc_id} 重建失败：{exc}")

    await query_cache.invalidate_kb(kb_id)

    with SessionLocal() as db:
        failed = _report(build_repositories(db), kb_id)

    if failed:
        logger.warning(f"以下文档重建失败: {failed}")
    return len(doc_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="重建知识库向量索引")
    parser.add_argument("kb_id", type=int, help="知识库 ID")
    parser.add_argument(
        "--keep-vectors",
        action="store_true",
        help="保留旧向量，仅重新处理文档（默认先清空 collection）",
    )
    args = parser.parse_args()
    count = asyncio.run(reindex(args.kb_id, args.keep_vectors))
    logger.info(f"完成：共处理 {count} 个文档")


if __name__ == "__main__":
    main()
