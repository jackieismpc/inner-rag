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
from inner_rag.models import DocStatus, Document
from inner_rag.services.cache import query_cache
from inner_rag.services.document import doc_service
from inner_rag.services.vector_store import vector_service


async def reindex(kb_id: int, keep_vectors: bool = False) -> int:
    if not keep_vectors:
        await vector_service.delete_kb(kb_id)
        logger.info(f"已清空知识库 {kb_id} 的旧向量")

    db = SessionLocal()
    try:
        docs = db.query(Document).filter(Document.kb_id == kb_id).all()
        doc_ids = [doc.id for doc in docs]
        for doc in docs:
            doc.status = DocStatus.PENDING
            doc.chunk_count = 0
            doc.error_msg = None
        db.commit()
    finally:
        db.close()

    for index, doc_id in enumerate(doc_ids, start=1):
        logger.info(f"[{index}/{len(doc_ids)}] 重建 doc_id={doc_id}")
        await doc_service.process_document(doc_id)

    await query_cache.invalidate_kb(kb_id)

    db = SessionLocal()
    try:
        rows = db.query(Document).filter(Document.kb_id == kb_id).all()
        for row in rows:
            logger.info(f"  {row.status.value:10s} chunks={row.chunk_count:4d}  {row.filename}")
        failed = [row.filename for row in rows if row.status == DocStatus.FAILED]
    finally:
        db.close()

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
