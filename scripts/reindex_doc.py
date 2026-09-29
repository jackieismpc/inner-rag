#!/usr/bin/env python
"""重建单个文档的向量索引。

用法:
    uv run scripts/reindex_doc.py <doc_id>
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


async def reindex_doc(doc_id: int) -> None:
    db = SessionLocal()
    try:
        doc = db.get(Document, doc_id)
        if doc is None:
            raise SystemExit(f"文档不存在: {doc_id}")
        kb_id = doc.kb_id
        filename = doc.filename
        doc.status = DocStatus.PENDING
        doc.chunk_count = 0
        doc.error_msg = None
        db.commit()
    finally:
        db.close()

    await vector_service.delete_document(kb_id, doc_id)
    logger.info(f"已删除 doc_id={doc_id} 的旧向量")

    await doc_service.process_document(doc_id)
    await query_cache.invalidate_kb(kb_id)

    db = SessionLocal()
    try:
        doc = db.get(Document, doc_id)
        if doc is not None:
            logger.info(f"完成: {filename} | status={doc.status.value} | chunks={doc.chunk_count}")
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="重建单个文档的向量索引")
    parser.add_argument("doc_id", type=int, help="文档 ID")
    args = parser.parse_args()
    asyncio.run(reindex_doc(args.doc_id))


if __name__ == "__main__":
    main()
