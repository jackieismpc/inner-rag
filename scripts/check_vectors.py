#!/usr/bin/env python
"""检查向量库内容：分块总数、各文件分块数、以及（可选）与数据库记录的一致性。

用法:
    uv run scripts/check_vectors.py            # 检查所有知识库
    uv run scripts/check_vectors.py 1          # 只检查 kb_id=1
"""

from __future__ import annotations

import argparse
from collections import Counter

from inner_rag.core.database import SessionLocal
from inner_rag.models import Document, KnowledgeBase
from inner_rag.services.vector_store import vector_service


def main() -> None:
    parser = argparse.ArgumentParser(description="检查向量库内容")
    parser.add_argument("kb_id", nargs="?", type=int, help="知识库 ID（省略则检查全部）")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        if args.kb_id is not None:
            kbs = [db.get(KnowledgeBase, args.kb_id)]
        else:
            kbs = db.query(KnowledgeBase).order_by(KnowledgeBase.id.asc()).all()
        kbs = [kb for kb in kbs if kb is not None]
        if not kbs:
            print("没有找到知识库")
            return

        for kb in kbs:
            vector_count = vector_service.count(kb.id)
            counts = vector_service.count_chunks_by_filename(kb.id)
            docs = db.query(Document).filter(Document.kb_id == kb.id).all()

            print(f"\n=== kb_id={kb.id}  {kb.name}  embedding={kb.embedding_model} ===")
            print(f"collection 向量数: {vector_count}  文件数: {len(counts)}")
            for filename, chunk_count in sorted(counts.items(), key=lambda kv: -kv[1]):
                print(f"  {chunk_count:5d} chunks | {filename}")

            # 数据库分块数 vs 实际向量数，不一致说明有残留或漏写
            mismatch = [
                (doc.filename, doc.chunk_count, counts.get(doc.filename, 0))
                for doc in docs
                if doc.status.value == "completed"
                and doc.chunk_count != counts.get(doc.filename, 0)
            ]
            if mismatch:
                print("⚠️  数据库记录与实际向量数不一致:")
                for filename, expected, actual in mismatch:
                    print(f"    {filename}: db={expected} vector_db={actual}")
            else:
                print("✅ 数据库记录与实际向量数一致")

            # collection 里出现数据库不认识的 doc_id 说明删除逻辑有残留
            known = {str(doc.id) for doc in docs}
            orphans = Counter(
                doc_id for doc_id in vector_service.list_doc_ids(kb.id) if doc_id not in known
            )
            if orphans:
                print(f"⚠️  存在无主向量（数据库已无对应文档）: {dict(orphans)}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
