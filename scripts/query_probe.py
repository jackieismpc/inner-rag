#!/usr/bin/env python
"""检索探针：直接看某个查询命中了哪些分块、相关度是多少。

用于排查「答非所问」——到底是没召回、被阈值过滤，还是 prompt 组装的问题。

用法:
    uv run scripts/query_probe.py <kb_id> "<query>" [--k 8] [--strategy hybrid] [--threshold 0.3]
"""

from __future__ import annotations

import argparse
import asyncio

from inner_rag.services.retrieval_log import RetrievalStats
from inner_rag.services.vector_store import Strategy, vector_service


async def probe(
    kb_id: int, query: str, k: int, strategy: Strategy, threshold: float | None
) -> None:
    results, filtered_out = await vector_service.search(
        kb_id=kb_id, query=query, k=k, strategy=strategy, score_threshold=threshold
    )

    print(f"\nkb_id={kb_id}  strategy={strategy}  k={k}  threshold={threshold}")
    print(f"query={query!r}")
    print(f"命中 {len(results)} 条，被阈值过滤 {filtered_out} 条\n")

    for index, (doc, score) in enumerate(results, start=1):
        meta = doc.metadata
        score_text = f"{score:.4f}" if score is not None else "n/a (mmr)"
        print(
            f"[{index}] relevance={score_text}  file={meta.get('filename')!r}  "
            f"chunk={meta.get('chunk_index')}  doc_id={meta.get('doc_id')}"
        )
        print(f"     {doc.page_content[:160]!r}\n")

    print("检索统计:", RetrievalStats.summary())


def main() -> None:
    parser = argparse.ArgumentParser(description="检索探针")
    parser.add_argument("kb_id", type=int)
    parser.add_argument("query", type=str)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--strategy", choices=["similarity", "mmr", "hybrid"], default="hybrid")
    parser.add_argument("--threshold", type=float, default=None, help="留空则使用配置默认值")
    args = parser.parse_args()
    asyncio.run(probe(args.kb_id, args.query, args.k, args.strategy, args.threshold))


if __name__ == "__main__":
    main()
