"""基准测试执行入口：跑一遍评测集，输出指标、落盘 JSON，并可写回 README 表格。

用法：

```bash
# 离线自检：mock provider + fixture 小库，不联网、不花钱（用于验证脚本与指标计算）
uv run python -m benchmark.run_bench --mode fixtures

# 真实小库：用 .env 里配置的 provider 查已有知识库，检索指标 + 延迟
uv run python -m benchmark.run_bench --mode kb --kb-id 3 --update-readme

# 真实小库 + 回答指标（会调用 LLM，产生费用）：要点命中率、引用精度、拒答正确率
uv run python -m benchmark.run_bench --mode kb --kb-id 3 --answer --update-readme

# 调参实验：换策略/阈值/k 后再跑，行会写进同一张表便于对比
uv run python -m benchmark.run_bench --mode kb --kb-id 3 --strategy hybrid --threshold 0.25
```

说明：

* **fixtures 模式**测的是「脚本与指标算得对不对」，mock embedding 只有词面相似度，
  分数没有质量含义，因此**不允许**写回 README（避免把自检数字当成绩）。
* **kb 模式**不经过 QueryCache 直接调向量库，保证延迟与召回是真实值；
  `--answer` 会额外调用 LLM，按题计费。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from benchmark import dataset as ds
from benchmark import metrics, report

FIXTURE_KB_ID = 990001


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="inner-rag 基准测试")
    parser.add_argument("--mode", choices=("fixtures", "kb"), default="fixtures")
    parser.add_argument("--dataset", type=Path, default=ds.DEFAULT_DATASET, help="评测集 jsonl")
    parser.add_argument("--fixtures-dir", type=Path, default=ds.DEFAULT_FIXTURES)
    parser.add_argument("--kb-id", type=int, help="kb 模式：要评测的知识库 id")
    parser.add_argument("--strategy", default="hybrid", choices=("similarity", "mmr", "hybrid"))
    parser.add_argument("--k", type=int, default=8, help="召回条数（默认与 TOP_K 一致）")
    parser.add_argument("--threshold", type=float, default=None, help="相关度阈值（默认取配置值）")
    parser.add_argument("--answer", action="store_true", help="kb 模式：额外跑回答指标（调 LLM）")
    parser.add_argument("--label", default="", help="配置标签，用于 README 表格里区分实验")
    parser.add_argument("--update-readme", action="store_true", help="把结果写入 README 基准表")
    parser.add_argument("--out-dir", type=Path, default=report.DEFAULT_RESULTS_DIR)
    return parser.parse_args(argv)


def _offline_env(workdir: Path) -> None:
    """在导入 inner_rag 之前把环境钉成离线 mock，避免受开发者本机 .env 影响。"""
    os.environ.update(
        {
            "LLM_PROVIDER": "mock",
            "EMBEDDING_PROVIDER": "mock",
            "EMBEDDING_MAX_INPUT_CHARS": "0",
            "DATABASE_URL": f"sqlite:///{workdir}/bench.db",
            "CHROMA_PERSIST_DIR": f"{workdir}/chroma",
            "UPLOAD_DIR": f"{workdir}/uploads",
            "LOG_DIR": f"{workdir}/logs",
            "LOG_RETRIEVAL": "false",
            "LOG_PROMPT": "false",
        }
    )


def _item_result(item: dict[str, Any], pages: list[int], filtered_out: int, ms: float) -> dict:
    expected = ds.expected_pages(item)
    return {
        "id": item["id"],
        "question": item["question"],
        "category": item["category"],
        "expect_refusal": item["expect_refusal"],
        "retrieved_pages": pages,
        "filtered_out": filtered_out,
        "hit": metrics.pages_hit(pages, expected),
        "rr": metrics.reciprocal_rank(pages, expected),
        "page_hit": metrics.page_hit_rate(pages, expected),
        "retrieval_ms": ms,
        "answer": None,
        "keyword_coverage": None,
        "forbidden_hit": None,
        "refusal": None,
        "citation_precision": None,
        "total_ms": None,
    }


async def run_fixtures(
    args: argparse.Namespace, items: list[dict], fixtures: list[dict]
) -> list[dict]:
    from langchain_core.documents import Document

    from inner_rag.services.vector_store import vector_service

    documents = [
        Document(
            page_content=fixture["text"],
            metadata={"page": fixture["page"], "filename": fixture["file"]},
        )
        for fixture in fixtures
    ]
    written = await vector_service.add_documents_async(
        kb_id=FIXTURE_KB_ID, documents=documents, doc_id=1, filename="fixtures"
    )
    print(f"[fixtures] 写入 {written} 个分块（{len(fixtures)} 段原文，mock embedding）")

    results: list[dict] = []
    for item in items:
        started = time.perf_counter()
        hits, filtered_out = await vector_service.similarity_search_async(
            kb_id=FIXTURE_KB_ID,
            query=item["question"],
            k=args.k,
            strategy=args.strategy,
            score_threshold=args.threshold,
        )
        elapsed = (time.perf_counter() - started) * 1000
        pages = [int(doc.metadata["page"]) for doc, _ in hits if doc.metadata.get("page")]
        results.append(_item_result(item, pages, filtered_out, elapsed))
    return results


async def run_kb(args: argparse.Namespace, items: list[dict]) -> list[dict]:
    from inner_rag.core.config import settings
    from inner_rag.core.database import SessionLocal
    from inner_rag.models.knowledge_base import KnowledgeBase
    from inner_rag.services.vector_store import vector_service

    if not args.kb_id:
        msg = "kb 模式必须指定 --kb-id"
        raise SystemExit(msg)

    with SessionLocal() as session:
        kb = session.get(KnowledgeBase, args.kb_id)
    if kb is None:
        raise SystemExit(f"知识库 {args.kb_id} 不存在（先建库并上传文档）")

    try:
        identity = settings.embedding_key
    except Exception as exc:  # ProviderError 等：配置问题要给出可读提示
        raise SystemExit(f"embedding 配置不可用：{exc}") from exc
    if kb.embedding_model and kb.embedding_model != identity:
        raise SystemExit(
            f"知识库 {args.kb_id} 建库用 {kb.embedding_model}，当前配置为 {identity}；"
            "请切回原模型或先跑 scripts/reindex_kb.py 重建索引"
        )

    results: list[dict] = []
    for item in items:
        started = time.perf_counter()
        hits, filtered_out = await vector_service.similarity_search_async(
            kb_id=args.kb_id,
            query=item["question"],
            k=args.k,
            strategy=args.strategy,
            score_threshold=args.threshold,
        )
        elapsed = (time.perf_counter() - started) * 1000
        pages = [int(doc.metadata["page"]) for doc, _ in hits if doc.metadata.get("page")]
        result = _item_result(item, pages, filtered_out, elapsed)

        if args.answer:
            from inner_rag.services.rag import rag_service

            began = time.perf_counter()
            answer, sources = await rag_service.chat(
                args.kb_id, item["question"], strategy=args.strategy
            )
            result["total_ms"] = (time.perf_counter() - began) * 1000
            result["answer"] = answer
            cited = [int(s["page"]) for s in sources if s.get("page")]
            expected = ds.expected_pages(item)
            result["keyword_coverage"] = metrics.keyword_coverage(answer, item["answer_keywords"])
            result["forbidden_hit"] = metrics.has_forbidden(
                answer, item.get("must_not_include", [])
            )
            result["refusal"] = metrics.is_refusal(answer)
            result["citation_precision"] = metrics.citation_precision(cited, expected)
        results.append(result)
    return results


def _config_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from inner_rag.core.config import settings

        return {
            "strategy": args.strategy,
            "k": args.k,
            "threshold": args.threshold
            if args.threshold is not None
            else settings.RETRIEVAL_SCORE_THRESHOLD,
            "llm": settings.LLM_PROVIDER,
            "embedding": settings.embedding_key,
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "embedding_max_input_chars": settings.EMBEDDING_MAX_INPUT_CHARS,
            "app_version": settings.APP_VERSION,
            "git_commit": _git_commit(),
        }
    except Exception as exc:  # pragma: no cover - 配置异常时仍要能落盘结果
        return {"error": str(exc)}


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    items = ds.load_eval_set(args.dataset)
    fixtures = ds.load_fixtures(args.fixtures_dir)
    problems = ds.validate_or_raise(items, fixtures)
    for problem in problems:
        print(f"[validate] {problem}")

    if args.mode == "fixtures":
        results = await run_fixtures(args, items, fixtures)
        default_label = f"mock+fixtures 自检/{args.strategy}/k={args.k}"
    else:
        results = await run_kb(args, items)
        config = _config_snapshot(args)
        default_label = f"kb{args.kb_id}/{config.get('embedding')}/{args.strategy}/k={args.k}"
        if args.answer:
            default_label += "+answer"

    record = {
        "date": report.today(),
        "mode": args.mode,
        "label": args.label or default_label,
        "config": _config_snapshot(args),
        "summary": metrics.summarize(results),
        "items": results,
    }
    report.save_result(record, args.out_dir)
    return record


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    workdir_ctx = None
    if args.mode == "fixtures":
        workdir_ctx = tempfile.TemporaryDirectory(prefix="inner-rag-bench-")
        _offline_env(Path(workdir_ctx.name))

    try:
        record = asyncio.run(_run(args))
    finally:
        if workdir_ctx is not None:
            workdir_ctx.cleanup()

    print(report.console_table(record))
    summary = record["summary"]
    print(
        f"\n配置：{record['label']}\n"
        f"题目：{summary['items']}（正样本 {summary['positives']} / 负样本 {summary['negatives']}）\n"
        f"Recall@{args.k}={summary['recall_at_k']:.1%}  MRR={summary['mrr']:.3f}  "
        f"页命中率={summary['page_hit_rate']:.1%}\n"
        f"检索 p50={summary['retrieval_p50_ms']:.1f}ms  p95={summary['retrieval_p95_ms']:.1f}ms\n"
        f"结果已写入 {record['result_file']}"
    )
    if args.mode == "fixtures":
        print("注意：fixtures 模式用 mock embedding，指标仅用于验证脚本，不代表检索质量。")

    if args.update_readme:
        if args.mode != "kb":
            print("拒绝写入 README：自检数据不进入成绩表（请用 --mode kb）")
            return 2
        report.update_readme(record)
        print("README 基准表已更新（同一天同一配置会覆盖旧行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
