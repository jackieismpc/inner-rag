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
  分数没有质量含义，因此**既不写回 README、也不落盘结果**——避免把自检数字当成绩，
  也避免产生随时间漂移的噪声文件（延迟每次都不一样）。
* **kb 模式**不经过 QueryCache 直接调检索层（向量 ∪ 词面融合），保证延迟与召回是真实值；
  `--answer` 会额外调用 LLM，按题计费。
* 检索统一走 ``inner_rag.services.retrieval.search``——与线上问答同一条路径，
  这样「评测涨了、线上没变」这类偏差不可能出现。
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

# 默认扫描网格：覆盖「召回全被挡掉」到「几乎不过滤」的整段，用来找拐点
AUTO_SWEEP = "0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.7"


def _parse_sweep(raw: str) -> list[float]:
    """把 `--threshold-sweep` 的取值解析成阈值列表（`auto` 走内置网格）。"""
    if raw.strip().lower() == "auto":
        raw = AUTO_SWEEP
    try:
        values = sorted({float(part) for part in raw.split(",") if part.strip()})
    except ValueError as exc:
        msg = f"--threshold-sweep 只接受逗号分隔的数字或 'auto'，收到 {raw!r}"
        raise SystemExit(msg) from exc
    if not values:
        raise SystemExit("--threshold-sweep 至少要给一个阈值")
    return values


def _sweep_table(curve: list[dict], current: float) -> str:
    """把阈值曲线排成人读的表格（标出当前配置所在的那一行）。"""
    lines = [
        "阈值定标（同一次未过滤召回反推，配置写错时也能先看数据再定值）",
        "  阈值    Recall@k    MRR     页命中率   空召回   平均保留条数",
    ]
    for row in curve:
        mark = " ← 当前配置" if abs(row["threshold"] - current) < 1e-9 else ""
        lines.append(
            f"  {row['threshold']:<7.2f} {row['recall_at_k']:>8.1%}   {row['mrr']:>5.3f}   "
            f"{row['page_hit_rate']:>7.1%}   {row['empty_items']:>5d}   "
            f"{row['kept_avg']:>11.2f}{mark}"
        )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="inner-rag 基准测试")
    parser.add_argument("--mode", choices=("fixtures", "kb"), default="fixtures")
    parser.add_argument("--dataset", type=Path, default=ds.DEFAULT_DATASET, help="评测集 jsonl")
    parser.add_argument("--fixtures-dir", type=Path, default=ds.DEFAULT_FIXTURES)
    parser.add_argument("--kb-id", type=int, help="kb 模式：要评测的知识库 id")
    parser.add_argument("--strategy", default="hybrid", choices=("similarity", "mmr", "hybrid"))
    parser.add_argument("--k", type=int, default=8, help="召回条数（默认与 TOP_K 一致）")
    parser.add_argument("--threshold", type=float, default=None, help="相关度阈值（默认取配置值）")
    parser.add_argument(
        "--threshold-sweep",
        nargs="?",
        const=AUTO_SWEEP,
        default=None,
        metavar="0,0.1,0.2",
        help=f"阈值定标：跑一次未过滤检索并打印各阈值下的指标（不给值则用内置网格 {AUTO_SWEEP}）",
    )
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
            "ZVEC_PATH": f"{workdir}/zvec",
            "CHROMA_PERSIST_DIR": f"{workdir}/chroma",
            "UPLOAD_DIR": f"{workdir}/uploads",
            "LOG_DIR": f"{workdir}/logs",
            "LOG_RETRIEVAL": "false",
            "LOG_PROMPT": "false",
        }
    )


def _item_result(
    item: dict[str, Any],
    spans: list[tuple[int, int]],
    scores: list[float | None],
    filtered_out: int,
    ms: float,
) -> dict:
    expected = ds.expected_pages(item)
    # 命中判定按页区间算（当前切分不跨页，区间退化为单点）；
    # retrieved_pages 保留单点列表，是为了让旧结果 JSON 与人工核对仍然可读。
    # scores / expected_pages 是阈值定标的原料：有了它们就能从这一次未过滤的召回
    # 反推任意阈值下的指标，不必为每个候选阈值各跑一遍（见 metrics.threshold_curve）。
    return {
        "id": item["id"],
        "question": item["question"],
        "category": item["category"],
        "expect_refusal": item["expect_refusal"],
        "expected_pages": list(expected),
        "retrieved_pages": [span[0] for span in spans],
        "retrieved_spans": [[span[0], span[1]] for span in spans],
        "scores": list(scores),
        "filtered_out": filtered_out,
        "hit": metrics.spans_hit(spans, expected),
        "rr": metrics.spans_reciprocal_rank(spans, expected),
        "page_hit": metrics.spans_page_hit_rate(spans, expected),
        "retrieval_ms": ms,
        "answer": None,
        "keyword_coverage": None,
        "forbidden_hit": None,
        "refusal": None,
        "citation_precision": None,
        "citation_hit": None,
        "total_ms": None,
        "context": "",
        "trace_id": None,
        "judge_correct": None,
        "judge_reason": None,
        "faithfulness": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "cost_usd": None,
    }


def _spans_of(hits: list[tuple[Any, float | None]]) -> list[tuple[int, int]]:
    """召回结果 -> 页区间列表（无页码的分块直接跳过，不参与命中判定）。"""
    from inner_rag.services.vector_store.base import page_span

    spans = []
    for doc, _score in hits:
        span = page_span(doc)
        if span is not None:
            spans.append(span)
    return spans


async def run_fixtures(
    args: argparse.Namespace, items: list[dict], fixtures: list[dict]
) -> list[dict]:
    from langchain_core.documents import Document

    from inner_rag.services import retrieval
    from inner_rag.services.lexical import lexical_index
    from inner_rag.services.vector_store import vector_service

    documents = [
        Document(
            page_content=fixture["text"],
            metadata={"page": fixture["page"], "filename": fixture["file"]},
        )
        for fixture in fixtures
    ]
    written = await vector_service.add_documents(
        kb_id=FIXTURE_KB_ID, documents=documents, doc_id=1, filename="fixtures"
    )
    lexical_index.invalidate(FIXTURE_KB_ID)
    print(f"[fixtures] 写入 {written} 个分块（{len(fixtures)} 段原文，mock embedding）")

    results: list[dict] = []
    for item in items:
        started = time.perf_counter()
        hits, filtered_out = await retrieval.search(
            kb_id=FIXTURE_KB_ID,
            query=item["question"],
            k=args.k,
            strategy=args.strategy,
            score_threshold=args.threshold,
        )
        elapsed = (time.perf_counter() - started) * 1000
        results.append(
            _item_result(
                item,
                _spans_of(hits),
                [score for _doc, score in hits],
                filtered_out,
                elapsed,
            )
        )
    return results


async def run_kb(args: argparse.Namespace, items: list[dict]) -> list[dict]:
    from inner_rag.core.config import settings
    from inner_rag.core.database import SessionLocal
    from inner_rag.models.knowledge_base import KnowledgeBase
    from inner_rag.services import retrieval

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
        hits, filtered_out = await retrieval.search(
            kb_id=args.kb_id,
            query=item["question"],
            k=args.k,
            strategy=args.strategy,
            score_threshold=args.threshold,
        )
        elapsed = (time.perf_counter() - started) * 1000
        result = _item_result(
            item,
            _spans_of(hits),
            [score for _doc, score in hits],
            filtered_out,
            elapsed,
        )

        if args.answer:
            from inner_rag.core.observability import tracer
            from inner_rag.services.rag import rag_service

            began = time.perf_counter()
            # 套一层 eval.item 当 trace 根：rag.request 及其子 span 都挂在它下面。
            # 这样拿到的 trace_id 就是整棵树的根 id，评测分数才能回写到「这一次调用」，
            # 而不是堆在实验维度上无处下钻（见 docs/evaluation.md 6.3）。
            async with tracer.span("eval.item", eval_id=item["id"], kb_id=args.kb_id):
                answer, sources = await rag_service.chat(
                    args.kb_id, item["question"], strategy=args.strategy
                )
                result["trace_id"] = tracer.current_trace_id()
            result["total_ms"] = (time.perf_counter() - began) * 1000
            result["answer"] = answer
            # 保存上下文：judge 判忠实度要有依据，否则只能给「无法验证」
            result["context"] = "\n".join(str(s.get("content") or "") for s in sources)
            cited: list[tuple[int, int]] = []
            for source in sources:
                start = source.get("page_start")
                end = source.get("page_end")
                if start is None or end is None:
                    continue
                span = (int(start), int(end))
                if span not in cited:
                    cited.append(span)
            expected = ds.expected_pages(item)
            result["keyword_coverage"] = metrics.keyword_coverage(answer, item["answer_keywords"])
            result["forbidden_hit"] = metrics.has_forbidden(
                answer, item.get("must_not_include", [])
            )
            result["refusal"] = metrics.is_refusal(answer)
            result["citation_precision"] = metrics.spans_citation_precision(cited, expected)
            result["citation_hit"] = metrics.spans_citation_hit(cited, expected)
            result["cited_spans"] = [[span[0], span[1]] for span in cited]
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
            "threshold_sweep": bool(getattr(args, "sweep_values", None)),
            # 融合权重必须进快照：否则结果 JSON 没法自己解释「这次涨了是因为改了什么」
            "hybrid_sparse_weight": settings.HYBRID_SPARSE_WEIGHT,
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


def _configured_threshold() -> float:
    """当前 .env 里的阈值，用来在扫描表里标出「你现在站在哪一行」。配置坏了就返回 1.0（不标）。"""
    try:
        from inner_rag.core.config import settings

        return float(settings.RETRIEVAL_SCORE_THRESHOLD)
    except Exception:
        return 1.0


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
        "result_file": None,
    }
    sweep_values = getattr(args, "sweep_values", None)
    if sweep_values:
        # 定标数据随结果一起落盘：下次想换网格不必重跑检索
        record["threshold_curve"] = metrics.threshold_curve(results, sweep_values)
    # 只有 kb 模式落盘：fixtures 是「脚本与指标算法」的自检，分数无质量含义，
    # 落盘的 JSON 只会带上每次不同的延迟噪声（历史版本甚至把它提交进了仓库）。
    if args.mode == "kb":
        report.save_result(record, args.out_dir)
    return record


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # 阈值定标：一次未过滤检索就能反推整条曲线，所以这里把检索阈值压到 0，
    # 本次 summary 因此是「过滤前」口径——与 docs/evaluation.md 要求的「过滤前/过滤后两列」一致。
    args.sweep_values = _parse_sweep(args.threshold_sweep) if args.threshold_sweep else []
    if args.sweep_values:
        print("阈值定标模式：本次检索不做阈值过滤（threshold=0），下表为各阈值下的推导指标")
        args.threshold = 0.0

    # 脚本不是 FastAPI 入口，没有 lifespan 替我们初始化追踪；
    # 不 configure 的话 span 只落本地计时，trace_id 全为空，分数也就无处回写。
    from inner_rag.core.observability import tracer

    tracer.configure()
    if args.answer:
        print(f"追踪：{tracer.status()['backend']}（{tracer.status()['reason']}）")

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
    result_line = (
        f"结果已写入 {record['result_file']}"
        if record["result_file"]
        else "自检模式不落盘（fixtures 仅验证脚本与指标算法）"
    )
    # 回答侧指标只在 --answer 跑过之后才有值；引用命中率是可比口径，精度随引用条数漂移，
    # 两个都打出来，避免只看一个得出相反结论（见 docs/evaluation.md 4.4）。
    answer_line = ""
    if summary.get("keyword_pass_rate") is not None:
        answer_line = (
            f"要点命中率={summary['keyword_pass_rate']:.1%}  "
            f"引用命中率={(summary.get('citation_hit_rate') or 0.0):.1%}  "
            f"引用精度={(summary.get('citation_precision') or 0.0):.1%}  "
            f"拒答正确率={(summary.get('refusal_accuracy') or 0.0):.1%}\n"
        )
    print(
        f"\n配置：{record['label']}\n"
        f"题目：{summary['items']}（正样本 {summary['positives']} / 负样本 {summary['negatives']}）\n"
        f"Recall@{args.k}={summary['recall_at_k']:.1%}  MRR={summary['mrr']:.3f}  "
        f"页命中率={summary['page_hit_rate']:.1%}\n"
        f"{answer_line}"
        f"检索 p50={summary['retrieval_p50_ms']:.1f}ms  p95={summary['retrieval_p95_ms']:.1f}ms\n"
        f"{result_line}"
    )
    if args.mode == "fixtures":
        print("注意：fixtures 模式用 mock embedding，指标仅用于验证脚本，不代表检索质量。")

    if record.get("threshold_curve"):
        print("\n" + _sweep_table(record["threshold_curve"], _configured_threshold()))

    if args.update_readme:
        if args.mode != "kb":
            print("拒绝写入 README：自检数据不进入成绩表（请用 --mode kb）")
            return 2
        report.update_readme(record)
        print("README 基准表已更新（同一天同一配置会覆盖旧行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
