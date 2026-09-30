#!/usr/bin/env python3
"""回答侧评测：LLM-as-judge 正确性、忠实度、token 与成本，并产出可读报告。

检索指标由 `benchmark/run_bench.py` 负责（不需要 LLM），这里补上「找到了但没答对」那一半
（见 `docs/evaluation.md` 1 / 4.2）。两种入口：

```bash
# 1) 自己跑问答（调 LLM 产生费用）
uv run scripts/eval_answer.py --kb-id 7

# 2) 复用 benchmark 已经跑出来的回答（省一次问答调用，只花 judge 的钱）
uv run scripts/eval_answer.py --from-result benchmark/results/2026-09-30-kb-xxx.json
```

产物：`docs/reports/eval-<日期>-<label>.md`（人读）+ 同名 `.json`（给下次报告做差值对比）。

两条硬约束：

* **judge identity 必须记录**：模型或 prompt 变了，分数就不可比，报告里写明 `judge` 与
  `judge_prompt_version`；
* **成本不拍脑袋**：单价属计费域，默认 0 并在报告里注明「未配置价格表」；
  要算真实费用请用 `--price-prompt` / `--price-completion`（USD / 1M tokens）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
# 脚本直接执行时 sys.path[0] 是 scripts/，而 benchmark 是仓库顶层目录（未安装成包）
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark import dataset as ds  # noqa: E402 - 需要先补 sys.path
from benchmark import langsmith_sync, metrics  # noqa: E402

DEFAULT_REPORT_DIR = REPO_ROOT / "docs" / "reports"

# judge prompt 的版本号：改 prompt 必须同时改这里，否则历史分数会被当成同一口径对比
JUDGE_PROMPT_VERSION = "v1"

JUDGE_PROMPT = """你是企业知识库问答的评分员。只根据「期望答案」判断「模型回答」是否正确，
不要因为回答写得详细就加分，也不要因为措辞不同就扣分。

[问题]
{question}

[期望答案]
{expected_answer}

[模型回答]
{answer}

[参考资料（模型检索到的上下文，可能为空）]
{context}

请按以下 JSON 输出，不要输出任何解释性文字：
{{"correct": 0 或 1, "faithfulness": 0 或 1, "reason": "一句话说明"}}

- correct：模型回答是否表达出期望答案的核心结论（同义改写算对）；
- faithfulness：模型回答里的关键断言是否都能在参考资料中找到依据（无法判断时给 0）。
"""

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_judge_response(text: str) -> dict[str, Any]:
    """从 judge 的原始输出里抠出 JSON 并归一化；解析不出来时按「未判定」处理。

    judge 是外部模型，输出带 ```json 围栏、前缀废话、全角标点都是常态，
    这里只做「尽量解析 + 解析失败显性化」，不重试、不猜分数。
    """
    if not text:
        return {"correct": None, "faithfulness": None, "reason": "judge 无输出"}
    cleaned = text.strip()
    fence = _JSON_BLOCK.search(cleaned)
    if fence:
        cleaned = fence.group(0)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return {"correct": None, "faithfulness": None, "reason": f"judge 输出无法解析：{text[:80]}"}

    def _score(value: Any) -> int | None:
        if isinstance(value, bool):
            return 1 if value else 0
        if isinstance(value, (int, float)):
            return 1 if float(value) >= 0.5 else 0
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "正确", "是"}:
                return 1
            if lowered in {"0", "false", "no", "错误", "否"}:
                return 0
        return None

    return {
        "correct": _score(payload.get("correct")),
        "faithfulness": _score(payload.get("faithfulness")),
        "reason": str(payload.get("reason", ""))[:200],
    }


def estimate_cost_usd(
    prompt_tokens: int, completion_tokens: int, price_prompt: float, price_completion: float
) -> float:
    """按 USD / 1M tokens 估算；单价为 0（未配置）时结果是 0，绝不用拍脑袋的价格。"""
    return round(prompt_tokens / 1_000_000 * price_prompt, 6) + round(
        completion_tokens / 1_000_000 * price_completion, 6
    )


def attribute_failure(result: dict[str, Any]) -> str:
    """把一道错题归因到具体环节——「分数低」没有行动价值，归因才有。

    * 检索失败：证据根本没召回；
    * 阈值过严：召回了但被相关度阈值挡掉；
    * 引用错误：召回与回答都对，但引用页码不对；
    * 生成失败：证据在上下文里，模型没用上。
    """
    if result.get("expect_refusal"):
        return "拒答失败（负样本未拒绝）" if not result.get("refusal") else "—"
    if not result.get("hit"):
        # 只有「一条都没召回且确实有被过滤的」才算阈值过严；
        # 召回了别的页却没命中证据，说明是语义没匹配上，改阈值也救不了。
        if not result.get("retrieved_pages") and result.get("filtered_out"):
            return "阈值过严（全部召回被过滤）"
        return "检索失败（证据未召回）"
    if result.get("judge_correct") == 0 or (result.get("keyword_coverage") or 0) < 1.0:
        if (result.get("citation_precision") or 0) < 0.5:
            return "引用错误（依据与回答不符）"
        return "生成失败（证据在上下文里但没答对）"
    return "—"


def _token_delta(before: dict[str, Any], after: dict[str, Any], kind: str) -> int:
    """从指标快照里取 token 增量（按 type 标签聚合，与 provider 无关）。

    为什么不直接读返回值：``rag_service.chat`` 的契约是 ``(answer, sources)``，
    为评测把 token 塞进返回值会污染业务接口；指标注册表本来就记了累计用量，
    取差值就能拿到本次调用的用量，且对流式/非流式都成立。
    """
    total = 0.0
    for key, value in after.get("counters", {}).items():
        if not key.startswith("rag_llm_tokens_total") or f'type="{kind}"' not in key:
            continue
        total += value - float(before.get("counters", {}).get(key, 0.0))
    return round(total)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
        ).stdout.strip()
    except Exception:
        return "unknown"


async def _judge(item: dict[str, Any], answer: str, context: str) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage

    from inner_rag.providers import get_chat_model

    model = get_chat_model()
    prompt = JUDGE_PROMPT.format(
        question=item["question"],
        expected_answer=item["expected_answer"],
        answer=answer[:2000],
        context=context[:2000] or "（无）",
    )
    message = await model.ainvoke([HumanMessage(content=prompt)])
    verdict = parse_judge_response(str(getattr(message, "content", "")))
    usage = getattr(message, "usage_metadata", None) or {}
    verdict["prompt_tokens"] = int(usage.get("input_tokens") or 0)
    verdict["completion_tokens"] = int(usage.get("output_tokens") or 0)
    return verdict


async def judge_items(
    items: list[dict[str, Any]],
    results: list[dict[str, Any]],
    *,
    price_prompt: float,
    price_completion: float,
) -> None:
    """逐题补齐 judge 正确性 / 忠实度 / token / 成本（就地修改 results）。"""
    from inner_rag.providers.specs import chat_spec

    by_id = {item["id"]: item for item in items}
    print(f"[eval] judge: {chat_spec().identity} / prompt {JUDGE_PROMPT_VERSION}")
    for result in results:
        item = by_id.get(result["id"])
        if item is None or result.get("answer") is None:
            continue
        verdict = await _judge(item, str(result["answer"]), str(result.get("context") or ""))
        result["judge_correct"] = verdict["correct"]
        result["judge_reason"] = verdict["reason"]
        result["faithfulness"] = verdict["faithfulness"]
        # judge 自己不走 rag_service，token 只能从 AIMessage 上取；
        # 两者相加才是这道题真实的 LLM 开销（问答 + 评分）
        prompt_tokens = int(result.get("prompt_tokens") or 0) + int(
            verdict.get("prompt_tokens") or 0
        )
        completion_tokens = int(result.get("completion_tokens") or 0) + int(
            verdict.get("completion_tokens") or 0
        )
        result["prompt_tokens"] = prompt_tokens
        result["completion_tokens"] = completion_tokens
        result["cost_usd"] = estimate_cost_usd(
            prompt_tokens, completion_tokens, price_prompt, price_completion
        )
        result["failure"] = attribute_failure(result)
        flag = {1: "✓", 0: "✗", None: "?"}[verdict["correct"]]
        print(f"  {result['id']:<24} judge={flag} {verdict['reason'][:40]}")


async def answer_items(kb_id: int, items: list[dict[str, Any]], strategy: str) -> list[dict]:
    """自己跑一遍「检索 + 问答」，产出与 benchmark 同构的逐题结果。"""
    from inner_rag.core.metrics import metrics as metrics_registry
    from inner_rag.services.rag import rag_service

    results: list[dict] = []
    for item in items:
        before = metrics_registry.snapshot()
        started = time.perf_counter()
        answer, sources = await rag_service.chat(kb_id, item["question"], strategy=strategy)
        total_ms = (time.perf_counter() - started) * 1000
        cited: list[tuple[int, int]] = []
        for source in sources:
            start = source.get("page_start")
            end = source.get("page_end")
            if start is None or end is None:
                continue
            span = (int(start), int(end))
            if span not in cited:
                cited.append(span)
        spans = [tuple(span) for span in cited]  # 只看模型真正引用到的区间
        expected = ds.expected_pages(item)
        results.append(
            {
                "id": item["id"],
                "question": item["question"],
                "category": item["category"],
                "expect_refusal": item["expect_refusal"],
                "retrieved_pages": [span[0] for span in spans],
                "retrieved_spans": [[span[0], span[1]] for span in spans],
                "filtered_out": 0,
                "hit": metrics.spans_hit(spans, expected),
                "rr": metrics.spans_reciprocal_rank(spans, expected),
                "page_hit": metrics.spans_page_hit_rate(spans, expected),
                "retrieval_ms": None,
                "answer": answer,
                "context": "\n".join(str(s.get("content") or "") for s in sources),
                "keyword_coverage": metrics.keyword_coverage(answer, item["answer_keywords"]),
                "forbidden_hit": metrics.has_forbidden(answer, item.get("must_not_include", [])),
                "refusal": metrics.is_refusal(answer),
                "citation_precision": metrics.spans_citation_precision(cited, expected),
                "cited_spans": [[span[0], span[1]] for span in cited],
                "total_ms": total_ms,
                "judge_correct": None,
                "judge_reason": None,
                "faithfulness": None,
                "prompt_tokens": _token_delta(before, metrics_registry.snapshot(), "prompt"),
                "completion_tokens": _token_delta(
                    before, metrics_registry.snapshot(), "completion"
                ),
                "cost_usd": None,
            }
        )
        flag = "✓" if results[-1]["hit"] else "✗"
        print(f"  {item['id']:<24} 引用命中={flag} {total_ms:.0f}ms")
    return results


def render_report(
    record: dict[str, Any], items: list[dict[str, Any]], previous: dict[str, Any] | None
) -> str:
    """报告正文：先给结论（能不能合入），再给证据（逐题明细）。"""
    summary = record["summary"]
    config = record["config"]
    lines: list[str] = []
    lines.append(f"# 评测报告 {record['date']} · {record['label']}")
    lines.append("")
    lines.append(f"- 生成时间：{record['generated_at']}")
    lines.append(f"- git commit：`{config.get('git_commit')}` / app {config.get('app_version')}")
    lines.append(
        f"- 配置：`{config.get('llm')}` + `{config.get('embedding')}`，"
        f"strategy={config.get('strategy')} k={config.get('k')} threshold={config.get('threshold')}"
    )
    lines.append(
        f"- 分块：size={config.get('chunk_size')} overlap={config.get('chunk_overlap')} "
        f"embedding_max_input_chars={config.get('embedding_max_input_chars')}"
    )
    lines.append(
        f"- judge：`{config.get('judge')}` / prompt `{config.get('judge_prompt_version')}`"
    )
    lines.append(
        f"- 估算成本：{summary.get('cost_usd')} USD"
        + (
            ""
            if config.get("price_prompt") or config.get("price_completion")
            else "（**未配置价格表**，token 用量见下表，成本留待 Phase 10）"
        )
    )
    lines.append("")

    lines.append("## 1. 指标")
    lines.append("")
    lines.append("| 指标 | 数值 | 门禁（G3） |")
    lines.append("| --- | --- | --- |")
    gate = {
        "recall_at_k": "≥ 0.8（Phase 8 目标）",
        "citation_precision": "≥ 0.8（Phase 8 目标）",
        "keyword_pass_rate": "≥ 0.8（Phase 8 目标）",
        "refusal_accuracy": "≥ 0.9",
        "false_refusal_rate": "不得上升",
    }
    for key, target in gate.items():
        value = summary.get(key)
        shown = "—" if value is None else f"{value:.1%}"
        lines.append(f"| {key} | {shown} | {target} |")
    for key in ("mrr", "page_hit_rate", "judge_accuracy", "faithfulness"):
        value = summary.get(key)
        shown = "—" if value is None else f"{value:.1%}"
        lines.append(f"| {key} | {shown} | — |")
    lines.append("")

    lines.append("## 2. 逐题明细")
    lines.append("")
    lines.append("| id | 检索命中 | 引用精度 | 要点 | judge | 忠实度 | 归因 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for result in record["items"]:
        hit = "✓" if result.get("hit") else "✗"
        cited = result.get("citation_precision")
        cited_txt = "—" if cited is None else f"{cited:.0%}"
        keyword = result.get("keyword_coverage")
        keyword_txt = "—" if keyword is None else f"{keyword:.0%}"
        judge = {1: "✓", 0: "✗", None: "?"}.get(result.get("judge_correct"), "?")
        faithful = result.get("faithfulness")
        faithful_txt = "—" if faithful is None else f"{faithful:.0%}"
        lines.append(
            f"| `{result['id']}` | {hit} | {cited_txt} | {keyword_txt} | {judge} | "
            f"{faithful_txt} | {result.get('failure', '—')} |"
        )
    lines.append("")

    lines.append("## 3. 失败分析")
    lines.append("")
    failures = [r for r in record["items"] if r.get("failure") not in {None, "—"}]
    if not failures:
        lines.append("本次无失败题。")
    else:
        for result in failures:
            lines.append(f"- `{result['id']}`：{result['failure']}")
            if result.get("judge_reason"):
                lines.append(f"  - judge：{result['judge_reason']}")
            lines.append(f"  - 召回页：{result.get('retrieved_pages')}")
    lines.append("")

    lines.append("## 4. 与上一次对比")
    lines.append("")
    if previous is None:
        lines.append("没有可对比的历史报告（本次为基线）。")
    else:
        prev_summary = previous.get("summary", {})
        lines.append(f"对比对象：{previous.get('date')} · {previous.get('label')}")
        lines.append("")
        lines.append("| 指标 | 上次 | 本次 | 差值 |")
        lines.append("| --- | --- | --- | --- |")
        for key in ("recall_at_k", "citation_precision", "keyword_pass_rate", "refusal_accuracy"):
            before = prev_summary.get(key)
            after = summary.get(key)
            if before is None or after is None:
                continue
            delta = (after - before) * 100
            lines.append(f"| {key} | {before:.1%} | {after:.1%} | {delta:+.1f}pp |")
    lines.append("")
    lines.append(
        f"（题目共 {summary.get('items')} 条：正样本 {summary.get('positives')} / "
        f"负样本 {summary.get('negatives')}）"
    )
    return "\n".join(lines) + "\n"


def _latest_previous(report_dir: Path, exclude: Path) -> dict[str, Any] | None:
    # eval-kb-*.json 是建库 manifest，不是评测报告：混进来会让「上次」变成 None
    candidates = sorted(
        (path for path in report_dir.glob("eval-*.json") if "kb-" not in path.name),
        key=lambda p: p.stat().st_mtime,
    )
    for path in reversed(candidates):
        if path.resolve() == exclude.resolve():
            continue
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # pragma: no cover - 历史文件损坏不该中断本次评测
            continue
    return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="回答侧评测（judge / 忠实度 / 成本）")
    parser.add_argument("--kb-id", type=int, help="自己跑问答时的知识库 id")
    parser.add_argument("--strategy", default="hybrid", choices=("similarity", "mmr", "hybrid"))
    parser.add_argument("--dataset", type=Path, default=ds.DEFAULT_DATASET)
    parser.add_argument("--from-result", type=Path, help="复用 benchmark 结果 JSON（省一次问答）")
    parser.add_argument("--label", default="", help="报告标题里的配置标签")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--price-prompt", type=float, default=0.0, help="USD / 1M prompt tokens")
    parser.add_argument(
        "--price-completion", type=float, default=0.0, help="USD / 1M completion tokens"
    )
    parser.add_argument("--no-judge", action="store_true", help="只出检索与关键词指标，不调 judge")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.kb_id and not args.from_result:
        msg = "必须指定 --kb-id 或 --from-result"
        raise SystemExit(msg)

    items = ds.load_eval_set(args.dataset)

    source_config: dict[str, Any] = {}
    if args.from_result:
        record = json.loads(args.from_result.read_text(encoding="utf-8"))
        source_config = record.get("config") or {}
        results = record["items"]
        for result in results:
            result.setdefault("context", "")
        label = args.label or record.get("label", "from-result")
    else:
        results = asyncio.run(answer_items(args.kb_id, items, args.strategy))
        label = args.label or f"kb{args.kb_id}/{args.strategy}"

    if not args.no_judge:
        asyncio.run(
            judge_items(
                items,
                results,
                price_prompt=args.price_prompt,
                price_completion=args.price_completion,
            )
        )
    for result in results:
        result.setdefault("failure", attribute_failure(result))

    # 把分数回写到对应 trace（追踪未启用时是 no-op，不影响报告产出）
    feedback_written = langsmith_sync.push_feedback(langsmith_sync.client_or_none(), results)
    if feedback_written:
        print(f"[eval] 已回写 {feedback_written} 条 feedback 到 LangSmith")

    from inner_rag.core.config import settings
    from inner_rag.providers.specs import chat_spec

    today = datetime.now().strftime("%Y-%m-%d")
    slug = re.sub(r"[^\w.-]+", "-", label)
    out_md = args.out_dir / f"eval-{today}-{slug}.md"
    out_json = args.out_dir / f"eval-{today}-{slug}.json"
    record = {
        "date": today,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "label": label,
        "config": {
            "strategy": args.strategy,
            "k": source_config.get("k"),
            "llm": chat_spec().identity,
            "embedding": settings.embedding_key,
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "embedding_max_input_chars": settings.EMBEDDING_MAX_INPUT_CHARS,
            "top_k": settings.TOP_K,
            "rerank_top_k": settings.RERANK_TOP_K,
            "threshold": settings.RETRIEVAL_SCORE_THRESHOLD,
            "app_version": settings.APP_VERSION,
            "git_commit": _git_commit(),
            "judge": chat_spec().identity,
            "judge_prompt_version": JUDGE_PROMPT_VERSION,
            "price_prompt": args.price_prompt,
            "price_completion": args.price_completion,
            "source": str(args.from_result) if args.from_result else f"kb:{args.kb_id}",
            "langsmith_feedback": feedback_written,
        },
        "summary": metrics.summarize(results),
        "items": results,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    previous = _latest_previous(args.out_dir, out_json)
    out_md.write_text(render_report(record, items, previous), encoding="utf-8")
    out_json.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n报告：{out_md}\n数据：{out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
