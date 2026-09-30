"""基准结果落盘与 README 表格更新。

README 里被下面两个标记包住的区域由脚本维护，手工改动会被覆盖：

    <!-- BEGIN BENCHMARK -->
    <!-- END BENCHMARK -->
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "benchmark" / "results"
README_PATH = REPO_ROOT / "README.md"
MARKER_BEGIN = "<!-- BEGIN BENCHMARK -->"
MARKER_END = "<!-- END BENCHMARK -->"

TABLE_HEADER = (
    "| 日期 | 配置 | 题数 | Recall@k | MRR | 页命中率 | 要点命中率 | 引用精度 | 拒答正确率 "
    "| 检索 p50 | 检索 p95 | 端到端 p50 | 结果文件 |"
)
TABLE_SEPARATOR = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def _ms(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f} ms"


def render_row(record: dict[str, Any]) -> str:
    summary = record["summary"]
    return (
        f"| {record['date']} | {record['label']} | {summary['items']} "
        f"| {_pct(summary['recall_at_k'])} | {summary['mrr']:.3f} "
        f"| {_pct(summary['page_hit_rate'])} | {_pct(summary['keyword_pass_rate'])} "
        f"| {_pct(summary['citation_precision'])} | {_pct(summary['refusal_accuracy'])} "
        f"| {_ms(summary['retrieval_p50_ms'])} | {_ms(summary['retrieval_p95_ms'])} "
        f"| {_ms(summary['total_p50_ms'])} | `{record['result_file']}` |"
    )


def save_result(record: dict[str, Any], results_dir: Path | str | None = None) -> Path:
    results_dir = Path(results_dir) if results_dir else DEFAULT_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "-", record["label"]).strip("-") or "run"
    path = results_dir / f"{record['date']}-{record['mode']}-{slug}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    record["result_file"] = str(path.relative_to(REPO_ROOT))
    return path


def update_readme(record: dict[str, Any], readme_path: Path | str | None = None) -> bool:
    """把一行结果写入 README 的基准区间；同一「日期 + 配置」的行会被覆盖。"""
    readme_path = Path(readme_path) if readme_path else README_PATH
    text = readme_path.read_text(encoding="utf-8")
    if MARKER_BEGIN not in text or MARKER_END not in text:
        msg = f"{readme_path} 缺少 {MARKER_BEGIN} / {MARKER_END} 标记"
        raise ValueError(msg)

    head, rest = text.split(MARKER_BEGIN, 1)
    body, tail = rest.split(MARKER_END, 1)
    rows = [
        line
        for line in body.strip().splitlines()
        if line.startswith("|") and not line.startswith("| 日期") and "---" not in line
    ]
    key = f"| {record['date']} | {record['label']} |"
    rows = [row for row in rows if not row.startswith(key)]
    rows.append(render_row(record))

    new_body = "\n" + "\n".join([TABLE_HEADER, TABLE_SEPARATOR, *rows]) + "\n"
    readme_path.write_text(f"{head}{MARKER_BEGIN}{new_body}{MARKER_END}{tail}", encoding="utf-8")
    return True


def today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def console_table(record: dict[str, Any]) -> str:
    """逐题明细（终端里看哪道题失败了）。

    「引用」列给的是**引用命中**（0/1）而不是引用精度：精度在一道题只引用了 1 条来源时
    会显示 100%，在一条都没引用时显示 0%，两个极端都不是「引对了没有」的答案。
    """
    lines = [f"{'id':<24} {'检索':<4} {'页命中':<7} {'要点':<6} {'引用命中':<8} {'检索耗时':<10}"]
    for item in record["items"]:
        keyword = "—" if item.get("keyword_coverage") is None else f"{item['keyword_coverage']:.0%}"
        cited = item.get("citation_hit")
        citation = "—" if cited is None else ("✓" if cited else "✗")
        lines.append(
            f"{item['id']:<24} {'✓' if item['hit'] else '✗':<4} {item['page_hit']:<7.2f} "
            f"{keyword:<6} {citation:<8} {item['retrieval_ms']:.1f} ms"
        )
    return "\n".join(lines)


def render_readme_section() -> str:
    """README 里该区间的固定说明（与 README 正文保持一致，供文档引用）。"""
    return "\n".join([TABLE_HEADER, TABLE_SEPARATOR])
