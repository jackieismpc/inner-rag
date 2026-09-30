#!/usr/bin/env python3
"""为评测集里的条目补离线 fixture（每段 < 200 字，随仓库提交）。

为什么需要 fixture：`benchmark.run_bench --mode fixtures` 是**不需要 PDF、不需要网络**的
自检通道（CI 用它验证指标算法），它的语料就是这些片段。`benchmark/dataset.py::validate`
因此要求**每个非负样本都有对应的 fixture**——这条约束是故意的：它逼着「新加一道题」和
「让这道题在离线下也能跑」同时完成，否则评测集会慢慢变成只有本机能验证的东西。

这个脚本把「从源 PDF 里裁出包含引用证据的一小段」自动化：

* 读 PDF 用 `benchmark.dataset.PageText`——**与入库解析器同源（pypdf）**，
  也和 `dataset.validate()` 的锚点校验同一个实现，所以裁出来的片段必然能通过
  「fixture 是页文本子串」这条校验；
* 窗口以引用片段为中心向外扩，长度上限 `--limit-chars`（默认 199，函数内部再留 1 字余量）；
* 已存在的 fixture 不重写（可重复执行，只补缺口）。

用法（在服务器上，PDF 在那边）：

```bash
uv run --extra local-embed python -u scripts/gen_eval_fixtures.py \
    --dataset docs/datasets/dragon_king/eval_v3.jsonl \
    --pdf data/uploads/龙族.pdf
```
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark import dataset as ds  # noqa: E402  （必须先补 sys.path）

DEFAULT_DATASET = ds.DEFAULT_DATASET
DEFAULT_FIXTURES = ds.DEFAULT_FIXTURES
DEFAULT_PDF = ds.DEFAULT_PDF
# fixture 正文上限（benchmark/dataset.py 的 load_fixtures 会在 >=200 时报错）
MAX_FIXTURE_CHARS = 199


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def read_items(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def read_index(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def excerpt_around(page_text: str, quote: str, limit: int) -> tuple[str | None, str]:
    """裁出包含 ``quote`` 的一小段（< MAX_FIXTURE_CHARS 字），且**必须仍是页文本的子串**。

    返回 ``(片段, 原因)``；片段为 None 时，原因要能区分「引用不在这一页」与「引用本身就超长」——
    前者是题目或页码的问题，后者只是窗口上限的问题，处理动作完全不同。

    实现要点：先在「去掉空白」的文本上定位引用，再借助映射回到原文本的下标——
    直接对原文本做子串匹配会因换行/缩进而失败，而返回去空白的文本又会破坏
    「fixture 是原文子串」这条校验。

    **上限必须留余量**（``limit`` 收敛到 ``MAX_FIXTURE_CHARS - 1``）：早期版本没有这层收敛，
    循环把窗口扩到正好 199 字，紧跟着又被 ``len(text) >= 199`` 自己否掉，于是**只要页文本
    长于 199 字就一律裁不出 fixture**——而源书几乎没有这么短的页。这个 bug 的失败长相很像
    「这几条题定位不到」，所以拖到评测集扩容才暴露（原来是 25 条恰好都没超上限）。
    """
    hard_limit = min(limit, MAX_FIXTURE_CHARS - 1)
    flat = normalize(page_text)
    target = normalize(quote)
    if not target:
        return None, "引用片段为空"
    position = flat.find(target)
    if position == -1:
        return None, "引用片段不在这一页（页码或提取器不一致）"

    # flat 下标 -> 原文本下标（跳过所有空白字符）
    mapping: list[int] = []
    for index, char in enumerate(page_text):
        if not char.isspace():
            mapping.append(index)
    if position + len(target) > len(mapping):
        return None, "引用片段超出页文本范围"

    start = mapping[position]
    end = mapping[position + len(target) - 1] + 1
    if end - start >= hard_limit:
        return None, f"引用片段本身 {end - start} 字 ≥ 上限 {hard_limit}"

    while end - start < hard_limit:
        grew = False
        if end < len(page_text):
            end = min(len(page_text), end + 1)
            grew = True
        if end - start < hard_limit and start > 0:
            start = max(0, start - 1)
            grew = True
        if not grew:
            break

    text = page_text[start:end].strip()
    if len(text) >= MAX_FIXTURE_CHARS:
        return None, f"裁出的片段仍有 {len(text)} 字 ≥ 上限 {MAX_FIXTURE_CHARS}"
    return text, ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="为评测集补离线 fixture")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--fixtures-dir", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--limit-chars", type=int, default=MAX_FIXTURE_CHARS)
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="重建 index.jsonl（保留磁盘上真实的 fixture 文件），用于索引与文件不一致时收口",
    )
    args = parser.parse_args(argv)

    if not args.pdf.exists():
        msg = f"源 PDF 不存在：{args.pdf}"
        raise SystemExit(msg)

    items = read_items(args.dataset)
    args.fixtures_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.fixtures_dir / "index.jsonl"

    entries = read_index(index_path)
    if args.rebuild_index:
        # 只保留磁盘上确实存在的 fixture 文件对应的条目（索引与文件不一致时收口）
        entries = [entry for entry in entries if (args.fixtures_dir / entry["file"]).exists()]
    covered = {eval_id for entry in entries for eval_id in entry["eval_ids"]}
    existing_ids = {entry["fixture_id"] for entry in entries}

    todo = [
        item
        for item in items
        if not item["expect_refusal"] and item["id"] not in covered and item.get("citations")
    ]
    if not todo:
        print(f"[fixtures] 无需补齐（{len(items)} 条题 / 已覆盖 {len(covered)} 条）")
        return 0

    # 与 dataset.validate() 共用同一个 PageText：裁出来的片段必然通过锚点校验
    reader = ds.PageText(args.pdf)
    added = 0
    skipped: list[str] = []
    for item in todo:
        citation = item["citations"][0]
        page = int(citation["page"])
        if not 1 <= page <= reader.page_count:
            skipped.append(f"{item['id']}: 页码 {page} 越界")
            continue
        page_text = reader(page)
        if not page_text:
            skipped.append(f"{item['id']}: p{page} 提不出文本（入库时可能走了 OCR 兜底）")
            continue
        excerpt, reason = excerpt_around(page_text, str(citation["quote"]), args.limit_chars)
        if excerpt is None:
            skipped.append(f"{item['id']}: {reason}（p{page}）")
            continue

        fixture_id = item["id"]
        filename = f"{fixture_id}.txt"
        if fixture_id in existing_ids:
            continue
        (args.fixtures_dir / filename).write_text(excerpt, encoding="utf-8")
        entries.append(
            {
                "fixture_id": fixture_id,
                "page": page,
                "file": filename,
                "eval_ids": [item["id"]],
                "chars": len(excerpt),
            }
        )
        existing_ids.add(fixture_id)
        added += 1

    # 索引按 fixture_id 排序：两份索引可以直接 diff
    entries.sort(key=lambda entry: str(entry["fixture_id"]))
    index_path.write_text(
        "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries), encoding="utf-8"
    )
    print(f"[fixtures] 新增 {added} 条 → {args.fixtures_dir}（索引共 {len(entries)} 条）")
    for reason in skipped:
        print(f"[fixtures] 跳过：{reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
