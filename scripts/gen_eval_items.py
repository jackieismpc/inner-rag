#!/usr/bin/env python3
"""从语料正文批量生成评测题（可续跑、带难度护栏）。

为什么要有这个脚本：评测集要能反映**真实检索难度**，前提是它覆盖全书，而不是围着少数几页
反复出题。人工按页出 100 道题成本很高，所以把「读一页 → 出题 → 校验 → 落盘」做成一次
可重放、可续跑的动作：中断了再跑一次就行（已出的页会被跳过）。

产物是一行一题的 JSONL，schema 与 `benchmark/dataset.py` 一致，可直接被
`benchmark.run_bench --dataset <file>` 消费。

三条硬校验（不通过就带着原因重问一次，仍不过则丢弃该页）——它们决定题目**是不是在测检索**：

1. `quote` 必须是该页原文里**逐字存在**的片段：否则这道题的锚点是错的，
   之后「召回是否命中预期页」的判定全都没意义；
2. `answer_keywords` 必须出现在 `expected_answer` 里：否则关键词覆盖率与参考答案对不上；
3. **难度护栏**：问题不得包含答案关键词，也不得与 `quote` 共享连续 6 字以上的片段。
   不设这条的话，出题模型会照着原文抄问法，于是评测退化成「字面匹配测试」——
   指标看起来很漂亮，却看不出任何真正的检索/理解改进。

用法（在服务器上，注意带上 `--extra local-embed` 以免把本地嵌入的依赖清掉）：

```bash
uv run --extra local-embed python -u scripts/gen_eval_items.py \
    --pages data/cache/parsed-<sha16>.jsonl \
    --existing docs/datasets/dragon_king/eval_v2.jsonl \
    --out docs/datasets/dragon_king/eval_v3.jsonl \
    --count 90 --negatives 8
```

`--pages` 文件由 `scripts/build_eval_kb.py` 的解析缓存生成（同名 JSONL，一行一页）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DATASET_DIR = REPO_ROOT / "docs" / "datasets" / "dragon_king"

# 自动生成题的 provenance：写进 notes，报告里能一眼分清「人工出的题」与「语料生成的题」
GENERATED_NOTE = "语料自动生成（scripts/gen_eval_items.py），经 quote/关键词/照抄度三重校验。"

# 问题与 quote 的最长公共片段门槛：低于它才认为「问法不是照抄原文」
MAX_SHARED_RUN = 5
# 太短的页（目录、扉页、空白页残留）出不了好题，直接跳过
MIN_PAGE_CHARS = 160
# 目录 / 扉页的判据：短行占比超过它就不是正文
MAX_SHORT_LINE_RATIO = 0.5

QUESTION_PROMPT = """你在为一部中文长篇小说构建检索评测集。下面是源文档**第 {page} 页**的原文。

请基于这一页出 {count} 道**事实型**问答题。硬性要求：

1. 答案必须**只用这一页**就能确定，不依赖其它页、不依赖常识；
2. 答案要具体可核验（人名 / 称谓 / 物件 / 地点 / 数字 / 动作），不要问「这一页讲了什么」这类开放题；
3. **问题里不要出现答案的关键词**，也不要照抄原文里连续 6 个字以上的片段
   —— 否则这道题在测「字面匹配」而不是「理解」；
4. `quote` 必须是本页原文里**逐字照抄**的一小段（≤40 字），它必须真正支持答案；
5. `answer_keywords` 给 1–2 个「正确回答里必须出现」的词；
6. `category` 取 `fact`（一般事实）或 `alias`（称号 / 绰号 → 人名）。

只输出 JSON 数组，不要任何解释或代码围栏：

[{{"question": "...", "expected_answer": "...", "answer_keywords": ["..."], "quote": "...", "category": "fact"}}]

第 {page} 页原文：
{text}"""

NEGATIVE_PROMPT = """你在为一部中文长篇小说（《龙族》）构建检索评测集的**负样本**。

负样本的用途是检验系统「该拒答时能不能拒答」。它要**看起来和小说有关**（套用小说里的人物 / 组织），
但问的必须是**小说这种文体根本不可能承载的信息**，例如：

- 财务 / 经营数据（净利润、市值、营收）；
- 现实世界的统计或新闻（某年的股价、某国的法律条文）；
- 其它作品的设定（与本书无关的动漫 / 游戏角色关系）；
- 需要外部数据库才能回答的事实（某公司的客服电话）。

请出 {count} 道这样的题。每道题给：
`question`（要套用下面给出的实体，制造词面命中的假象）、
`expected_answer`（说明「语料不含该信息，应当明确拒答，不得凭空作答」）、
`negative_kind`（取 finance / reality / crossover / database 之一）、
`answer_keywords` 固定为空数组。

只输出 JSON 数组，不要任何解释或代码围栏：

[{{"question": "...", "expected_answer": "...", "negative_kind": "finance", "answer_keywords": []}}]

可套用的实体：{entities}"""


# ── 输入输出 ─────────────────────────────────────────────────────────────


def normalize(text: str) -> str:
    """去掉所有空白，用于跨排版噪声的片段比对（与 benchmark/dataset.py 同一口径）。"""
    return re.sub(r"\s+", "", text or "")


def read_pages(path: Path) -> list[dict[str, Any]]:
    """读逐页文本 JSONL（`{"page": n, "text": "..."}`）。"""
    pages: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            pages.append(json.loads(line))
    if not pages:
        msg = f"页文本为空：{path}"
        raise SystemExit(msg)
    return pages


def read_items(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def used_anchor_pages(items: list[dict[str, Any]]) -> set[int]:
    """已被现有评测题占用的锚点页（含前后 3 页）：避免在同一处反复出题。"""
    used: set[int] = set()
    for item in items:
        for citation in item.get("citations", []):
            page = int(citation["page"])
            used.update(range(page - 3, page + 4))
    return used


def is_substantive(text: str) -> bool:
    """这一页够不够出题。

    除了「太短」，还要挡掉**目录 / 扉页**：它们能出的题只有「编号 Ⅱ 的作品叫什么」这种，
    答案就是字面相邻的一行，检索侧怎么都能命中——度量不了检索能力，却会拉高 Recall。
    判据是短行占比（目录页几乎每行都是一个短条目）。
    """
    if len(text) < MIN_PAGE_CHARS:
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    short = sum(1 for line in lines if len(line) <= 6)
    return short / len(lines) <= MAX_SHORT_LINE_RATIO


def select_pages(
    pages: list[dict[str, Any]], *, count: int, seed: int, exclude: set[int]
) -> list[dict[str, Any]]:
    """挑要出题的页：按正文长度过滤后在全书**均匀撒点**，再按种子打散。

    均匀而不是随机：随机抽样会让题目前半本扎堆（长篇小说前段出场人物多、文本更稠），
    指标就会带上「位置偏差」。均匀撒点 + `--count` 控制总量，覆盖面可预期。
    """
    usable = [
        page for page in pages if is_substantive(page["text"]) and int(page["page"]) not in exclude
    ]
    if not usable:
        msg = "没有可用的页（可能全被现有锚点排除了），换一个 --seed 或减少 --count"
        raise SystemExit(msg)
    if count >= len(usable):
        chosen = usable
    else:
        stride = len(usable) / count
        chosen = [usable[min(int(index * stride), len(usable) - 1)] for index in range(count)]
        # 取整可能产生重复，去重后顺延补齐
        seen = {int(page["page"]) for page in chosen}
        for page in usable:
            if len(chosen) >= count:
                break
            if int(page["page"]) not in seen:
                chosen.append(page)
                seen.add(int(page["page"]))
    random.Random(seed).shuffle(chosen)
    return chosen


# ── 校验（难度护栏）──────────────────────────────────────────────────────


def _longest_shared_run(left: str, right: str) -> int:
    """两个字符串的最长公共连续片段长度（题目与原文的「照抄程度」）。"""
    left, right = normalize(left), normalize(right)
    if not left or not right:
        return 0
    best = 0
    previous = [0] * (len(right) + 1)
    for i in range(1, len(left) + 1):
        current = [0] * (len(right) + 1)
        for j in range(1, len(right) + 1):
            if left[i - 1] == right[j - 1]:
                current[j] = previous[j - 1] + 1
                best = max(best, current[j])
        previous = current
    return best


def validate_draft(draft: dict[str, Any], page_text: str) -> str | None:
    """返回不合格的原因（None = 通过）。文案里写清「怎么改」，因为它会被回给模型。"""
    question = str(draft.get("question") or "").strip()
    answer = str(draft.get("expected_answer") or "").strip()
    quote = str(draft.get("quote") or "").strip()
    keywords = draft.get("answer_keywords") or []
    category = str(draft.get("category") or "fact")

    if len(question) < 6:
        return "question 太短"
    if not answer:
        return "expected_answer 为空"
    if category not in {"fact", "alias", "multihop"}:
        return f"category 只允许 fact/alias/multihop，收到 {category}"
    if not isinstance(keywords, list) or not keywords:
        return "answer_keywords 必须是非空列表"
    if quote and normalize(quote) not in normalize(page_text):
        return "quote 不是本页原文里逐字存在的片段，请从原文**原样复制**一小段"

    for keyword in keywords:
        if str(keyword) not in answer:
            return f"answer_keywords 里的「{keyword}」没有出现在 expected_answer 里"
        if str(keyword) in question:
            return f"问题里出现了答案关键词「{keyword}」，这道题会退化成字面匹配测试"

    shared = _longest_shared_run(question, quote or page_text)
    if shared > MAX_SHARED_RUN:
        return f"问题与原文有连续 {shared} 个字相同，请换一种问法（不要照抄原文）"
    return None


def parse_json_array(text: str) -> list[dict[str, Any]]:
    """从模型输出里抠出 JSON 数组：围栏、前后废话、单对象都是常态。"""
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start == -1 or end == -1:
        # 有些模型会漏掉外层方括号，直接给一个对象
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end == -1:
            return []
        cleaned = f"[{cleaned[start : end + 1]}]"
    else:
        cleaned = cleaned[start : end + 1]
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, dict):
        return [parsed]
    return [item for item in parsed if isinstance(item, dict)]


# ── 生成 ─────────────────────────────────────────────────────────────────


TRUNCATED_HINT = (
    "\n\n上一次的输出没有形成完整的 JSON（很可能是被输出长度限制截断了）。"
    "请只输出**一个** JSON 对象，各字段尽量简短，确保是完整合法的 JSON。"
)


async def ask(model: Any, prompt: str, *, attempts: int = 3) -> list[dict[str, Any]]:
    """一次模型调用，失败就换个说法重试。

    两类失败都要能自愈，否则「跑一百页」这种事会随机丢样本：

    * **调用异常**（超时 / 限流）：退避后重试；
    * **输出被截断**（推理模型的思考会吃掉输出配额，实测 `deepseek-flash` 在 2048 tokens
      下经常只吐出半截 JSON）：`parse_json_array` 会解析不出来，这时把「上次被截断了、
      只出一个对象、字段简短」告诉模型再试一次，而不是依赖部署环境把 token 上限调大。
    """
    for attempt in range(1, attempts + 1):
        try:
            message = await model.ainvoke(prompt)
        except Exception as exc:
            if attempt == attempts:
                print(f"    [warn] 模型调用连续失败：{str(exc)[:120]}")
                return []
            await asyncio.sleep(2 * attempt)
            continue

        content = str(getattr(message, "content", ""))
        items = parse_json_array(content)
        if items:
            return items
        print(f"    [warn] 第 {attempt} 次输出解析不出 JSON：{content[:120]!r}")
        prompt = f"{prompt}{TRUNCATED_HINT}"
    return []


async def generate_for_page(
    model: Any, page: int, text: str, *, existing_ids: set[str], per_page: int
) -> list[dict[str, Any]]:
    """一页 → 若干条合格题。校验不过就把原因回给模型重问一次，仍不过则丢弃这一条。"""
    prompt = QUESTION_PROMPT.format(page=page, text=text[:1800], count=per_page)
    drafts = await ask(model, prompt)

    accepted: list[dict[str, Any]] = []
    index = 0
    while index < len(drafts) and len(accepted) < per_page:
        draft = drafts[index]
        problem = validate_draft(draft, text)
        if problem:
            print(f"    [retry] p{page} 第 {index + 1} 条不合格：{problem}")
            repaired = await ask(model, _repair_prompt(prompt, problem, drafts))
            if not repaired:
                break
            drafts = repaired
            draft = repaired[min(index, len(repaired) - 1)]
            problem = validate_draft(draft, text)
            if problem:
                print(f"    [drop] p{page} 重问后仍不合格：{problem}")
                index += 1
                continue

        item_id = f"corpus-p{page}-{index + 1}"
        if item_id not in existing_ids:
            accepted.append(
                {
                    "id": item_id,
                    "question": str(draft["question"]).strip(),
                    "category": str(draft.get("category") or "fact"),
                    "expected_answer": str(draft["expected_answer"]).strip(),
                    "answer_keywords": [str(word) for word in draft["answer_keywords"]],
                    "citations": [{"page": int(page), "quote": str(draft["quote"]).strip()}],
                    "expect_refusal": False,
                    "notes": GENERATED_NOTE,
                }
            )
        index += 1
    return accepted


def _repair_prompt(prompt: str, problem: str, drafts: list[dict[str, Any]]) -> str:
    """把不合格的原因回给模型——只说「不合格」它只会换个说法，说清哪条不过它才会改。"""
    return (
        f"{prompt}\n\n上一次的草稿不合格：{problem}\n"
        f"不合格的草稿：{json.dumps(drafts, ensure_ascii=False)}\n请重新输出整个 JSON 数组。"
    )


async def generate_negatives(
    model: Any, entities: list[str], *, count: int, existing_ids: set[str], offset: int
) -> list[dict[str, Any]]:
    """负样本：套用小说的实体、问小说不可能承载的信息类型。"""
    drafts = await ask(model, NEGATIVE_PROMPT.format(count=count, entities="、".join(entities)))
    accepted: list[dict[str, Any]] = []
    for index, draft in enumerate(drafts):
        question = str(draft.get("question") or "").strip()
        kind = str(draft.get("negative_kind") or "").strip()
        if len(question) < 6 or kind not in {"finance", "reality", "crossover", "database"}:
            print(f"    [drop] 负样本不合格（kind={kind!r}）：{question[:30]}")
            continue
        if not any(entity in question for entity in entities):
            # 不套用小说实体就制造不出「词面命中的假象」，考不到前置门
            print(f"    [drop] 负样本没有套用小说实体：{question[:30]}")
            continue
        item_id = f"corpus-negative-{offset + index + 1}"
        if item_id in existing_ids:
            continue
        accepted.append(
            {
                "id": item_id,
                "question": question,
                "category": "negative",
                "expected_answer": str(
                    draft.get("expected_answer") or "语料不含该信息，应当明确拒答，不得凭空作答。"
                ).strip(),
                "answer_keywords": [],
                "citations": [],
                "expect_refusal": True,
                "notes": f"语料自动生成（负样本 / {kind}）：套用小说实体制造词面命中，考察拒答。",
            }
        )
    return accepted


# ── 主流程 ───────────────────────────────────────────────────────────────


async def run(args: argparse.Namespace) -> int:
    from inner_rag.providers.factory import get_chat_model
    from inner_rag.providers.specs import chat_spec

    pages = read_pages(args.pages)
    existing = read_items(args.existing)
    out_items = read_items(args.out)

    # 续跑：输出文件已有的题 + `--existing` 里的题一起当「已占用」
    seen_ids = {item["id"] for item in existing} | {item["id"] for item in out_items}
    chosen = select_pages(
        pages,
        count=args.count,
        seed=args.seed,
        exclude=used_anchor_pages(existing) | used_anchor_pages(out_items),
    )
    # 已经出过题的页跳过：这就是「再跑一次就续上」的实现
    done_pages = {int(item["citations"][0]["page"]) for item in out_items if item.get("citations")}
    pending = [page for page in chosen if int(page["page"]) not in done_pages]

    model = get_chat_model()
    print(
        f"[gen] 模型 {chat_spec().identity} | 页 {len(pages)} | 计划 {len(chosen)} 页 | "
        f"待出题 {len(pending)} 页 | 已有 {len(out_items)} 条"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # 追加写 + 每页立刻 flush：中断后再跑一次就是续跑，最多丢正在进行的那一页
    with args.out.open("a", encoding="utf-8") as handle:
        for number, page in enumerate(pending, start=1):
            page_no = int(page["page"])
            started = time.perf_counter()
            items = await generate_for_page(
                model,
                page_no,
                page["text"],
                existing_ids=seen_ids,
                per_page=args.per_page,
            )
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                seen_ids.add(item["id"])
            handle.flush()
            print(
                f"[gen] {number}/{len(pending)} p{page_no} → {len(items)} 条 "
                f"({time.perf_counter() - started:.1f}s)"
            )

        if args.negatives:
            entities = _entities(existing + out_items)
            negatives = await generate_negatives(
                model,
                entities,
                count=args.negatives,
                existing_ids=seen_ids,
                offset=len(read_items(args.out)),
            )
            for item in negatives:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[gen] 负样本 → {len(negatives)} 条")

    total = len(read_items(args.out))
    print(f"[gen] 完成 {args.out} 共 {total} 条")
    return 0


def _entities(items: list[dict[str, Any]], limit: int = 12) -> list[str]:
    """从已有题目里挑出小说实体，供负样本套用（候选来自 answer_keywords，都是人名/专名）。"""
    candidates: list[str] = []
    for item in items:
        for word in item.get("answer_keywords") or []:
            if 2 <= len(str(word)) <= 8 and str(word) not in candidates:
                candidates.append(str(word))
    return candidates[:limit]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从语料正文批量生成评测题")
    parser.add_argument("--pages", type=Path, required=True, help="逐页文本 JSONL（解析缓存）")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_DATASET_DIR / "eval_v3.jsonl",
        help="输出评测集 JSONL（追加写，可续跑）",
    )
    parser.add_argument("--existing", type=Path, default=None, help="已有评测集（用于避重）")
    parser.add_argument("--count", type=int, default=90, help="计划出题的页数")
    parser.add_argument("--per-page", type=int, default=1, help="每页出几道题")
    parser.add_argument("--negatives", type=int, default=0, help="额外的负样本条数")
    parser.add_argument("--seed", type=int, default=20260930, help="页选择与打散的随机种子")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(run(parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
