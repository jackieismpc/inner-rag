"""评测集与 fixture 的读取、schema 校验与锚点校验。

本模块不依赖 inner_rag，也不要求存在 PDF：缺 PDF 时只做 schema 与 fixture 校验，
这样 CI 与本地无网环境都能跑。有 PDF 时额外做「引用片段必须真的出现在该页」的校验，
避免「评测集自己写错」被误判成模型答错。

**读 PDF 一律走 `PageText`**（pypdf，与入库时的解析器同源）。另有 `gen_eval_fixtures.py`
用它裁离线片段——两边共用同一个类，就不会再出现「校验用的文本」和「入库用的文本」是两回事。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "docs" / "datasets" / "dragon_king" / "eval_v3.jsonl"
DEFAULT_FIXTURES = REPO_ROOT / "docs" / "datasets" / "dragon_king" / "fixtures"
DEFAULT_PDF = REPO_ROOT / "data" / "uploads" / "龙族.pdf"

REQUIRED_FIELDS = (
    "id",
    "question",
    "category",
    "expected_answer",
    "answer_keywords",
    "expect_refusal",
)
VALID_CATEGORIES = {"fact", "alias", "multihop", "negative", "refusal"}


def normalize(text: str) -> str:
    """去掉所有空白，用于「跨换行/排版噪声」的片段比对。"""
    return re.sub(r"\s+", "", text or "")


def load_eval_set(path: Path | str | None = None) -> list[dict[str, Any]]:
    """读取并校验评测集；有问题直接抛 ValueError（附 id 与原因）。"""
    path = Path(path) if path else DEFAULT_DATASET
    if not path.exists():
        msg = f"评测集不存在：{path}"
        raise FileNotFoundError(msg)

    items: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError as exc:
            msg = f"{path}:{line_no} 不是合法 JSON：{exc}"
            raise ValueError(msg) from exc
        _check_item(item, f"{path}:{line_no}")
        items.append(item)

    if not items:
        msg = f"评测集为空：{path}"
        raise ValueError(msg)

    seen: set[str] = set()
    for item in items:
        if item["id"] in seen:
            msg = f"评测集存在重复 id：{item['id']}"
            raise ValueError(msg)
        seen.add(item["id"])
    return items


def _check_item(item: dict[str, Any], where: str) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in item]
    if missing:
        msg = f"{where} 缺少字段 {missing}"
        raise ValueError(msg)
    if item["category"] not in VALID_CATEGORIES:
        msg = f"{where} category={item['category']!r} 不在 {sorted(VALID_CATEGORIES)}"
        raise ValueError(msg)
    if not isinstance(item["answer_keywords"], list):
        msg = f"{where} answer_keywords 必须是列表"
        raise ValueError(msg)
    citations = item.get("citations", [])
    if not isinstance(citations, list):
        msg = f"{where} citations 必须是列表"
        raise ValueError(msg)
    for citation in citations:
        if "page" not in citation or "quote" not in citation:
            msg = f"{where} citation 必须含 page 与 quote：{citation!r}"
            raise ValueError(msg)
    if item["expect_refusal"] and (citations or item["answer_keywords"]):
        msg = f"{where} 负样本不应带 citations / answer_keywords"
        raise ValueError(msg)
    if not item["expect_refusal"] and not citations:
        msg = f"{where} 非负样本必须有 citations 锚点"
        raise ValueError(msg)


def expected_pages(item: dict[str, Any]) -> list[int]:
    return [int(c["page"]) for c in item.get("citations", [])]


def load_fixtures(fixtures_dir: Path | str | None = None) -> list[dict[str, Any]]:
    """读取 fixture 索引与正文；每段必须 < 200 字。"""
    fixtures_dir = Path(fixtures_dir) if fixtures_dir else DEFAULT_FIXTURES
    index_path = fixtures_dir / "index.jsonl"
    if not index_path.exists():
        msg = f"fixture 索引不存在：{index_path}"
        raise FileNotFoundError(msg)

    fixtures: list[dict[str, Any]] = []
    for raw in index_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        entry = json.loads(raw)
        text_path = fixtures_dir / entry["file"]
        text = text_path.read_text(encoding="utf-8").strip()
        if len(text) >= 200:
            msg = f"{entry['fixture_id']} 超过 200 字（{len(text)}）"
            raise ValueError(msg)
        fixtures.append({**entry, "text": text})
    return fixtures


class PageText:
    """按物理页号取 PDF 页文本（带缓存）。

    **提取器必须与 `inner_rag.services.parser` 的 PDF 分支同源**——两边都用 ``pypdf`` 的
    ``extract_text()``。理由是页码的「真值」来自入库分块：分块文本是 pypdf 提取的，页码也是
    那时写下的。这里曾经用 pymupdf 的 ``get_text()``，两套提取器对同一页给出的文本并不相同
    （实测源书 p9：pypdf 222 字 / pymupdf 184 字，断句位置也不同），于是「引用片段必须出现在
    该页原文里」这条校验比对的是**另一份文本**，会把本来正确的题目判成锚点错误，而真正的
    错题反而可能蒙混过关。`tests/test_benchmark_metrics.py` 有一条用例钉住这个一致性。
    """

    def __init__(self, pdf_path: Path | str) -> None:
        from pypdf import PdfReader  # 只有真的要读 PDF 时才需要这个依赖

        self._reader = PdfReader(str(pdf_path))
        self._cache: dict[int, str] = {}

    @property
    def page_count(self) -> int:
        return len(self._reader.pages)

    def __call__(self, page: int) -> str:
        if page not in self._cache:
            if not 1 <= page <= self.page_count:
                msg = f"页码 {page} 超出 1..{self.page_count}"
                raise ValueError(msg)
            self._cache[page] = (self._reader.pages[page - 1].extract_text() or "").strip()
        return self._cache[page]


def validate(
    items: list[dict[str, Any]],
    fixtures: list[dict[str, Any]],
    pdf_path: Path | str | None = DEFAULT_PDF,
) -> list[str]:
    """返回问题列表（空列表 = 通过）。缺 PDF 时跳过原文比对。"""
    problems: list[str] = []

    covered = {eid for fixture in fixtures for eid in fixture["eval_ids"]}
    for item in items:
        if not item["expect_refusal"] and item["id"] not in covered:
            problems.append(f"{item['id']}: 没有对应的离线 fixture")

    pdf = Path(pdf_path) if pdf_path else None
    if pdf is None or not pdf.exists():
        problems.append(f"[skip] 未找到 PDF（{pdf}），已跳过原文锚点校验")
        return problems

    reader = PageText(pdf)

    for item in items:
        for citation in item.get("citations", []):
            page = int(citation["page"])
            try:
                text = reader(page)
            except ValueError as exc:
                problems.append(f"{item['id']}: {exc}")
                continue
            if normalize(citation["quote"]) not in normalize(text):
                problems.append(f"{item['id']}: 引用片段不在 p{page}：{citation['quote'][:24]}…")

    for fixture in fixtures:
        try:
            text = reader(int(fixture["page"]))
        except ValueError as exc:
            problems.append(f"{fixture['fixture_id']}: {exc}")
            continue
        if normalize(fixture["text"]) not in normalize(text):
            problems.append(f"{fixture['fixture_id']}: fixture 不是 p{fixture['page']} 原文子串")
    return problems


def validate_or_raise(items: list[dict[str, Any]], fixtures: list[dict[str, Any]]) -> list[str]:
    """只把「真问题」当失败；缺 PDF 的 skip 提示不算失败。"""
    problems = validate(items, fixtures)
    hard = [p for p in problems if not p.startswith("[skip]")]
    if hard:
        msg = "评测集校验失败：\n  - " + "\n  - ".join(hard)
        raise ValueError(msg)
    return problems
