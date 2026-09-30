#!/usr/bin/env python3
"""从 `data/uploads/龙族.pdf` 构建评测用知识库（小库 / 全库）并产出 manifest。

为什么要有这个脚本：评测结论要可复现，就必须知道「这个库到底是哪几页」。
手工在界面上传一个切好的 PDF 无法回答这个问题——页窗口、embedding identity、
分块参数散落在操作过程里，过两周就没人说得清两次评测的数字能不能比。所以这里把
建库本身做成一次可重放的动作，并把全部上下文写进 manifest。

用法：

```bash
uv run scripts/build_eval_kb.py --profile small --name dragon_king_small --owner admin
uv run scripts/build_eval_kb.py --profile full  --name dragon_king_full  --owner admin
```

**断点续跑**：默认行为是「续跑」——同名知识库已存在时**复用它**（连同已入库的向量），
已入库的分块由向量库自己跳过（契约见 `services/vector_store/base.py`）。
所以入库中途被 429 / 超时 / 进程被杀打断时，**再跑一次同一条命令就行**，不会重算、
也不会重复累积。需要彻底重来时显式加 `--rebuild`。

产物：

* 知识库与文档行（走真实的解析 → 分块 → 嵌入 → 入库链路，不是旁路写向量）；
* `docs/reports/eval-kb-<日期>-<profile>.json`：页窗口、页数、分块数、embedding identity、
  PDF sha256、构建耗时。

manifest 里 `embedding_identity` 与 `embedding_max_input_chars` 必须记录：换 embedding
或改截断长度会让指标不可比（见 `docs/evaluation.md` 2.3）。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# 仓库根目录（scripts/ 的上一级）
REPO_ROOT = Path(__file__).resolve().parent.parent
# 脚本直接执行时 sys.path[0] 是 scripts/，而 benchmark 是仓库顶层目录（未安装成包）
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark import dataset as ds  # noqa: E402

DEFAULT_PDF = REPO_ROOT / "data" / "uploads" / "龙族.pdf"
DEFAULT_REPORT_DIR = REPO_ROOT / "docs" / "reports"
# 解析缓存落在这里（`data/` 已 gitignore：缓存是本机产物，不进版本库）
PARSE_CACHE_DIR = REPO_ROOT / "data" / "cache"
# 小库的子 PDF 落在这里（已 gitignore：大文件不入库）

# 锚点页来自 docs/datasets/dragon_king/eval_v1.jsonl 的 citations，
# 每个窗口是「锚点 ±2 页」：分块不跨页（实测每页约 211 字符 < CHUNK_SIZE），
# 但上下文页能让证据所在的段落不被切断（见 docs/evaluation.md 2.1）。
ANCHOR_WINDOWS: tuple[tuple[int, int], ...] = (
    (116, 126),  # 118 / 123：诺诺真名、昂热
    (329, 335),  # 331：狮心会会长、校长
    (381, 387),  # 383：白王言灵
    (5276, 5282),  # 5278：绘梨衣言灵
    (5902, 5908),  # 5904：Sakura 花名
    (6070, 6083),  # 6072 / 6079：高天原工作
    (6353, 6359),  # 6355：Sakura·路
    (6411, 6417),  # 6413：绘梨衣称呼
)

# 无关章节：用于验证「阈值能挡住不相关内容」。选连续整段而不是随机页，
# 因为真实检索面对的就是连续正文，随机抽样会让噪声失真。
IRRELEVANT_WINDOWS: tuple[tuple[int, int], ...] = (
    (900, 959),
    (3000, 3049),
    (8000, 8049),
)


def resolve_windows(profile: str, total_pages: int | None = None) -> list[tuple[int, int]]:
    """profile -> 页窗口列表（闭区间，1-based 物理页）。

    ``full`` 需要 total_pages（从 PDF 读出）；``small`` 是固定窗口，与页数无关。
    """
    if profile == "small":
        return list(ANCHOR_WINDOWS) + list(IRRELEVANT_WINDOWS)
    if profile == "full":
        if not total_pages:
            msg = "full profile 需要 total_pages"
            raise ValueError(msg)
        return [(1, total_pages)]
    msg = f"未知 profile: {profile}（可选 small / full）"
    raise ValueError(msg)


def window_pages(windows: list[tuple[int, int]]) -> list[int]:
    """窗口展开成去重、升序的页号列表。"""
    pages: set[int] = set()
    for start, end in windows:
        if start > end:
            msg = f"页窗口非法（start > end）: [{start}, {end}]"
            raise ValueError(msg)
        pages.update(range(start, end + 1))
    return sorted(pages)


def sha256_file(path: Path) -> str:
    """大文件分块读，避免一次性占内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ── 解析缓存 ──────────────────────────────────────────────────────────────
#
# 整本书逐页解析一次约 4 分钟，而断点续跑可能要跑好几次（每次重试都要重来一遍解析，
# 而解析跟入库成败毫无关系）。所以把「源文件 → 逐页文本」这一步的结果落盘：
# 缓存键是文件内容的 sha256，因此**换文件必然失效、同一文件必然命中**，
# 不会拿到过期文本。写入走「临时文件 + 原子替换」，写到一半被杀不会留下半份缓存。


def _parse_cache_path(source: Path) -> Path:
    return PARSE_CACHE_DIR / f"parsed-{sha256_file(source)[:16]}.jsonl"


def _read_parse_cache(cache: Path) -> list[Any]:
    from langchain_core.documents import Document

    documents: list[Document] = []
    with cache.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            documents.append(
                Document(
                    page_content=row["text"],
                    metadata={"source": row["source"], "page": int(row["page"])},
                )
            )
    return documents


def _write_parse_cache(cache: Path, documents: list[Any], source: Path) -> None:
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for doc in documents:
            if "page" not in doc.metadata:
                # 非 PDF 解析结果没有页概念，缓存格式以页为单位，直接放弃缓存
                return
            handle.write(
                json.dumps(
                    {
                        "page": int(doc.metadata["page"]),
                        "text": doc.page_content,
                        "source": str(source),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    tmp.replace(cache)


def parse_source(source: Path, *, refresh: bool = False) -> tuple[list[Any], dict[str, Any]]:
    """解析源文件（带磁盘缓存），返回 (逐页段落, 元信息)。

    元信息里的 ``page_count`` 与 ``DocumentParser.parse`` 保持同一口径：都是**去掉空页后**
    的段落数，所以缓存命中与首次解析走的是同一套页号语义（`resolve_windows('full', ...)`
    依赖这个值）。
    """
    from inner_rag.services.parser import DocumentParser

    cache = _parse_cache_path(source)
    if cache.exists() and not refresh:
        documents = _read_parse_cache(cache)
        if documents:
            print(f"[eval-kb] 解析缓存命中 {cache.name}（{len(documents)} 页，跳过重新解析）")
            return documents, _parse_meta(source, documents)

    started = time.perf_counter()
    documents, _ = DocumentParser().parse(str(source), source.name)
    print(f"[eval-kb] 解析完成 {len(documents)} 页，耗时 {time.perf_counter() - started:.1f}s")
    _write_parse_cache(cache, documents, source)
    return documents, _parse_meta(source, documents)


def _parse_meta(source: Path, documents: list[Any]) -> dict[str, Any]:
    # ``page_count`` 是「去空页后的段落数」，``max_page`` 是「最大物理页号」——两者不是一回事。
    # 解析会跳过空页（封面/扉页/无文本页），所以物理页号是稀疏的：段落数 11138，但最大页号可能
    # 是 11165。``resolve_windows('full', ...)`` 需要的是**最大页号**（窗口是页号区间，
    # ``select_pages`` 按物理页号筛），用段落数会漏掉末尾那段页号大于段落数的正文。
    return {
        "filename": source.name,
        "page_count": len(documents),
        "max_page": max((int(doc.metadata.get("page", 0)) for doc in documents), default=0),
        "total_chars": sum(len(doc.page_content) for doc in documents),
    }


def select_pages(documents: list[Any], pages: set[int]) -> list[Any]:
    """从解析结果里挑出指定页（保留**源 PDF 的物理页号**）。

    早期实现先把窗口页抽成一个子 PDF 再走上传链路，结果知识库里存的页码是子 PDF 的
    局部页号（1..N）：引用「第 166 页」在源书里根本翻不到，评测锚点（5904 这类源页号）
    也全部对不上——Recall 直接归零，而回答其实是对的。教训：**页码必须与源文档一致**，
    所以这里不做任何重编号，只做筛选。
    """
    selected = [doc for doc in documents if int(doc.metadata.get("page", 0)) in pages]
    if not selected:
        msg = f"窗口页在解析结果里一页都没命中（共 {len(documents)} 页）"
        raise ValueError(msg)
    return selected


def build_manifest(
    *,
    kb_id: int,
    name: str,
    profile: str,
    source: Path,
    windows: list[tuple[int, int]],
    pages: int,
    chunks: int,
    embedding_identity: str,
    duration_s: float,
    config: dict[str, Any],
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """manifest 内容：字段顺序固定，便于两份 JSON 直接 diff。"""
    return {
        "kb_id": kb_id,
        "name": name,
        "profile": profile,
        "source": str(source.relative_to(REPO_ROOT))
        if source.is_relative_to(REPO_ROOT)
        else str(source),
        "source_sha256": sha256_file(source),
        "page_windows": [[start, end] for start, end in windows],
        "pages": pages,
        "chunks": chunks,
        "chunk_size": config["chunk_size"],
        "chunk_overlap": config["chunk_overlap"],
        "embedding_identity": embedding_identity,
        "embedding_max_input_chars": config["embedding_max_input_chars"],
        "app_version": config["app_version"],
        "git_commit": config["git_commit"],
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "duration_s": round(duration_s, 1),
        "est_cost_usd": 0.0,
        "warnings": warnings or [],
    }


def irrelevant_warnings(source: Path, windows: list[tuple[int, int]]) -> list[str]:
    """无关章节自检：这些页不该出现任何评测题的证据原文。

    用「证据原文（quote）」而不是「答案关键词」判断：主角名字几乎每页都出现，
    拿关键词当判据会把正常正文误报成相关。命中即给出警告——不阻断构建，
    因为「无关」只是为了让召回有噪声，不是正确性前提。
    """
    try:
        items = ds.load_eval_set()
    except Exception:  # 评测集本身有问题时不阻塞建库，交给评测阶段报错
        return []

    quotes = [
        ds.normalize(citation["quote"])
        for item in items
        for citation in item.get("citations", [])
        if citation.get("quote")
    ]
    if not quotes:
        return []

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(source))
    except Exception as exc:  # pragma: no cover - PDF 读不出来是环境问题
        return [f"无法读取 PDF 做无关性自检：{exc}"]

    warnings: list[str] = []
    irrelevant = set(window_pages(list(IRRELEVANT_WINDOWS)))
    for page in sorted(irrelevant):
        try:
            text = ds.normalize(reader.pages[page - 1].extract_text() or "")
        except Exception:  # pragma: no cover - 单页解析失败不值得中断
            continue
        for quote in quotes:
            if quote and quote in text:
                warnings.append(f"无关章节 p{page} 命中评测证据原文：{quote[:20]}…")
    return warnings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建龙族评测知识库")
    parser.add_argument("--profile", choices=("small", "full"), default="small")
    parser.add_argument("--name", default="", help="知识库名称（默认 dragon_king_<profile>）")
    parser.add_argument("--source", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--owner", default="admin", help="知识库 owner 的用户名")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="同名知识库已存在时先删除（含向量）再重建；不加则复用已有库续跑",
    )
    parser.add_argument(
        "--attempts", type=int, default=5, help="单次入库的尝试次数（每次失败会自动续跑）"
    )
    parser.add_argument(
        "--refresh-parse", action="store_true", help="忽略解析缓存，强制重新解析源文件"
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args(argv)


def add_source_document(repos: Any, kb_id: int, source: Path) -> int:
    """建一条文档行并返回 ``doc_id``。

    分块键是 ``(doc_id, chunk_index)``，所以**续跑必须复用同一条文档行**——换一个 doc_id
    会让所有已入库的分块「看起来没写过」，续跑直接失效。
    ``source_type`` 取历史默认值 ``upload``：该字段此前由模型 default 填充，
    显式写出以免新旧 manifest 之间的文档元数据出现无意义差异。
    """
    from inner_rag.repositories import NewDocument

    return repos.docs.add_many(
        [
            NewDocument(
                kb_id=kb_id,
                filename=source.name,
                file_path=str(source),
                file_type="pdf",
                file_size=source.stat().st_size,
                source_type="upload",
            )
        ]
    )[0]


async def build(args: argparse.Namespace) -> dict[str, Any]:
    from inner_rag.core.config import settings
    from inner_rag.core.database import SessionLocal
    from inner_rag.repositories import build_repositories
    from inner_rag.services.vector_store import vector_service

    source = args.source
    if not source.exists():
        msg = f"源 PDF 不存在: {source}"
        raise SystemExit(msg)

    name = args.name or f"dragon_king_{args.profile}"
    started = time.perf_counter()

    # 用真实解析器逐页解析源 PDF：页码就是源文档的物理页，引用与评测锚点共用同一套编号。
    # 代价是要把整本书读一遍（约 4 分钟），换来的是「库里的页码能直接翻到原文」；
    # 结果按文件 sha256 缓存到 data/cache/，续跑时这一步会直接跳过。
    parsed, meta = parse_source(source, refresh=args.refresh_parse)
    # full 窗口要用「最大物理页号」而不是段落数：空页被解析跳过，页号是稀疏的，
    # 用段落数会把末尾「页号 > 段落数」的正文页漏掉（实测龙族 11138 段但最大页号 11165）。
    total_pages = int(meta.get("max_page") or meta["page_count"])
    windows = resolve_windows(args.profile, total_pages)
    pages = window_pages(windows)
    print(
        f"[eval-kb] profile={args.profile} 源共 {meta['page_count']} 页，"
        f"窗口 {len(windows)} 段 / {len(pages)} 页"
    )
    selected = select_pages(parsed, set(pages))
    chars = sum(len(doc.page_content) for doc in selected)
    print(f"[eval-kb] 选中 {len(selected)} 页 / {chars} 字符")

    with SessionLocal() as db:
        repos = build_repositories(db)
        owner = repos.users.find_by_username(args.owner)
        if owner is None:
            raise SystemExit(f"用户 {args.owner} 不存在（用 --owner 指定一个已有账号）")
        # 同名库检查：知识库是「用户级」量级，脚本里取回全量再筛，不为此扩仓储契约
        existing = next((kb for kb in repos.kbs.list_all() if kb.name == name), None)

        if existing is not None and args.rebuild:
            print(f"[eval-kb] --rebuild：删除旧库 id={existing.id}（含全部向量）")
            await vector_service.delete_kb(existing.id)
            repos.kbs.delete(existing)
            existing = None

        if existing is None:
            kb = repos.kbs.create(
                name=name,
                description=f"龙族评测库（{args.profile}），由 scripts/build_eval_kb.py 构建",
                icon=None,
                embedding_model=settings.embedding_key,
                owner_id=owner.id,
            )
            doc_id = add_source_document(repos, kb.id, source)
            kb_id = kb.id
        else:
            # 续跑：复用知识库与文档行，**不删向量**——已入库的分块就是进度本身
            kb_id = existing.id
            reused = next(
                (doc for doc in repos.docs.list_by_kb(kb_id) if doc.filename == source.name), None
            )
            doc_id = reused.id if reused is not None else add_source_document(repos, kb_id, source)
            print(
                f"[eval-kb] 续跑：复用知识库 id={kb_id} / 文档 id={doc_id}，"
                f"库内已有 {vector_service.count(kb_id)} 个分块"
            )

    # 入库链路：分块 → 嵌入 → 写向量（不经 HTTP 上传，因为窗口页不是一份独立文件）。
    # 失败处理是两层的：单批失败由 embedding_service 退避重试（EMBED_MAX_ATTEMPTS），
    # 整份文档中断则由这里重试——而重试之所以安全，是因为向量库按 (doc_id, chunk_index)
    # 跳过已入库分块，重跑等于续跑，不会重算也不会累积重复。
    for attempt in range(1, max(1, args.attempts) + 1):
        try:
            written = await vector_service.add_documents(
                kb_id=kb_id, documents=selected, doc_id=doc_id, filename=source.name
            )
            print(f"[eval-kb] 入库完成 kb_id={kb_id} doc_id={doc_id} 本次新写入={written}")
            break
        except Exception as exc:
            in_store = vector_service.count(kb_id)
            if attempt == max(1, args.attempts):
                raise SystemExit(
                    f"入库尝试 {attempt} 次仍未完成（库内已有 {in_store} 个分块）。"
                    f"进度已保存在向量库里：再次运行本脚本（不加 --rebuild）即可续跑"
                ) from exc
            print(
                f"[eval-kb] 入库第 {attempt}/{args.attempts} 次中断：{str(exc)[:120]}"
                f"（库内已有 {in_store} 个分块，退避后续跑）"
            )
            time.sleep(5 * attempt)

    # 分块数以**向量库实际条数**为准，而不是 add_documents 的返回值：
    # 续跑时返回值只含「本次新写入」的部分，直接拿去写 manifest 会少算。
    chunks = vector_service.count(kb_id)
    print(f"[eval-kb] 库内分块数={chunks}（源窗口 {len(selected)} 页）")
    if chunks == 0:
        raise SystemExit("入库后库内仍为 0 个分块：manifest 不落盘")

    with SessionLocal() as db:
        repos = build_repositories(db)
        document = repos.docs.get(doc_id)
        if document is not None:
            repos.docs.mark_completed(
                document,
                chunk_count=chunks,
                char_count=chars,
                meta={
                    "page_windows": [[start, end] for start, end in windows],
                    "source_pages": len(selected),
                    "built_by": "scripts/build_eval_kb.py",
                },
            )
            repos.docs.sync_kb_doc_count(kb_id)

    manifest = build_manifest(
        kb_id=kb_id,
        name=name,
        profile=args.profile,
        source=source,
        windows=windows,
        pages=len(selected),
        chunks=chunks,
        embedding_identity=settings.embedding_key,
        duration_s=time.perf_counter() - started,
        config={
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "embedding_max_input_chars": settings.EMBEDDING_MAX_INPUT_CHARS,
            "app_version": settings.APP_VERSION,
            "git_commit": _git_commit(),
        },
        warnings=irrelevant_warnings(source, windows) if args.profile == "small" else [],
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"eval-kb-{datetime.now().strftime('%Y-%m-%d')}-{args.profile}.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[eval-kb] manifest: {out}")
    for warning in manifest["warnings"]:
        print(f"[eval-kb] 警告：{warning}")
    return manifest


def _git_commit() -> str:
    import subprocess

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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = asyncio.run(build(args))
    print(
        f"\n知识库：{manifest['name']}（id={manifest['kb_id']}）\n"
        f"页窗口：{len(manifest['page_windows'])} 段 / {manifest['pages']} 页 / "
        f"{manifest['chunks']} 分块\n"
        f"embedding：{manifest['embedding_identity']}\n"
        f"耗时：{manifest['duration_s']}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
