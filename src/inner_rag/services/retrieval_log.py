"""检索日志、Prompt 日志与命中率统计。"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

from loguru import logger

from inner_rag.core.config import settings

_retrieval_logger: logging.Logger | None = None
_prompt_logger: logging.Logger | None = None


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def setup_rag_loggers() -> None:
    """在应用 lifespan 中调用一次，初始化独立日志文件。"""
    global _retrieval_logger, _prompt_logger

    _retrieval_logger = logging.getLogger("rag.retrieval")
    _retrieval_logger.setLevel(logging.INFO)
    if not _retrieval_logger.handlers:
        handler = logging.FileHandler(f"{settings.LOG_DIR}/retrieval.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        _retrieval_logger.addHandler(handler)

    _prompt_logger = logging.getLogger("rag.prompt")
    _prompt_logger.setLevel(logging.INFO)
    if not _prompt_logger.handlers:
        handler = logging.FileHandler(f"{settings.LOG_DIR}/prompt.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        _prompt_logger.addHandler(handler)


class RetrievalStats:
    """内存中的检索质量统计（重启清零，后续可替换为 Redis 持久化）。"""

    _total: int = 0
    _cache_hits: int = 0
    _empty_results: int = 0
    _score_filtered: int = 0
    _total_latency_ms: float = 0.0

    @classmethod
    def record(
        cls, cache_hit: bool, result_count: int, filtered_count: int, latency_ms: float
    ) -> None:
        cls._total += 1
        if cache_hit:
            cls._cache_hits += 1
        if result_count == 0:
            cls._empty_results += 1
        cls._score_filtered += filtered_count
        cls._total_latency_ms += latency_ms

    @classmethod
    def summary(cls) -> dict[str, float]:
        total = cls._total or 1
        return {
            "total_queries": cls._total,
            "cache_hit_rate": round(cls._cache_hits / total, 4),
            "empty_result_rate": round(cls._empty_results / total, 4),
            "avg_filtered_chunks": round(cls._score_filtered / total, 2),
            "avg_latency_ms": round(cls._total_latency_ms / total, 2),
        }


def log_retrieval(
    kb_id: int,
    query: str,
    results: list,
    latency_ms: float,
    cache_hit: bool,
    strategy: str,
    filtered_count: int = 0,
) -> None:
    if not settings.LOG_RETRIEVAL:
        return

    scores = [score for _, score in results[:3] if score is not None]
    record = {
        "ts": _utc_now_iso(),
        "kb_id": kb_id,
        "query": query[:200],
        "strategy": strategy,
        "cache_hit": cache_hit,
        "result_count": len(results),
        "filtered_out": filtered_count,
        "latency_ms": round(latency_ms, 2),
        "top_scores": [round(float(s), 4) for s in scores],
        "top_files": list({r.metadata.get("filename", "?") for r, _ in results[:3]})
        if results
        else [],
    }
    if _retrieval_logger:
        _retrieval_logger.info(json.dumps(record, ensure_ascii=False))

    hit_tag = "CACHE_HIT" if cache_hit else "VECTOR_SEARCH"
    logger.info(
        f"[RETRIEVAL] {hit_tag} | kb={kb_id} | strategy={strategy} | "
        f"results={len(results)} | filtered={filtered_count} | "
        f"latency={latency_ms:.1f}ms | query={query[:50]!r}"
    )

    RetrievalStats.record(cache_hit, len(results), filtered_count, latency_ms)


def log_prompt(
    kb_id: int,
    question: str,
    context_length: int,
    history_turns: int,
    prompt_tokens_est: int,
) -> None:
    if not settings.LOG_PROMPT:
        return

    record = {
        "ts": _utc_now_iso(),
        "kb_id": kb_id,
        "question": question[:200],
        "context_length": context_length,
        "history_turns": history_turns,
        "prompt_tokens_est": prompt_tokens_est,
    }
    if _prompt_logger:
        _prompt_logger.info(json.dumps(record, ensure_ascii=False))

    logger.info(
        f"[PROMPT] kb={kb_id} | ctx_len={context_length} | "
        f"history={history_turns}turns | est_tokens={prompt_tokens_est} | "
        f"question={question[:50]!r}"
    )
