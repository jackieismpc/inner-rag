"""进程内指标：计数器 + 带分位数的直方图。

为什么是进程内（以及它的边界）：本阶段要回答的是「这次改动有没有让检索变慢、空召回变多」，
单进程累计量就够用。多副本部署时每个副本各记一份——这不是 bug，但看板必须按实例聚合，
`README` 与 `docs/observability.md` 都写明了这一点；接 Prometheus 远程写是 Phase 10 的事。

直方图为什么用**有界蓄水池**而不是保存全量样本：分位数只关心尾部分布，全量保存会让高 QPS
接口的内存随流量无上限增长。蓄水池满了丢最旧的样本，误差换来的是确定的内存上限。

为什么没有 cost 指标：token 单价属于「计费域」的事实，写死一张没有来源的价格表只会产出
看起来权威、实际是错的数字。成本看板与价格表一起留到 Phase 10。
"""

from __future__ import annotations

import math
import threading
from collections import deque
from typing import Any

# 每个「指标名 + 标签组合」保留的样本数：p99 在万级样本上误差可接受
RESERVOIR_SIZE = 10_000

# 指标说明（Prometheus 的 # HELP）：只给对外暴露的指标写，避免导出面无限膨胀
DESCRIPTIONS: dict[str, str] = {
    "rag_requests_total": "HTTP 请求数（按端点与状态码）",
    "rag_request_duration_ms": "HTTP 端到端耗时（毫秒）",
    "rag_retrieve_duration_ms": "检索耗时（毫秒，按策略）",
    "rag_retrieve_empty_total": "空召回次数（阈值过高或 embedding 截断的信号）",
    "rag_retrieve_filtered_total": "被相关度阈值滤掉的分块数",
    "rag_cache_hits_total": "缓存命中次数（按命名空间）",
    "rag_llm_ttfb_ms": "流式首 token 延迟（毫秒，按 provider）",
    "rag_llm_tokens_total": "模型 token 用量（按 provider 与 prompt/completion）",
    "rag_ingest_documents_total": "文档入库结果（按状态）",
    "tracing_errors_total": "追踪上报失败次数（失败只计数，不影响请求）",
}

Labels = dict[str, str]
_KEY = tuple[str, tuple[tuple[str, str], ...]]


def _key(name: str, labels: Labels | None) -> _KEY:
    return (name, tuple(sorted((labels or {}).items())))


def _render(key: _KEY) -> str:
    """渲染成 `name{label="value",...}`：JSON 快照与 Prometheus 共用一套命名。"""
    name, labels = key
    if not labels:
        return name
    rendered = ",".join(f'{label}="{value}"' for label, value in labels)
    return f"{name}{{{rendered}}}"


def _quantile(ordered: list[float], quantile: float) -> float:
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return round(ordered[index], 3)


def _summarize(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "count": len(ordered),
        "sum": round(sum(ordered), 3),
        "p50": _quantile(ordered, 0.50),
        "p95": _quantile(ordered, 0.95),
        "p99": _quantile(ordered, 0.99),
    }


class MetricsRegistry:
    """指标注册表：counter 与 histogram 各一张表，用锁保护（后台任务与线程都会写）。"""

    def __init__(self, reservoir_size: int = RESERVOIR_SIZE) -> None:
        self._reservoir_size = reservoir_size
        self._lock = threading.Lock()
        self._counters: dict[_KEY, float] = {}
        self._histograms: dict[_KEY, deque[float]] = {}

    # ── 写入 ──────────────────────────────────────────────────────────

    def increment(self, name: str, value: float = 1.0, *, labels: Labels | None = None) -> None:
        key = _key(name, labels)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0.0) + value

    def observe(self, name: str, value: float, *, labels: Labels | None = None) -> None:
        key = _key(name, labels)
        with self._lock:
            samples = self._histograms.get(key)
            if samples is None:
                samples = deque(maxlen=self._reservoir_size)
                self._histograms[key] = samples
            samples.append(round(value, 3))

    def reset(self) -> None:
        """清零（测试隔离与手动重置用；不对外暴露成接口，避免被误当修复手段）。"""
        with self._lock:
            self._counters.clear()
            self._histograms.clear()

    # ── 读取 ──────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counters = {_render(key): round(value, 6) for key, value in self._counters.items()}
            histograms = {
                _render(key): _summarize(list(samples)) for key, samples in self._histograms.items()
            }
        return {"counters": counters, "histograms": histograms}

    def render_prometheus(self) -> str:
        """导出为 Prometheus 文本格式（counter / summary 两种类型）。"""
        with self._lock:
            counter_items = sorted(self._counters.items())
            histogram_items = sorted(self._histograms.items())

        lines: list[str] = []
        emitted: set[str] = set()

        for key, value in counter_items:
            name = key[0]
            if name not in emitted:
                emitted.add(name)
                lines.append(f"# HELP {name} {DESCRIPTIONS.get(name, name)}")
                lines.append(f"# TYPE {name} counter")
            lines.append(f"{_render(key)} {round(value, 6)}")

        for key, samples in histogram_items:
            name = key[0]
            summary_key = f"{name}_summary"
            if summary_key not in emitted:
                emitted.add(summary_key)
                lines.append(f"# HELP {name} {DESCRIPTIONS.get(name, name)}")
                lines.append(f"# TYPE {name} summary")
            stats = _summarize(list(samples))
            rendered = _render(key)
            for quantile in ("0.5", "0.95", "0.99"):
                field = {"0.5": "p50", "0.95": "p95", "0.99": "p99"}[quantile]
                lines.append(f'{rendered[:-1]},quantile="{quantile}"}} {stats[field]}')
            lines.append(f"{name}_count{rendered[len(name) :]} {stats['count']}")
            lines.append(f"{name}_sum{rendered[len(name) :]} {stats['sum']}")

        return "\n".join(lines) + "\n"


def record_llm_tokens(provider: str, prompt_tokens: int, completion_tokens: int) -> None:
    """token 用量按 prompt / completion 分开记：两者的单价与优化手段都不同。"""
    if prompt_tokens:
        metrics.increment(
            "rag_llm_tokens_total",
            prompt_tokens,
            labels={"provider": provider, "type": "prompt"},
        )
    if completion_tokens:
        metrics.increment(
            "rag_llm_tokens_total",
            completion_tokens,
            labels={"provider": provider, "type": "completion"},
        )


metrics = MetricsRegistry()
