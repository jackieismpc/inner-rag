"""embedding 分批的健壮性测试。

入库吞吐与可用性都压在这一层：全库入库是「几万个分块 × 分钟级」的批量任务，
所以这里要钉死四件事——**并发真的有上限**、**批次顺序与输入一致**、**单批失败可重试**、
**硬失败时兄弟批次被取消**。

最后一条最容易被忽略：早期实现让异常直接穿透 ``asyncio.gather``，其余批次仍会在后台继续跑，
调用方重试后两轮请求叠在一起，既打满免费额度，也让「到底跑了多少」无法解释。

末尾一组是本地后端的**设备解析**：GPU 用不上时必须「退回 CPU + 把成因写进告警」，
而不是抛错或悄悄变慢——判别依据是这条实测经历：驱动 535（CUDA 12.2）配上 PyPI 默认的
cu13x 构建，``torch.cuda.is_available()`` 会是 ``False``，而 PyTorch 只在 stderr 上打一次
UserWarning，日志一多就淹没了。
"""

from __future__ import annotations

import asyncio
import sys
import types
from collections.abc import Iterator
from typing import Any

import pytest
from loguru import logger

from inner_rag.core.config import settings
from inner_rag.providers.embeddings import (
    SentenceTransformerEmbeddings,
    build_openai_compatible_embeddings,
    build_sentence_transformers_embeddings,
)
from inner_rag.providers.specs import ProviderSpec
from inner_rag.services.embedding import EmbeddingService


class _FakeEmbeddings:
    """可编程的假后端：按「批首文本」记账，因此重试与首次调用能被区分开。"""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []
        self.attempts: dict[str, int] = {}
        self.inflight = 0
        self.max_inflight = 0
        self.fail_once: set[str] = set()  # 这些批的第一次调用失败
        self.always_fail: str | None = None  # 这个批每次都失败

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        key = texts[0]
        self.calls.append(list(texts))
        self.attempts[key] = self.attempts.get(key, 0) + 1
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            await asyncio.sleep(0.01)  # 让并发有机会真正重叠
            if key == self.always_fail:
                msg = "上游持续 500"
                raise RuntimeError(msg)
            if key in self.fail_once and self.attempts[key] == 1:
                msg = "上游偶发 429"
                raise RuntimeError(msg)
            return [
                [float(position), float(len(text))] + [0.0] * (self.dim - 2)
                for position, text in enumerate(texts)
            ]
        finally:
            self.inflight -= 1

    async def aembed_query(self, _text: str) -> list[float]:
        return [0.0] * self.dim


def _service(monkeypatch: pytest.MonkeyPatch, backend: _FakeEmbeddings) -> EmbeddingService:
    service = EmbeddingService()
    service._embeddings = backend  # type: ignore[assignment]
    monkeypatch.setattr(settings, "EMBED_RETRY_BACKOFF", 0.0)
    return service


@pytest.fixture(autouse=True)
def _fast_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "EMBED_BATCH_SIZE", 3)
    monkeypatch.setattr(settings, "EMBEDDING_CONCURRENCY", 2)
    monkeypatch.setattr(settings, "EMBED_MAX_ATTEMPTS", 3)
    monkeypatch.setattr(settings, "EMBED_PROGRESS_EVERY", 100)  # 单测不打进度日志


async def test_batches_keep_input_order(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _FakeEmbeddings()
    service = _service(monkeypatch, backend)
    texts = [f"t{index}" for index in range(10)]

    vectors = await service._embed_in_batches(texts)

    # 批次按输入顺序切分……
    assert backend.calls == [texts[0:3], texts[3:6], texts[6:9], texts[9:10]]
    # ……批内位置也按顺序，因此拼接结果与输入一一对应
    assert [int(vector[0]) for vector in vectors] == [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
    assert len(vectors) == 10


async def test_concurrency_is_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _FakeEmbeddings()
    service = _service(monkeypatch, backend)

    await service._embed_in_batches([f"t{index}" for index in range(12)])

    assert backend.max_inflight <= settings.EMBEDDING_CONCURRENCY


async def test_a_transient_failure_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _FakeEmbeddings()
    backend.fail_once = {"t3"}  # 第二批首元素
    service = _service(monkeypatch, backend)

    vectors = await service._embed_in_batches([f"t{index}" for index in range(6)])

    assert len(vectors) == 6
    assert backend.attempts["t3"] == 2  # 只重试了失败的那一批


async def test_a_permanent_failure_raises_after_exhausting_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeEmbeddings()
    backend.always_fail = "t0"
    service = _service(monkeypatch, backend)

    with pytest.raises(RuntimeError, match="上游持续 500"):
        await service._embed_in_batches([f"t{index}" for index in range(9)])

    assert backend.attempts["t0"] == settings.EMBED_MAX_ATTEMPTS


async def test_sibling_batches_are_cancelled_on_hard_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """一批彻底失败后其余批必须被取消——否则它们在后台继续请求，调用方重试时双倍打账。"""
    backend = _FakeEmbeddings()
    backend.always_fail = "t0"
    service = _service(monkeypatch, backend)

    with pytest.raises(RuntimeError):
        await service._embed_in_batches([f"t{index}" for index in range(30)])

    calls_at_failure = len(backend.calls)
    await asyncio.sleep(0.05)  # 给「本该被取消」的任务一点冒头时间
    assert backend.inflight == 0  # 没有请求悬挂在半空
    assert len(backend.calls) == calls_at_failure  # 也没有任务在后台继续发请求


async def test_empty_input_makes_no_request(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _FakeEmbeddings()
    service = _service(monkeypatch, backend)

    assert await service._embed_in_batches([]) == []
    assert backend.calls == []


async def test_batch_size_is_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _FakeEmbeddings()
    service = _service(monkeypatch, backend)

    await service._embed_in_batches([f"t{index}" for index in range(7)])

    assert sorted(len(call) for call in backend.calls) == [1, 3, 3]


def test_timeout_is_configured_for_openai_compatible_backends() -> None:
    """默认 600s 太长：免费网关抽风时会变成「安静地挂十分钟」，必须显式设小。"""
    assert 0 < settings.EMBEDDING_TIMEOUT <= 120

    embeddings = build_openai_compatible_embeddings(
        ProviderSpec(
            name="openrouter",
            label="OpenRouter",
            kind="embedding",
            model="test-model",
            base_url="https://example.invalid/v1",
            api_key="test-key",
        )
    )
    inner = embeddings
    while hasattr(inner, "_inner"):
        inner = inner._inner  # type: ignore[attr-defined]

    assert inner.request_timeout == settings.EMBEDDING_TIMEOUT  # type: ignore[attr-defined]


# ── 本地后端的设备解析 ────────────────────────────────────────────────


def _fake_torch(*, available: bool, cuda_build: str | None) -> types.ModuleType:
    """只回答两个问题的假 torch：GPU 能不能用、构建时对齐的 CUDA 版本。"""
    module = types.ModuleType("torch")
    module.cuda = types.SimpleNamespace(is_available=lambda: available)
    module.version = types.SimpleNamespace(cuda=cuda_build)
    return module


@pytest.fixture
def warnings() -> Iterator[list[str]]:
    """接住 WARNING 级日志（设备解析的告警是这里唯一要断言的东西）。"""
    lines: list[str] = []
    sink_id = logger.add(
        lambda message: lines.append(str(message)), level="WARNING", format="{message}"
    )
    try:
        yield lines
    finally:
        logger.remove(sink_id)


@pytest.fixture
def _auto_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """钉住配置端：.env 里可能被改过，设备解析的用例必须从 ``auto`` 出发。"""
    monkeypatch.setattr(settings, "SENTENCE_TRANSFORMERS_DEVICE", "auto")


@pytest.mark.usefixtures("_auto_device")
def test_device_auto_picks_cuda_when_available(
    monkeypatch: pytest.MonkeyPatch, warnings: list[str]
) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(available=True, cuda_build="12.4"))

    assert SentenceTransformerEmbeddings().device == "cuda"
    assert warnings == []  # 一切正常时不该产生噪声


@pytest.mark.usefixtures("_auto_device")
def test_device_explicit_cuda_index_is_preserved(
    monkeypatch: pytest.MonkeyPatch, warnings: list[str]
) -> None:
    """多卡机器上 ``cuda:1`` 这类写法要原样传给 sentence-transformers，不能被压成 ``cuda``。"""
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(available=True, cuda_build="12.4"))
    monkeypatch.setattr(settings, "SENTENCE_TRANSFORMERS_DEVICE", "cuda:1")

    assert SentenceTransformerEmbeddings().device == "cuda:1"
    assert warnings == []


def test_device_cpu_request_never_touches_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """显式要 CPU 时不该去探 CUDA——纯 CPU 机器上 ``import torch`` 本身就可能很慢。"""
    monkeypatch.setitem(sys.modules, "torch", None)  # 一 import 就炸
    monkeypatch.setattr(settings, "SENTENCE_TRANSFORMERS_DEVICE", "cpu")

    assert SentenceTransformerEmbeddings().device == "cpu"


@pytest.mark.usefixtures("_auto_device")
def test_device_falls_back_to_cpu_when_the_build_is_newer_than_the_driver(
    monkeypatch: pytest.MonkeyPatch, warnings: list[str]
) -> None:
    """实测场景：驱动 12.2 配 cu13x 构建。退回 CPU 是可以的，但必须说清是「构建比驱动新」。"""
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(available=False, cuda_build="13.0"))

    assert SentenceTransformerEmbeddings().device == "cpu"
    assert len(warnings) == 1
    assert "13.0" in warnings[0] and "回退到 CPU" in warnings[0]


@pytest.mark.usefixtures("_auto_device")
def test_device_falls_back_to_cpu_when_torch_is_cpu_only(
    monkeypatch: pytest.MonkeyPatch, warnings: list[str]
) -> None:
    """另一种成因：装的就是 CPU 版 torch。告警文案要与「构建过新」区分开。"""
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(available=False, cuda_build=None))

    assert SentenceTransformerEmbeddings().device == "cpu"
    assert len(warnings) == 1
    assert "CPU 版 torch" in warnings[0]


@pytest.mark.usefixtures("_auto_device")
def test_device_falls_back_to_cpu_when_torch_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """没装 torch 的机器上构造实例也不能炸：真正要用模型时 ``_load()`` 才报「装哪个 extra」。"""
    monkeypatch.setitem(sys.modules, "torch", None)

    assert SentenceTransformerEmbeddings().device == "cpu"


@pytest.mark.usefixtures("_auto_device")
def test_device_is_resolved_once(monkeypatch: pytest.MonkeyPatch, warnings: list[str]) -> None:
    """解析结果要缓存：否则每取一次 device 就重探一次 CUDA，告警还会重复刷屏。"""
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(available=False, cuda_build="13.0"))
    embeddings = SentenceTransformerEmbeddings()

    assert embeddings.device == "cpu"
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(available=True, cuda_build="13.0"))
    assert embeddings.device == "cpu"  # 仍旧吃第一次的结果
    assert len(warnings) == 1


def test_local_embedding_builder_returns_the_local_backend() -> None:
    """默认嵌入必须落在本地 Qwen 上——云端免费额度按请求数算，全量建库一次就超了。"""
    spec = ProviderSpec(
        name="sentence_transformers",
        label="Local (sentence-transformers)",
        kind="embedding",
        model="Qwen/Qwen3-Embedding-0.6B",
        base_url="",
        api_key="",
    )
    embeddings: Any = build_sentence_transformers_embeddings(spec)
    # EMBEDDING_MAX_INPUT_CHARS > 0 时会套一层 TruncatingEmbeddings，剥掉再看本体
    while hasattr(embeddings, "_inner"):
        embeddings = embeddings._inner  # type: ignore[attr-defined]

    assert isinstance(embeddings, SentenceTransformerEmbeddings)
    assert embeddings.model_name == settings.SENTENCE_TRANSFORMERS_MODEL
    # 构造器按约定不做重活：此时还没加载任何权重
    assert embeddings._model is None
