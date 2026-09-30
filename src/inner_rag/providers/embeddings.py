"""Embedding 模型构造：把 ``ProviderSpec`` 变成 LangChain 的 Embeddings。

* Ollama 走 ``langchain-ollama``（本地、无需密钥）；
* OpenRouter / OpenAI 都是 OpenAI 兼容的 ``/embeddings`` 接口，统一走
  ``langchain-openai`` 的 OpenAIEmbeddings，只换 ``base_url`` 与 ``model``；
* Mock 为本项目自带的确定性离线向量，用于本地演示与测试；
* 可选按字符数截断超长输入（``EMBEDDING_MAX_INPUT_CHARS``），避免超出小上下文
  模型的窗口（例如 liquid/lfm-2.5-embedding-350m:free 只有 512 token）。

与 chat 一样，Phase 7 起构造器按名字进 ``EMBEDDING_BUILDERS`` 查表，不再有 if 分支分发。
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from loguru import logger
from pydantic import SecretStr

from inner_rag.core.config import settings

if TYPE_CHECKING:  # 只在类型检查时引入，避免 specs <-> embeddings 的 import 环
    from inner_rag.providers.specs import ProviderSpec

_WORD_RE = re.compile(r"[a-z0-9_]+")
_CJK_CHAR_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def _tokens(text: str) -> list[str]:
    """粗粒度分词：英文/数字按词，中文按相邻双字（bigram）。"""
    lowered = text.lower()
    words = _WORD_RE.findall(lowered)
    chars = _CJK_CHAR_RE.findall(lowered)
    bigrams = ["".join(chars[i : i + 2]) for i in range(len(chars) - 1)]
    return words + bigrams


class MockEmbeddings(Embeddings):
    """离线确定性 embedding：哈希词袋 + L2 归一化。

    共享词/字的文本相似度更高，因此 mock provider 下也能完整演示
    「上传 -> 向量检索 -> 引用来源」，且不需要任何网络与密钥。
    注意：它只做到词面重合，不具备真实语义能力，不能用于效果评估。
    """

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim or settings.MOCK_EMBEDDING_DIM

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in _tokens(text):
            digest = hashlib.blake2b(token.encode(), digest_size=4).digest()
            vector[int.from_bytes(digest, "big") % self.dim] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            # 空文本也要给出单位向量，否则 cosine 距离会算出 nan
            vector[0] = 1.0
            return vector
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


class TruncatingEmbeddings(Embeddings):
    """按字符数截断超长输入的包装器。

    分块是按字符数切的，而 embedding 模型的上下文按 token 算。对于上下文很小的
    模型（如 512 token）超长输入会被上游报 400 或静默截断，这里显式截断使行为可预期。
    """

    def __init__(self, inner: Embeddings, max_chars: int) -> None:
        self._inner = inner
        self._max_chars = max_chars

    def _clip(self, text: str) -> str:
        if len(text) <= self._max_chars:
            return text
        logger.debug(f"[EMBEDDING] 输入 {len(text)} 字符超过上限 {self._max_chars}，已截断")
        return text[: self._max_chars]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed_documents([self._clip(text) for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(self._clip(text))

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._inner.aembed_documents([self._clip(text) for text in texts])

    async def aembed_query(self, text: str) -> list[float]:
        return await self._inner.aembed_query(self._clip(text))


def _with_truncation(embeddings: Embeddings) -> Embeddings:
    """按配置给后端套上截断包装（0 表示不截断）。"""
    if settings.EMBEDDING_MAX_INPUT_CHARS > 0:
        return TruncatingEmbeddings(embeddings, settings.EMBEDDING_MAX_INPUT_CHARS)
    return embeddings


class SentenceTransformerEmbeddings(Embeddings):
    """本地（on-premise）embedding：直接在本机跑 sentence-transformers 模型。

    存在的理由：云端网关的免费额度通常按**请求数**限流（例如 1000 次/天），而全量建库
    一次就是几百次请求——反复调优时根本不够用。本地模型没有配额、没有费用，也不依赖网络。

    三个实现要点：

    * **模型懒加载**：构造实例只读配置，不碰权重。否则启动期、健康检查与
      ``/api/system/providers`` 都会去加载几个 GB 的模型（构造器按约定不发请求、不做重活）。
    * **``sentence_transformers`` 只在真正用到时才 import**：它是可选 extra
      （``uv sync --extra local-embed``），默认安装里不存在；放在模块顶层会让
      ``providers.embeddings`` 无法导入，进而把整个 provider 注册表打挂。
    * **编码串行化**：sentence-transformers 的 GPU 推理不是线程安全的，而上层的分批是并发派发的。
      用锁串行化，让 GPU 靠**单次大 batch** 吃满，而不是靠多线程抢。
    """

    def __init__(self) -> None:
        self.model_name = settings.SENTENCE_TRANSFORMERS_MODEL
        self._device = settings.SENTENCE_TRANSFORMERS_DEVICE.strip().lower()
        self._batch_size = max(1, settings.SENTENCE_TRANSFORMERS_BATCH_SIZE)
        self._normalize = settings.SENTENCE_TRANSFORMERS_NORMALIZE
        self._lock = threading.Lock()
        self._model: Any = None
        self._resolved_device: str | None = None

    @property
    def device(self) -> str:
        """实际使用的设备（``auto`` 会被解析成 ``cuda`` / ``cpu``），只解析一次。"""
        if self._resolved_device is None:
            self._resolved_device = self._resolve_device()
        return self._resolved_device

    def _resolve_device(self) -> str:
        """把配置解析成实际能用的设备。

        请求了 ``cuda`` 却不可用时**退回 CPU 并明确告警**，而不是直接抛错——CPU 只是慢，
        任务还能跑完（「任何单点问题都不该让整体流程失败」）。但告警必须写清成因，因为这条
        最容易变成「跑得慢但没人知道为什么」：

        * 装的是 CPU 版 torch（``torch.version.cuda is None``）；
        * **torch 的 CUDA 构建比宿主机驱动支持的版本新**——实测案例：驱动 535（CUDA 12.2）
          配上 PyPI 默认的 cu13x 构建，``torch.cuda.is_available()`` 会是 ``False``，
          而 PyTorch 只在 stderr 上打一次 UserWarning，很容易被日志淹掉。
        """
        requested = self._device or "auto"
        if requested == "cpu":
            return "cpu"
        try:
            import torch
        except ImportError:  # pragma: no cover - 依赖缺失时由 _load 给出更完整的报错
            return "cpu"

        if torch.cuda.is_available():
            return "cuda" if requested == "auto" else requested

        built_for = getattr(torch.version, "cuda", None) or "无（CPU 版 torch）"
        logger.warning(
            f"[EMBEDDING] CUDA 不可用，回退到 CPU（本地嵌入会明显变慢）。"
            f"请求的 device={requested}，torch 的 CUDA 构建={built_for}；"
            "若是构建比驱动新，需换装匹配的 CUDA 构建"
            "（见 docs/configuration.md 3.1；torch 由 pyproject.toml 的 pytorch-cu124 索引决定）"
        )
        return "cpu"

    def _load(self) -> Any:
        """加载（并缓存）模型；依赖缺失时给出「装哪个 extra」的明确报错。"""
        if self._model is not None:
            return self._model
        # 必须在 import huggingface_hub 之前落到环境里：它在导入时把端点读成模块级常量，
        # 之后再设环境变量不生效（这也是为什么 .env 里的 HF_ENDPOINT 要在代码里代为导出）。
        if settings.HF_ENDPOINT:
            os.environ.setdefault("HF_ENDPOINT", settings.HF_ENDPOINT)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - 依赖齐全时不触发
            # 函数内 import：specs 在模块级 import 本模块，模块级反向 import 会成环
            from inner_rag.providers.specs import ProviderError

            raise ProviderError(
                "本地 embedding 需要可选依赖 sentence-transformers 与 torch，"
                "请执行 `uv sync --extra local-embed` 后重试"
            ) from exc

        device = self.device
        logger.info(
            f"[EMBEDDING] 加载本地模型 {self.model_name} device={device} "
            f"batch={self._batch_size} normalize={self._normalize}"
        )
        self._model = SentenceTransformer(
            self.model_name,
            device=device,
            # 允许联网时才让上游兜底下载；离线环境置 False 可以避免「本地模型」变成隐式下载
            local_files_only=not settings.SENTENCE_TRANSFORMERS_ALLOW_DOWNLOAD,
        )
        return self._model

    def _encode(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        extra: dict[str, Any] = {}
        # 部分检索型模型（Qwen3-Embedding 系列）给「查询侧」配了 instruction 前缀，
        # sentence-transformers 会以 prompt_name="query" 暴露出来。有就用，没有就按普通文本编码——
        # 不硬编码模型名，换模型时不需要改代码。
        if is_query and "query" in (getattr(model, "prompts", None) or {}):
            extra["prompt_name"] = "query"
        with self._lock:
            vectors = model.encode(
                texts,
                batch_size=self._batch_size,
                normalize_embeddings=self._normalize,
                # 只取向量，不过滤长度：截断交给模型的 max_seq_length，那里是按 token 精确切的
                show_progress_bar=False,
                convert_to_numpy=True,
                **extra,
            )
        return [list(map(float, vector)) for vector in vectors]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text], is_query=True)[0]


# ── 各后端的构造器（不发起任何网络请求）────────────────────────────────


def build_mock_embeddings(spec: ProviderSpec) -> Embeddings:
    return _with_truncation(MockEmbeddings())


def build_ollama_embeddings(spec: ProviderSpec) -> Embeddings:
    return _with_truncation(OllamaEmbeddings(base_url=spec.base_url, model=spec.model))


def build_openai_compatible_embeddings(spec: ProviderSpec) -> Embeddings:
    """OpenRouter / OpenAI 这类 OpenAI 兼容后端共用的构造器。"""
    return _with_truncation(
        OpenAIEmbeddings(
            base_url=spec.base_url,
            model=spec.model,
            api_key=SecretStr(spec.api_key),
            # 上游不一定是 OpenAI 官方模型，跳过 tiktoken 的上下文长度校验
            check_embedding_ctx_length=False,
            # 默认超时 600s 太长：免费/共享网关抽风时会变成「安静地挂十分钟」，
            # 显式设小才能让上层按批重试（见 services/embedding.py 的分批重试）。
            # 写 `timeout=` 而不是字段名 `request_timeout`：pydantic 的 alias 是它，
            # mypy 只认 alias（两者在运行期等价）。
            timeout=settings.EMBEDDING_TIMEOUT,
        )
    )


def build_sentence_transformers_embeddings(spec: ProviderSpec) -> Embeddings:
    """本地 GPU/CPU 后端：模型懒加载，构造器本身不做重活（与其它后端同一约定）。"""
    return _with_truncation(SentenceTransformerEmbeddings())


EMBEDDING_BUILDERS: dict[str, Callable[[ProviderSpec], Embeddings]] = {
    "ollama": build_ollama_embeddings,
    "openrouter": build_openai_compatible_embeddings,
    "openai": build_openai_compatible_embeddings,
    "sentence_transformers": build_sentence_transformers_embeddings,
    "mock": build_mock_embeddings,
}
