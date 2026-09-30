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
import re
from collections.abc import Callable
from typing import TYPE_CHECKING

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
        )
    )


EMBEDDING_BUILDERS: dict[str, Callable[[ProviderSpec], Embeddings]] = {
    "ollama": build_ollama_embeddings,
    "openrouter": build_openai_compatible_embeddings,
    "openai": build_openai_compatible_embeddings,
    "mock": build_mock_embeddings,
}
