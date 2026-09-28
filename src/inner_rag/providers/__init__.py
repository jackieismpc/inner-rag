"""模型后端抽象层：chat / embeddings 多 provider 统一出口。

业务代码只 import 这里的工厂函数，不直接依赖某家 SDK：

>>> from inner_rag.providers import get_chat_model, get_embeddings

具体用哪家 provider 由 .env 的 ``LLM_PROVIDER`` / ``EMBEDDING_PROVIDER`` 决定，
可选项与参数见 ``inner_rag.providers.specs``。
"""

from inner_rag.providers.factory import (
    chat_health,
    chat_models,
    embedding_health,
    get_chat_model,
    get_embeddings,
    reset_cache,
)
from inner_rag.providers.specs import ProviderError, provider_catalog

__all__ = [
    "ProviderError",
    "chat_health",
    "chat_models",
    "embedding_health",
    "get_chat_model",
    "get_embeddings",
    "provider_catalog",
    "reset_cache",
]
