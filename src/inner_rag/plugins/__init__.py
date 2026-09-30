"""插件点注册表（可插拔能力的唯一出处）。

每个插件点有「接口 + 内置实现 + 配置项 + 探活 + 契约测试」五件套（见 `docs/architecture.md`
第 2 节）。本包只提供注册表本身；内置实现在各自的实现模块里注册，第三方实现在安装包里声明
entry point（组名见 :mod:`inner_rag.plugins.registry`）后自动被发现。

``plugin_status()`` 把「当前用的是哪个实现、还有哪些可选」汇总成一份可序列化的报告，
供 ``GET /api/system/plugins`` 与 ``/api/system/health`` 使用。
"""

from inner_rag.plugins.registry import (
    ALL_REGISTRIES,
    ChatProvider,
    EmbeddingProvider,
    PluginError,
    Registry,
    cache_backends,
    chat_providers,
    embedding_providers,
    plugin_status,
    task_queues,
    vector_stores,
)

__all__ = [
    "ALL_REGISTRIES",
    "ChatProvider",
    "EmbeddingProvider",
    "PluginError",
    "Registry",
    "cache_backends",
    "chat_providers",
    "embedding_providers",
    "plugin_status",
    "task_queues",
    "vector_stores",
]
