"""插件注册表：把「内置后端 if 分支」换成「按名字查表」，并允许外部包注册实现。

为什么要有这一层：

* 分支写死在内置模块里，第三方想接一个后端就只能改源码；注册表把「有哪些实现」变成
  运行时可枚举的数据，因此换后端 = 加一个实现 + 改一行配置；
* 报错文案需要「当前可选哪些后端」，注册表天然提供这个列表，不用各处维护常量（也就不会漏改）；
* 测试与替换演练可以临时注册一个假实现（见 `tests/test_plugins.py`），不必改生产代码。

三条约定：

1. **注册在实现模块的 import 期完成**（内置实现由自己所在模块注册，import 即注册，不需要
   「初始化」步骤）。唯一的例外是 provider：`specs.py` 与 `chat.py` / `embeddings.py` 互相引用，
   import 期注册会读到半初始化的模块，因此改为在解析入口惰性注册（见 `specs._ensure_builtins`）；
2. **同名重复注册直接报错**：静默覆盖会让「到底加载了哪个实现」变成谜；
3. **entry point 加载失败只告警并跳过**：第三方包坏了不该让整个服务起不来（降级契约见
   `docs/architecture.md` 第 6 节）；与内置同名时保留内置实现，避免第三方包悄悄换掉内置后端。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models.chat_models import BaseChatModel

    from inner_rag.providers.specs import ProviderSpec
    from inner_rag.services.cache import CacheBackend
    from inner_rag.services.task_queue import TaskQueue
    from inner_rag.services.vector_store.base import VectorStore


class PluginError(ValueError):
    """插件注册表自身的错误：内置实现之间撞名（属于编码错误，启动就该炸）。"""


@dataclass(frozen=True)
class ChatProvider:
    """一个 chat 后端的完整实现：配置 → spec（``spec``），spec → 模型实例（``build``）。

    两者成对注册，是为了让「这个后端读了哪几个环境变量」和「它怎么建客户端」在同一个地方，
    避免只注册一半（能通过探活却建不出实例）。
    """

    spec: Callable[[], ProviderSpec]
    build: Callable[[ProviderSpec], BaseChatModel]


@dataclass(frozen=True)
class EmbeddingProvider:
    """一个 embedding 后端的完整实现，语义同 :class:`ChatProvider`。"""

    spec: Callable[[], ProviderSpec]
    build: Callable[[ProviderSpec], Embeddings]


class Registry[T]:
    """名字 → 实现的注册表。每个插件点一个实例（chat / embedding / vector store / cache / queue）。"""

    def __init__(self, kind: str, group: str) -> None:
        self._kind = kind  # 人类可读的插件点名字，只用于日志与报错
        self._group = group  # entry point 组名，第三方包用它注册
        self._impls: dict[str, T] = {}
        self._entry_points_loaded = False

    @property
    def kind(self) -> str:
        return self._kind

    @property
    def group(self) -> str:
        return self._group

    def register(self, name: str, impl: T) -> None:
        """注册一个实现。同一名字注册两次是编码错误（内置实现撞名），直接抛。"""
        if name in self._impls:
            msg = f"{self._kind} 已注册同名实现: {name!r}"
            raise PluginError(msg)
        self._impls[name] = impl

    def get(self, name: str) -> T | None:
        return self._impls.get(name)

    def names(self) -> tuple[str, ...]:
        """已注册的名字，按注册顺序（= 配置名列表的展示顺序）。"""
        return tuple(self._impls)

    def items(self) -> Iterator[tuple[str, T]]:
        return iter(self._impls.items())

    def __contains__(self, name: object) -> bool:
        return name in self._impls

    def load_entry_points(self) -> list[str]:
        """加载第三方实现（幂等，可被反复调用）。返回本次真正注册进来的名字。"""
        if self._entry_points_loaded:
            return []
        self._entry_points_loaded = True

        loaded: list[str] = []
        for entry in entry_points(group=self._group):
            if entry.name in self._impls:
                logger.warning(
                    f"[PLUGIN] {self._group}:{entry.name} 与内置 {self._kind} 同名，忽略第三方实现"
                )
                continue
            try:
                impl = entry.load()
            except Exception as exc:
                logger.warning(f"[PLUGIN] 加载 {self._group}:{entry.name} 失败，已跳过: {exc}")
                continue
            self._impls[entry.name] = impl
            loaded.append(entry.name)
            logger.info(f"[PLUGIN] 已加载第三方 {self._kind}: {entry.name}")
        return loaded


# 五个插件点。provider 的注册表存的是「spec + builder」成对实现，其余存的是无参工厂函数。
chat_providers: Registry[ChatProvider] = Registry("chat provider", "inner_rag.chat_providers")
embedding_providers: Registry[EmbeddingProvider] = Registry(
    "embedding provider", "inner_rag.embedding_providers"
)
vector_stores: Registry[Callable[[], VectorStore]] = Registry(
    "vector store", "inner_rag.vector_stores"
)
cache_backends: Registry[Callable[[], CacheBackend]] = Registry(
    "cache backend", "inner_rag.cache_backends"
)
task_queues: Registry[Callable[[], TaskQueue]] = Registry("task queue", "inner_rag.task_queues")

ALL_REGISTRIES: tuple[Registry[Any], ...] = (
    chat_providers,
    embedding_providers,
    vector_stores,
    cache_backends,
    task_queues,
)
