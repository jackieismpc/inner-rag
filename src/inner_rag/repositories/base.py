"""关系库仓储契约（Ports）。

为什么要有这一层：SQLAlchemy 会话一旦泄漏到业务代码，换关系库（SQLite → PostgreSQL、
或换成别的 ORM / 微服务）就得改所有调用点；同时「同一个查询被复制到三个地方」也是 bug 温床
（比如「access 到底按 owner 还是 member」）。本文件只声明**接口**，实现见 `sqlalchemy.py`。

三条约定：

1. **仓储是访问关系库的唯一入口**：`api/`、`services/`、`scripts/` 都不再自己拼查询；
2. **事务边界写在仓储里**：单实体写入（create / update / delete / 状态迁移）各自提交；
   批量写入用 `add_many` 一次提交（要么全在、要么全不在）；`sync_kb_doc_count` 把「文档
   完成」与「库计数更新」放在同一次提交里，避免出现「已完成但计数没变」的中间态；
3. **返回 ORM 实体**（不是 DTO）：实体在此层是数据契约，跨层传递不会泄漏 SQL 细节；
   调用方拿到的对象在 `expire_on_commit=False` 下仍可用（见 `core/database.py`）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from inner_rag.models import (
    Conversation,
    DocStatus,
    Document,
    KBMember,
    KBPermission,
    KnowledgeBase,
    Message,
    User,
)


@dataclass(frozen=True)
class NewDocument:
    """新建文档的入参（`add_many` 一次提交一批）。"""

    kb_id: int
    filename: str
    file_path: str
    file_type: str
    file_size: int
    source_type: str


class UserRepository(Protocol):
    """用户表**读取**（登录态解析、成员授权时按用户名找人）。

    写入（建号 / 改口令）只在 `scripts/create_user.py`：那是运维动作，需要口令哈希与交互式
    输入，不属于业务链路。
    """

    def get(self, user_id: int) -> User | None: ...

    def find_by_username(self, username: str) -> User | None: ...


class KnowledgeBaseRepository(Protocol):
    """知识库聚合（知识库 + 成员授权）。"""

    def create(
        self,
        *,
        name: str,
        description: str | None,
        icon: str | None,
        embedding_model: str,
        owner_id: int,
    ) -> KnowledgeBase: ...

    def get(self, kb_id: int) -> KnowledgeBase | None: ...

    def list_all(self) -> list[KnowledgeBase]: ...

    def list_page(
        self, *, kb_ids: list[int], keyword: str | None, page: int, page_size: int
    ) -> tuple[list[KnowledgeBase], int]: ...

    def update(self, kb: KnowledgeBase, values: dict[str, Any]) -> KnowledgeBase: ...

    def delete(self, kb: KnowledgeBase) -> None: ...

    def set_doc_count(self, kb_id: int, count: int) -> None: ...

    # ── 成员授权 ───────────────────────────────────────────────────────

    def owned_kb_ids(self, user_id: int, kb_ids: list[int] | None = None) -> set[int]: ...

    def member_permissions(
        self, user_id: int, kb_ids: list[int] | None = None
    ) -> dict[int, KBPermission]: ...

    def get_member(self, kb_id: int, user_id: int) -> KBMember | None: ...

    def list_members(self, kb_id: int) -> list[tuple[KBMember, User]]: ...

    def upsert_member(self, kb_id: int, user_id: int, permission: KBPermission) -> KBMember: ...

    def remove_member(self, kb_id: int, user_id: int) -> bool: ...


class DocumentRepository(Protocol):
    """文档聚合（元数据 + 状态机 + 分块计数）。"""

    def get(self, doc_id: int) -> Document | None: ...

    def list_by_kb(self, kb_id: int) -> list[Document]: ...

    def list_page(
        self,
        *,
        kb_id: int,
        status: str | None,
        keyword: str | None,
        page: int,
        page_size: int,
    ) -> tuple[list[Document], int]: ...

    def add_many(self, items: list[NewDocument]) -> list[int]: ...

    def update_status(
        self, doc: Document, status: DocStatus, error_msg: str | None = None
    ) -> None: ...

    def mark_completed(
        self,
        doc: Document,
        *,
        chunk_count: int,
        char_count: int,
        meta: dict[str, Any],
    ) -> None: ...

    def reset_for_reprocess(self, doc: Document) -> None: ...

    def reset_kb_for_reprocess(self, kb_id: int) -> list[int]: ...

    def count_completed(self, kb_id: int) -> int: ...

    def sync_kb_doc_count(self, kb_id: int) -> int: ...

    def delete(self, doc: Document) -> None: ...


class ConversationRepository(Protocol):
    """会话聚合（会话 + 消息）。"""

    def get(self, conv_id: int) -> Conversation | None: ...

    def create(self, *, kb_id: int, title: str) -> Conversation: ...

    def list_page(
        self, *, kb_id: int, page: int, page_size: int
    ) -> tuple[list[Conversation], int]: ...

    def messages(self, conv_id: int) -> list[Message]: ...

    def history(self, conv_id: int, limit: int = ...) -> list[Message]: ...

    def append_message(
        self, conv_id: int, role: str, content: str, sources: list[dict[str, Any]] | None = None
    ) -> Message: ...

    def touch(self, conv_id: int) -> None: ...

    def delete(self, conv: Conversation) -> None: ...
