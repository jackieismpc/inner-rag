"""仓储的 SQLAlchemy 实现（SQLite 与 PostgreSQL 共用一套代码）。

实现风格上的几条取舍：

* 只写「两种方言都成立」的 SQL：不用 SQLite 特有的函数与语法，时间戳统一用 UTC
  （`datetime.now(UTC)`），JSON 字段用 SQLAlchemy 的通用 JSON 类型；
* 查询与写入都显式分页 / 计数，不做「查出来再在 Python 里筛」；
* 每个方法自带提交（见 `base.py` 的事务边界约定），因此没有跨方法的事务泄漏；
* 不吞异常：越权、不存在这类判定由上层（`api/deps.py`）翻译成 HTTP 语义。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

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
from inner_rag.repositories.base import NewDocument

MESSAGE_HISTORY_LIMIT_DEFAULT = 20


class SqlUserRepository:
    """用户表读取（写入只在 scripts/create_user.py）。"""

    def __init__(self, db: Session) -> None:
        self._db = db

    def get(self, user_id: int) -> User | None:
        return self._db.get(User, user_id)

    def find_by_username(self, username: str) -> User | None:
        return self._db.query(User).filter(User.username == username).one_or_none()


class SqlKnowledgeBaseRepository:
    """知识库聚合的 SQLAlchemy 实现。"""

    def __init__(self, db: Session) -> None:
        self._db = db

    # ── 知识库 ─────────────────────────────────────────────────────────

    def create(
        self,
        *,
        name: str,
        description: str | None,
        icon: str | None,
        embedding_model: str,
        owner_id: int,
    ) -> KnowledgeBase:
        kb = KnowledgeBase(
            name=name,
            description=description,
            icon=icon,
            embedding_model=embedding_model,
            owner_id=owner_id,
        )
        self._db.add(kb)
        self._db.commit()
        self._db.refresh(kb)
        return kb

    def get(self, kb_id: int) -> KnowledgeBase | None:
        return self._db.get(KnowledgeBase, kb_id)

    def list_all(self) -> list[KnowledgeBase]:
        """全部知识库（按 id 升序）：诊断脚本与运维用，业务接口一律走 `list_page`。"""
        return self._db.query(KnowledgeBase).order_by(KnowledgeBase.id.asc()).all()

    def list_page(
        self, *, kb_ids: list[int], keyword: str | None, page: int, page_size: int
    ) -> tuple[list[KnowledgeBase], int]:
        """按「可见 id 列表」过滤后再分页：过滤发生在 SQL 里，不是查完再筛。"""
        query = self._db.query(KnowledgeBase).filter(KnowledgeBase.id.in_(kb_ids))
        if keyword:
            query = query.filter(KnowledgeBase.name.like(f"%{keyword}%"))
        total = query.count()
        items = (
            query.order_by(KnowledgeBase.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
            .all()
        )
        return items, total

    def update(self, kb: KnowledgeBase, values: dict[str, Any]) -> KnowledgeBase:
        for key, value in values.items():
            setattr(kb, key, value)
        self._db.commit()
        self._db.refresh(kb)
        return kb

    def delete(self, kb: KnowledgeBase) -> None:
        # 成员授权由外键级联删除（SQLite 需要 PRAGMA foreign_keys=ON，见 core/database.py）
        self._db.delete(kb)
        self._db.commit()

    def set_doc_count(self, kb_id: int, count: int) -> None:
        self._db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).update({"doc_count": count})
        self._db.commit()

    # ── 成员授权 ───────────────────────────────────────────────────────

    def owned_kb_ids(self, user_id: int, kb_ids: list[int] | None = None) -> set[int]:
        query = self._db.query(KnowledgeBase.id).filter(KnowledgeBase.owner_id == user_id)
        if kb_ids is not None:
            query = query.filter(KnowledgeBase.id.in_(kb_ids))
        return {kb_id for (kb_id,) in query}

    def member_permissions(
        self, user_id: int, kb_ids: list[int] | None = None
    ) -> dict[int, KBPermission]:
        query = self._db.query(KBMember).filter(KBMember.user_id == user_id)
        if kb_ids is not None:
            query = query.filter(KBMember.kb_id.in_(kb_ids))
        return {member.kb_id: member.permission for member in query}

    def get_member(self, kb_id: int, user_id: int) -> KBMember | None:
        return self._db.get(KBMember, (kb_id, user_id))

    def list_members(self, kb_id: int) -> list[tuple[KBMember, User]]:
        rows = (
            self._db.query(KBMember, User)
            .join(User, KBMember.user_id == User.id)
            .filter(KBMember.kb_id == kb_id)
            .order_by(KBMember.created_at)
            .all()
        )
        return [(member, member_user) for member, member_user in rows]

    def upsert_member(self, kb_id: int, user_id: int, permission: KBPermission) -> KBMember:
        """重复授权即改权限（幂等），因此不需要单独的「改权限」接口。"""
        member = self._db.get(KBMember, (kb_id, user_id))
        if member is None:
            member = KBMember(kb_id=kb_id, user_id=user_id, permission=permission)
            self._db.add(member)
        else:
            member.permission = permission
        self._db.commit()
        return member

    def remove_member(self, kb_id: int, user_id: int) -> bool:
        member = self._db.get(KBMember, (kb_id, user_id))
        if member is None:
            return False
        self._db.delete(member)
        self._db.commit()
        return True


class SqlDocumentRepository:
    """文档聚合的 SQLAlchemy 实现。"""

    def __init__(self, db: Session) -> None:
        self._db = db

    def get(self, doc_id: int) -> Document | None:
        return self._db.get(Document, doc_id)

    def list_by_kb(self, kb_id: int) -> list[Document]:
        return (
            self._db.query(Document)
            .filter(Document.kb_id == kb_id)
            .order_by(Document.id.asc())
            .all()
        )

    def list_page(
        self,
        *,
        kb_id: int,
        status: str | None,
        keyword: str | None,
        page: int,
        page_size: int,
    ) -> tuple[list[Document], int]:
        query = self._db.query(Document).filter(Document.kb_id == kb_id)
        if status:
            query = query.filter(Document.status == status)
        if keyword:
            query = query.filter(Document.filename.like(f"%{keyword}%"))
        total = query.count()
        items = (
            query.order_by(Document.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
            .all()
        )
        return items, total

    def add_many(self, items: list[NewDocument]) -> list[int]:
        """一次提交一批新文档：上传 / 路径导入是「要么全受理、要么全不受理」。"""
        rows = [
            Document(
                kb_id=item.kb_id,
                filename=item.filename,
                file_path=item.file_path,
                file_type=item.file_type,
                file_size=item.file_size,
                source_type=item.source_type,
                status=DocStatus.PENDING,
            )
            for item in items
        ]
        self._db.add_all(rows)
        self._db.flush()
        doc_ids = [row.id for row in rows]
        self._db.commit()
        return doc_ids

    def update_status(self, doc: Document, status: DocStatus, error_msg: str | None = None) -> None:
        doc.status = status
        doc.error_msg = error_msg
        self._db.commit()

    def mark_completed(
        self,
        doc: Document,
        *,
        chunk_count: int,
        char_count: int,
        meta: dict[str, Any],
    ) -> None:
        doc.status = DocStatus.COMPLETED
        doc.chunk_count = chunk_count
        doc.char_count = char_count
        doc.meta_info = meta
        self._db.commit()

    def reset_for_reprocess(self, doc: Document) -> None:
        doc.status = DocStatus.PENDING
        doc.chunk_count = 0
        doc.error_msg = None
        self._db.commit()

    def reset_kb_for_reprocess(self, kb_id: int) -> list[int]:
        """整库打回待处理（全量重建用），一次提交并返回 doc_id 列表（按 id 升序）。

        与逐个 `reset_for_reprocess` 的区别只在提交次数：重建一个几千文档的库时，
        逐个提交既慢又会让中途失败的库停在「一半 PENDING 一半 COMPLETED」的半截状态。
        """
        doc_ids = [
            doc_id
            for (doc_id,) in self._db.query(Document.id)
            .filter(Document.kb_id == kb_id)
            .order_by(Document.id.asc())
        ]
        self._db.query(Document).filter(Document.kb_id == kb_id).update(
            {"status": DocStatus.PENDING, "chunk_count": 0, "error_msg": None}
        )
        self._db.commit()
        return doc_ids

    def count_completed(self, kb_id: int) -> int:
        return (
            self._db.query(Document)
            .filter(Document.kb_id == kb_id, Document.status == DocStatus.COMPLETED)
            .count()
        )

    def sync_kb_doc_count(self, kb_id: int) -> int:
        """把知识库的 `doc_count` 对齐到实际完成数，返回新值。

        与文档状态迁移放同一次提交：否则会出现「文档已完成、库计数还是旧值」的中间态，
        列表页展示的文档数与详情页对不上。
        """
        count = self.count_completed(kb_id)
        self._db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).update({"doc_count": count})
        self._db.commit()
        return count

    def delete(self, doc: Document) -> None:
        self._db.delete(doc)
        self._db.commit()


class SqlConversationRepository:
    """会话聚合的 SQLAlchemy 实现。"""

    def __init__(self, db: Session) -> None:
        self._db = db

    def get(self, conv_id: int) -> Conversation | None:
        return self._db.get(Conversation, conv_id)

    def create(self, *, kb_id: int, title: str) -> Conversation:
        conv = Conversation(kb_id=kb_id, title=title)
        self._db.add(conv)
        self._db.commit()
        self._db.refresh(conv)
        return conv

    def list_page(self, *, kb_id: int, page: int, page_size: int) -> tuple[list[Conversation], int]:
        query = self._db.query(Conversation).filter(Conversation.kb_id == kb_id)
        total = query.count()
        items = (
            query.order_by(Conversation.updated_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
            .all()
        )
        return items, total

    def messages(self, conv_id: int) -> list[Message]:
        return (
            self._db.query(Message)
            .filter(Message.conv_id == conv_id)
            .order_by(Message.created_at.asc(), Message.id.asc())
            .all()
        )

    def history(self, conv_id: int, limit: int = MESSAGE_HISTORY_LIMIT_DEFAULT) -> list[Message]:
        """取**最近** limit 条消息，按时间正序返回。

        先按 (created_at, id) 倒序取 N 条再反转：直接 `order_by(created_at.asc()).limit(N)`
        拿到的是最早 N 条，长对话里等于把近期上下文全丢了。
        """
        rows = (
            self._db.query(Message)
            .filter(Message.conv_id == conv_id)
            .order_by(Message.created_at.desc(), Message.id.desc())
            .limit(limit)
            .all()
        )
        return list(reversed(rows))

    def append_message(
        self, conv_id: int, role: str, content: str, sources: list[dict[str, Any]] | None = None
    ) -> Message:
        message = Message(conv_id=conv_id, role=role, content=content, sources=sources)
        self._db.add(message)
        self._db.commit()
        self._db.refresh(message)
        return message

    def touch(self, conv_id: int) -> None:
        """刷新会话活跃时间，保证会话列表按最近使用排序。"""
        self._db.query(Conversation).filter(Conversation.id == conv_id).update(
            {"updated_at": datetime.now(UTC)}
        )
        self._db.commit()

    def delete(self, conv: Conversation) -> None:
        self._db.delete(conv)
        self._db.commit()
