"""仓储契约测试（Phase 7.4）：关系库访问收敛到 `repositories/` 之后的行为。

为什么单独一个文件：`repositories/` 是唯一直接持有 SQLAlchemy 会话的地方，
它的语义（事务边界、排序、级联、作用域）无法从 HTTP 层用例反推——`api/` 用例只看状态码，
看不出「history 取的是最近 N 条」这类细节。这里直接对着仓储断言。

读回策略：写操作用一个会话、验证用**新开的会话**读。仓储的批量更新
（`Query.update()`）会同步会话里的对象，若沿用同一会话断言，就等于在验证
「SQLAlchemy 的身份映射」，而不是「数据真的落库了」。
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime

import pytest
from sqlalchemy import update

from conftest import create_user
from inner_rag.core.database import SessionLocal
from inner_rag.models import Conversation, DocStatus, Document, KBPermission, User
from inner_rag.repositories import NewDocument, Repositories, build_repositories


@contextmanager
def session() -> Iterator[Repositories]:
    """独立会话的仓储：用来验证写入真的落库（不受上一次查询的缓存影响）。"""
    with SessionLocal() as db:
        yield build_repositories(db)


@pytest.fixture
def owner() -> User:
    return create_user()


@pytest.fixture
def kb_id(owner: User) -> Iterator[int]:
    """一个用完即删的知识库（删库会级联带走文档 / 会话 / 成员）。"""
    with session() as repos:
        kb = repos.kbs.create(
            name="pytest-repo-kb",
            description="仓储用例",
            icon=None,
            embedding_model="mock",
            owner_id=owner.id,
        )
    yield kb.id
    with session() as repos:
        fresh = repos.kbs.get(kb.id)
        if fresh is not None:
            repos.kbs.delete(fresh)


def _new_doc(kb_id: int, filename: str) -> NewDocument:
    return NewDocument(
        kb_id=kb_id,
        filename=filename,
        file_path=f"/tmp/{filename}",
        file_type="txt",
        file_size=10,
        source_type="upload",
    )


def _add_docs(repos: Repositories, kb_id: int, *names: str) -> list[int]:
    return repos.docs.add_many([_new_doc(kb_id, name) for name in names])


def _read_doc(doc_id: int) -> Document:
    """新会话读一个文档，顺便断言它存在（省掉一堆 `assert x is not None`）。"""
    with session() as repos:
        doc = repos.docs.get(doc_id)
    assert doc is not None
    return doc


def _doc_count(kb_id: int) -> int:
    with session() as repos:
        kb = repos.kbs.get(kb_id)
    assert kb is not None
    return kb.doc_count


class TestUserRepository:
    def test_get_and_find_by_username(self, owner: User) -> None:
        with session() as repos:
            assert repos.users.get(owner.id) is not None
            found = repos.users.find_by_username(owner.username)
        assert found is not None
        assert found.id == owner.id

    def test_missing_lookups_return_none(self) -> None:
        with session() as repos:
            assert repos.users.get(10**9) is None
            assert repos.users.find_by_username("no-such-user") is None


class TestKnowledgeBaseRepository:
    def test_create_then_get_roundtrip(self, kb_id: int) -> None:
        with session() as repos:
            kb = repos.kbs.get(kb_id)
        assert kb is not None
        assert kb.name == "pytest-repo-kb"
        assert kb.description == "仓储用例"
        assert kb.doc_count == 0

    def test_list_page_filters_in_sql_not_in_python(self, kb_id: int) -> None:
        with session() as repos:
            items, total = repos.kbs.list_page(kb_ids=[kb_id], keyword=None, page=1, page_size=10)
            assert [kb.id for kb in items] == [kb_id]
            assert total == 1

            # 不可见的 id 一律不返回（分页在 SQL 里过滤，不是查完再筛）
            assert repos.kbs.list_page(kb_ids=[], keyword=None, page=1, page_size=10) == ([], 0)

            # 关键字参与过滤：证明它没有被忽略
            assert repos.kbs.list_page(
                kb_ids=[kb_id], keyword="绝不匹配的关键字", page=1, page_size=10
            ) == ([], 0)

    def test_update_only_touches_given_fields(self, kb_id: int) -> None:
        with session() as repos:
            kb = repos.kbs.get(kb_id)
            assert kb is not None
            repos.kbs.update(kb, {"name": "renamed"})
        with session() as repos:
            fresh = repos.kbs.get(kb_id)
        assert fresh is not None
        assert fresh.name == "renamed"
        assert fresh.description == "仓储用例"

    def test_set_doc_count_is_independent_of_documents(self, owner: User) -> None:
        """`set_doc_count` 只写字段、不重算：它服务于「已知计数」的直写场景。"""
        with session() as repos:
            kb = repos.kbs.create(
                name="pytest-repo-count",
                description=None,
                icon=None,
                embedding_model="mock",
                owner_id=owner.id,
            )
            repos.kbs.set_doc_count(kb.id, 42)
        try:
            assert _doc_count(kb.id) == 42
        finally:
            with session() as repos:
                fresh = repos.kbs.get(kb.id)
                if fresh is not None:
                    repos.kbs.delete(fresh)

    def test_owner_is_not_a_member_row(self, kb_id: int, owner: User) -> None:
        """拥有者权限来自 owner_id，不该在成员表里留一行（否则改成员会覆盖 owner）。"""
        with session() as repos:
            assert repos.kbs.get_member(kb_id, owner.id) is None
            assert kb_id in repos.kbs.owned_kb_ids(owner.id)
            assert repos.kbs.member_permissions(owner.id) == {}

    def test_upsert_member_is_idempotent(self, kb_id: int) -> None:
        member = create_user()
        with session() as repos:
            first = repos.kbs.upsert_member(kb_id, member.id, KBPermission.READ)
            second = repos.kbs.upsert_member(kb_id, member.id, KBPermission.WRITE)
            # 复合主键 (kb_id, user_id)：重复授权改的是同一行，不是插入第二条
            assert (second.kb_id, second.user_id) == (first.kb_id, first.user_id)
            assert second.permission is KBPermission.WRITE

        with session() as repos:
            stored = repos.kbs.get_member(kb_id, member.id)
            assert stored is not None
            assert stored.permission is KBPermission.WRITE
            assert repos.kbs.member_permissions(member.id) == {kb_id: KBPermission.WRITE}
            assert len(repos.kbs.list_members(kb_id)) == 1

    def test_list_members_joins_user(self, kb_id: int) -> None:
        member = create_user(display_name="被授权人")
        with session() as repos:
            repos.kbs.upsert_member(kb_id, member.id, KBPermission.READ)
            rows = repos.kbs.list_members(kb_id)
        assert [(m.user_id, u.username) for m, u in rows] == [(member.id, member.username)]
        assert rows[0][1].display_name == "被授权人"

    def test_remove_member_reports_whether_row_existed(self, kb_id: int) -> None:
        member = create_user()
        with session() as repos:
            assert repos.kbs.remove_member(kb_id, member.id) is False
            repos.kbs.upsert_member(kb_id, member.id, KBPermission.READ)
            assert repos.kbs.remove_member(kb_id, member.id) is True
            assert repos.kbs.get_member(kb_id, member.id) is None

    def test_lookups_are_scoped(self, owner: User) -> None:
        """`kb_ids` 参数为空列表时必须是空集，不能退化成「全部」。"""
        with session() as repos:
            assert repos.kbs.owned_kb_ids(owner.id, kb_ids=[]) == set()
            assert repos.kbs.owned_kb_ids(10**9) == set()
            assert repos.kbs.member_permissions(10**9) == {}


class TestDocumentRepository:
    def test_add_many_returns_ids_in_input_order(self, kb_id: int) -> None:
        with session() as repos:
            doc_ids = _add_docs(repos, kb_id, "f0.txt", "f1.txt", "f2.txt")
        assert len(doc_ids) == 3
        assert all(doc.status is DocStatus.PENDING for doc in map(_read_doc, doc_ids))
        with session() as repos:
            assert [doc.filename for doc in repos.docs.list_by_kb(kb_id)] == [
                "f0.txt",
                "f1.txt",
                "f2.txt",
            ]

    def test_add_many_empty_is_noop(self, kb_id: int) -> None:
        with session() as repos:
            assert repos.docs.add_many([]) == []

    def test_list_by_kb_is_scoped(self, kb_id: int, owner: User) -> None:
        with session() as repos:
            other = repos.kbs.create(
                name="pytest-repo-scope",
                description=None,
                icon=None,
                embedding_model="mock",
                owner_id=owner.id,
            )
            _add_docs(repos, kb_id, "mine.txt")
            _add_docs(repos, other.id, "theirs.txt")
        try:
            with session() as repos:
                assert [doc.filename for doc in repos.docs.list_by_kb(kb_id)] == ["mine.txt"]
        finally:
            with session() as repos:
                fresh = repos.kbs.get(other.id)
                if fresh is not None:
                    repos.kbs.delete(fresh)

    def test_list_page_status_and_keyword_filters(self, kb_id: int) -> None:
        with session() as repos:
            first = _add_docs(repos, kb_id, "alpha.txt", "beta.txt")[0]
            doc = repos.docs.get(first)
            assert doc is not None
            repos.docs.update_status(doc, DocStatus.FAILED, error_msg="boom")

        with session() as repos:
            items, total = repos.docs.list_page(
                kb_id=kb_id, status="failed", keyword=None, page=1, page_size=10
            )
            assert [doc.id for doc in items] == [first]
            assert total == 1

            items, total = repos.docs.list_page(
                kb_id=kb_id, status=None, keyword="beta", page=1, page_size=10
            )
            assert [doc.filename for doc in items] == ["beta.txt"]
            assert total == 1

            # 分页：总数是过滤后的总数，与 page_size 无关
            items, total = repos.docs.list_page(
                kb_id=kb_id, status=None, keyword=None, page=2, page_size=1
            )
            assert len(items) == 1
            assert total == 2

    def test_update_status_persists_error_message(self, kb_id: int) -> None:
        with session() as repos:
            doc_id = _add_docs(repos, kb_id, "a.txt")[0]
            doc = repos.docs.get(doc_id)
            assert doc is not None
            repos.docs.update_status(doc, DocStatus.FAILED, error_msg="boom")

        doc = _read_doc(doc_id)
        assert doc.status is DocStatus.FAILED
        assert doc.error_msg == "boom"

        # error_msg 默认参数为 None：重试成功时要把上一次的错误清掉
        with session() as repos:
            doc = repos.docs.get(doc_id)
            assert doc is not None
            repos.docs.update_status(doc, DocStatus.COMPLETED)
        assert _read_doc(doc_id).error_msg is None

    def test_mark_completed_then_sync_doc_count(self, kb_id: int) -> None:
        with session() as repos:
            doc_id = _add_docs(repos, kb_id, "a.txt")[0]
            doc = repos.docs.get(doc_id)
            assert doc is not None
            repos.docs.mark_completed(doc, chunk_count=7, char_count=70, meta={"k": "v"})
            assert repos.docs.count_completed(kb_id) == 1
            # mark_completed 不改库计数（那是另一次提交），sync 才对齐
            assert repos.docs.sync_kb_doc_count(kb_id) == 1

        assert _doc_count(kb_id) == 1
        doc = _read_doc(doc_id)
        assert doc.status is DocStatus.COMPLETED
        assert doc.chunk_count == 7
        assert doc.char_count == 70
        assert doc.meta_info == {"k": "v"}

    def test_count_completed_ignores_unfinished(self, kb_id: int) -> None:
        with session() as repos:
            done, _pending = _add_docs(repos, kb_id, "done.txt", "pending.txt")
            doc = repos.docs.get(done)
            assert doc is not None
            repos.docs.mark_completed(doc, chunk_count=1, char_count=1, meta={})
            assert repos.docs.count_completed(kb_id) == 1
            assert repos.docs.sync_kb_doc_count(kb_id) == 1
        assert _doc_count(kb_id) == 1

    def test_reset_for_reprocess_clears_error_and_sets_pending(self, kb_id: int) -> None:
        with session() as repos:
            doc_id = _add_docs(repos, kb_id, "a.txt")[0]
            doc = repos.docs.get(doc_id)
            assert doc is not None
            repos.docs.mark_completed(doc, chunk_count=5, char_count=50, meta={"k": "v"})
            repos.docs.update_status(doc, DocStatus.FAILED, error_msg="old error")
            doc = repos.docs.get(doc_id)
            assert doc is not None
            repos.docs.reset_for_reprocess(doc)

        doc = _read_doc(doc_id)
        assert doc.status is DocStatus.PENDING
        assert doc.chunk_count == 0
        assert doc.error_msg is None

    def test_reset_kb_for_reprocess_only_touches_that_kb(self, kb_id: int, owner: User) -> None:
        """整库重置必须限定 kb_id：跨库误伤会让另一个库的索引凭空失效。"""
        with session() as repos:
            other = repos.kbs.create(
                name="pytest-repo-kb-other",
                description=None,
                icon=None,
                embedding_model="mock",
                owner_id=owner.id,
            )
            target_ids = _add_docs(repos, kb_id, "t0.txt", "t1.txt")
            other_ids = _add_docs(repos, other.id, "o.txt")
            # 两边都推到终态，这样「只有目标库被重置」才是可观测的
            for doc_id in target_ids + other_ids:
                doc = repos.docs.get(doc_id)
                assert doc is not None
                repos.docs.mark_completed(doc, chunk_count=1, char_count=1, meta={})
            assert repos.docs.reset_kb_for_reprocess(kb_id) == target_ids

        try:
            assert [d.status for d in map(_read_doc, target_ids)] == [DocStatus.PENDING] * 2
            assert [d.status for d in map(_read_doc, other_ids)] == [DocStatus.COMPLETED]
        finally:
            with session() as repos:
                fresh = repos.kbs.get(other.id)
                if fresh is not None:
                    repos.kbs.delete(fresh)

    def test_reset_kb_for_reprocess_returns_ids_in_order_and_empty_for_unknown(
        self, kb_id: int
    ) -> None:
        with session() as repos:
            doc_ids = _add_docs(repos, kb_id, "c.txt", "b.txt", "a.txt")
        with session() as repos:
            # 按 id 升序（= 插入顺序），与 list_by_kb 一致；空库返回空列表而不是报错
            assert repos.docs.reset_kb_for_reprocess(kb_id) == doc_ids
            assert repos.docs.reset_kb_for_reprocess(10**9) == []

    def test_delete_removes_row(self, kb_id: int) -> None:
        with session() as repos:
            doc_id = _add_docs(repos, kb_id, "a.txt")[0]
            doc = repos.docs.get(doc_id)
            assert doc is not None
            repos.docs.mark_completed(doc, chunk_count=1, char_count=1, meta={})
            repos.docs.sync_kb_doc_count(kb_id)

            doc = repos.docs.get(doc_id)
            assert doc is not None
            repos.docs.delete(doc)
            assert repos.docs.sync_kb_doc_count(kb_id) == 0

        with session() as repos:
            assert repos.docs.get(doc_id) is None
        assert _doc_count(kb_id) == 0


class TestConversationRepository:
    def test_create_and_list_page_scoped_by_kb(self, kb_id: int) -> None:
        with session() as repos:
            conv = repos.convs.create(kb_id=kb_id, title="第一个会话")
        with session() as repos:
            items, total = repos.convs.list_page(kb_id=kb_id, page=1, page_size=10)
            assert [item.id for item in items] == [conv.id]
            assert total == 1
            assert repos.convs.list_page(kb_id=conv.id + 10**6, page=1, page_size=10) == ([], 0)
            fetched = repos.convs.get(conv.id)
        assert fetched is not None
        assert fetched.title == "第一个会话"

    def test_history_returns_most_recent_n_in_chronological_order(self, kb_id: int) -> None:
        """回归用例：直接 `asc().limit(N)` 会取到最早 N 条，长对话里等于丢掉近期上下文。"""
        with session() as repos:
            conv = repos.convs.create(kb_id=kb_id, title="长对话")
            for i in range(5):
                repos.convs.append_message(conv.id, role="user", content=f"m{i}")

        with session() as repos:
            assert [m.content for m in repos.convs.history(conv.id, limit=2)] == ["m3", "m4"]
            assert [m.content for m in repos.convs.messages(conv.id)] == [f"m{i}" for i in range(5)]

    def test_append_message_persists_sources(self, kb_id: int) -> None:
        sources = [{"doc_id": 1, "page": 2, "score": 0.5}]
        with session() as repos:
            conv = repos.convs.create(kb_id=kb_id, title="带引用")
            message = repos.convs.append_message(
                conv.id, role="assistant", content="答案", sources=sources
            )
            assert message.role == "assistant"
        with session() as repos:
            assert repos.convs.messages(conv.id)[0].sources == sources

    def test_touch_refreshes_updated_at(self, kb_id: int) -> None:
        """touch 要真的写回时间，否则会话列表按「最近使用」排序会失效。"""
        with session() as repos:
            conv = repos.convs.create(kb_id=kb_id, title="会话")

        # 先把 updated_at 压到 2000 年：这样「有没有写」是确定的，不受时间精度影响
        with SessionLocal() as db:
            db.execute(
                update(Conversation)
                .where(Conversation.id == conv.id)
                .values(updated_at=datetime(2000, 1, 1))
            )
            db.commit()

        with session() as repos:
            repos.convs.touch(conv.id)

        with session() as repos:
            fresh = repos.convs.get(conv.id)
        assert fresh is not None
        assert fresh.updated_at is not None
        assert fresh.updated_at.year > 2000

    def test_delete_conversation_cascades_messages(self, kb_id: int) -> None:
        with session() as repos:
            conv = repos.convs.create(kb_id=kb_id, title="待删")
            repos.convs.append_message(conv.id, role="user", content="hi")
            repos.convs.delete(conv)
            assert repos.convs.messages(conv.id) == []
        with session() as repos:
            assert repos.convs.get(conv.id) is None


class TestCascadeOnKbDelete:
    def test_deleting_kb_takes_documents_and_conversations(self, owner: User) -> None:
        """删库必须连带清干净聚合：否则会留下指向不存在知识库的孤儿文档（向量则是残留）。"""
        with session() as repos:
            kb = repos.kbs.create(
                name="pytest-repo-cascade",
                description=None,
                icon=None,
                embedding_model="mock",
                owner_id=owner.id,
            )
            doc_id = _add_docs(repos, kb.id, "a.txt")[0]
            conv = repos.convs.create(kb_id=kb.id, title="会话")
            repos.kbs.upsert_member(kb.id, owner.id, KBPermission.READ)
            repos.kbs.delete(kb)

        with session() as repos:
            assert repos.kbs.get(kb.id) is None
            assert repos.docs.get(doc_id) is None
            assert repos.convs.get(conv.id) is None
            assert repos.kbs.get_member(kb.id, owner.id) is None
