"""知识库级 ACL 判定的唯一实现（纯策略，不抛 HTTP 状态码）。

访问级别：

* ``READ``  —— 看知识库详情、文档列表、会话与问答；
* ``WRITE`` —— READ + 上传 / 删除文档、重新解析（改变库内内容）；
* ``OWNER`` —— WRITE + 改知识库设置、删库、管理成员。

判定只有这一处实现：路由不得自己拼权限条件，否则迟早在某个接口上漏一次。
HTTP 语义（404 / 403 映射）在 ``api/deps.py``。

本模块只声明需要的数据读取能力（``ACLReader`` 协议），不 import 仓储实现：
``core/`` 是策略层，不能反过来依赖数据库适配器；结构化协议让仓储不需要继承任何东西，
而 mypy 仍会检查双方签名是否兼容。
"""

from __future__ import annotations

import enum
from typing import Protocol

from inner_rag.models import KBMember, KBPermission, KnowledgeBase, User


class AccessLevel(enum.IntEnum):
    READ = 1
    WRITE = 2
    OWNER = 3


_PERMISSION_LEVELS = {
    KBPermission.READ: AccessLevel.READ,
    KBPermission.WRITE: AccessLevel.WRITE,
}


class ACLReader(Protocol):
    """ACL 判定所需的最小读取能力（成员关系 + 归属关系）。"""

    def get_member(self, kb_id: int, user_id: int) -> KBMember | None: ...

    def owned_kb_ids(self, user_id: int, kb_ids: list[int] | None = None) -> set[int]: ...

    def member_permissions(
        self, user_id: int, kb_ids: list[int] | None = None
    ) -> dict[int, KBPermission]: ...


def level_of(kbs: ACLReader, kb: KnowledgeBase, user: User) -> AccessLevel | None:
    """当前用户对该知识库的访问级别；无权限返回 None。owner 优先于成员表。"""
    if kb.owner_id == user.id:
        return AccessLevel.OWNER
    member = kbs.get_member(kb.id, user.id)
    return None if member is None else _PERMISSION_LEVELS[member.permission]


def level_map(kbs: ACLReader, user: User, kb_ids: list[int]) -> dict[int, AccessLevel]:
    """批量算权限（列表接口用）：一次查归属 + 一次查成员，避免逐行 N+1。"""
    if not kb_ids:
        return {}

    result = dict.fromkeys(kbs.owned_kb_ids(user.id, kb_ids), AccessLevel.OWNER)
    for kb_id, permission in kbs.member_permissions(user.id, kb_ids).items():
        # setdefault：owner 与成员记录同时存在时以 OWNER 为准
        result.setdefault(kb_id, _PERMISSION_LEVELS[permission])
    return result


def accessible_kb_ids(kbs: ACLReader, user: User) -> list[int]:
    """「我可见的知识库 id」= own ∪ member。

    返回具体列表而不是 SQL 子查询：子查询会把 SQLAlchemy 类型泄漏到仓储边界之外，
    而知识库数量本来就不大（列表接口也只分页展示）。
    """
    return sorted(kbs.owned_kb_ids(user.id) | set(kbs.member_permissions(user.id)))
