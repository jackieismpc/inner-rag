"""知识库级 ACL 判定的唯一实现（纯策略，不抛 HTTP 状态码）。

访问级别：

* ``READ``  —— 看知识库详情、文档列表、会话与问答；
* ``WRITE`` —— READ + 上传 / 删除文档、重新解析（改变库内内容）；
* ``OWNER`` —— WRITE + 改知识库设置、删库、管理成员。

判定只有这一处实现：路由不得自己拼权限条件，否则迟早在某个接口上漏一次。
HTTP 语义（404 / 403 映射）在 ``api/deps.py``。
"""

from __future__ import annotations

import enum

from sqlalchemy import CompoundSelect, select, union
from sqlalchemy.orm import Session

from inner_rag.models import KBMember, KBPermission, KnowledgeBase, User


class AccessLevel(enum.IntEnum):
    READ = 1
    WRITE = 2
    OWNER = 3


_PERMISSION_LEVELS = {
    KBPermission.READ: AccessLevel.READ,
    KBPermission.WRITE: AccessLevel.WRITE,
}


def level_of(db: Session, kb: KnowledgeBase, user: User) -> AccessLevel | None:
    """当前用户对该知识库的访问级别；无权限返回 None。owner 优先于成员表。"""
    if kb.owner_id == user.id:
        return AccessLevel.OWNER
    member = db.get(KBMember, (kb.id, user.id))
    return None if member is None else _PERMISSION_LEVELS[member.permission]


def level_map(db: Session, user: User, kb_ids: list[int]) -> dict[int, AccessLevel]:
    """批量算权限（列表接口用）：一次查库 + 一次查成员，避免逐行 N+1。"""
    if not kb_ids:
        return {}

    levels = {
        kb_id
        for (kb_id,) in db.query(KnowledgeBase.id).filter(
            KnowledgeBase.owner_id == user.id, KnowledgeBase.id.in_(kb_ids)
        )
    }
    result = dict.fromkeys(levels, AccessLevel.OWNER)
    members = (
        db.query(KBMember).filter(KBMember.user_id == user.id, KBMember.kb_id.in_(kb_ids)).all()
    )
    for member in members:
        # setdefault：owner 与成员记录同时存在时以 OWNER 为准
        result.setdefault(member.kb_id, _PERMISSION_LEVELS[member.permission])
    return result


def accessible_kb_ids(user: User) -> CompoundSelect[int]:
    """「我可见的知识库 id」子查询：own ∪ member，可直接用于 ``in_()``。"""
    owned = select(KnowledgeBase.id).where(KnowledgeBase.owner_id == user.id)
    member = select(KBMember.kb_id).where(KBMember.user_id == user.id)
    return union(owned, member)
