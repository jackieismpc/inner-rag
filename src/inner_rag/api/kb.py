"""知识库管理 API：CRUD + 成员授权（知识库级 ACL）。"""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger
from sqlalchemy.orm import Session

from inner_rag.api.deps import ensure_kb_access, get_current_user
from inner_rag.core.access import AccessLevel, accessible_kb_ids, level_map
from inner_rag.core.config import settings
from inner_rag.core.database import get_db
from inner_rag.models import KBMember, KBPermission, KnowledgeBase, User
from inner_rag.schemas import (
    KBCreate,
    KBMemberAdd,
    KBMemberOut,
    KBMemberPermission,
    KBOut,
    KBUpdate,
    PageData,
    ResponseModel,
    UserOut,
)
from inner_rag.services.cache import query_cache
from inner_rag.services.vector_store import vector_service

router = APIRouter(prefix="/api/kb", tags=["知识库"])


def _kb_out(kb: KnowledgeBase, level: AccessLevel) -> KBOut:
    """知识库出参的唯一出口：顺手填上请求者对它的权限。

    ``my_permission`` 不在 ORM 模型里（它是「谁在看」的属性，不是库的属性），
    因此必须显式带上；统一走这个函数可以保证没有哪个接口漏填。
    """
    out = KBOut.model_validate(kb)
    out.my_permission = level.name.lower()
    return out


@router.get("", response_model=ResponseModel)
def list_kbs(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    keyword: str | None = None,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """列出「我拥有或被授权」的知识库：过滤发生在 SQL 里，不是查完再筛。"""
    query = db.query(KnowledgeBase).filter(KnowledgeBase.id.in_(accessible_kb_ids(user)))
    if keyword:
        query = query.filter(KnowledgeBase.name.like(f"%{keyword}%"))
    total = query.count()
    items = (
        query.order_by(KnowledgeBase.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    levels = level_map(db, user, [item.id for item in items])
    return ResponseModel(
        data=PageData(
            total=total,
            items=[_kb_out(item, levels[item.id]) for item in items],
            page=page,
            page_size=page_size,
        )
    )


@router.post("", response_model=ResponseModel)
def create_kb(
    body: KBCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    payload = body.model_dump()
    # 建库时锁定 embedding 标识，之后换模型会被校验拦下，避免向量空间不一致
    payload["embedding_model"] = payload.get("embedding_model") or settings.embedding_key
    kb = KnowledgeBase(**payload, owner_id=user.id)
    db.add(kb)
    db.commit()
    db.refresh(kb)
    logger.info(
        f"[KB] 创建知识库 id={kb.id} name={kb.name!r} owner={user.username} "
        f"embedding={kb.embedding_model}"
    )
    return ResponseModel(data=_kb_out(kb, AccessLevel.OWNER))


@router.get("/{kb_id}", response_model=ResponseModel)
def get_kb(
    kb_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    kb, level = ensure_kb_access(db, kb_id, user, AccessLevel.READ)
    data = _kb_out(kb, level).model_dump()
    data.update(vector_service.get_kb_stats(kb_id))
    return ResponseModel(data=data)


@router.put("/{kb_id}", response_model=ResponseModel)
def update_kb(
    kb_id: int,
    body: KBUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """改设置属于拥有者权限：成员能读能写文档，但不能改库的属性。"""
    kb, _ = ensure_kb_access(db, kb_id, user, AccessLevel.OWNER)
    for key, value in body.model_dump(exclude_none=True).items():
        setattr(kb, key, value)
    db.commit()
    db.refresh(kb)
    return ResponseModel(data=_kb_out(kb, AccessLevel.OWNER))


@router.delete("/{kb_id}", response_model=ResponseModel)
def delete_kb(
    kb_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    kb, _ = ensure_kb_access(db, kb_id, user, AccessLevel.OWNER)

    # 先删向量与上传文件，再删数据库记录，避免留下孤儿数据（成员授权由外键级联删除）
    vector_service.delete_kb(kb_id)
    shutil.rmtree(Path(settings.UPLOAD_DIR) / f"kb_{kb_id}", ignore_errors=True)
    db.delete(kb)
    db.commit()
    query_cache.invalidate_kb_sync(kb_id)
    logger.info(f"[KB] 删除知识库 id={kb_id} owner={user.username}")
    return ResponseModel(message="删除成功")


# ── 成员授权（仅拥有者）───────────────────────────────────────────────


@router.get("/{kb_id}/members", response_model=ResponseModel)
def list_members(
    kb_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    kb, _ = ensure_kb_access(db, kb_id, user, AccessLevel.OWNER)
    rows = (
        db.query(KBMember, User)
        .join(User, KBMember.user_id == User.id)
        .filter(KBMember.kb_id == kb_id)
        .order_by(KBMember.created_at)
        .all()
    )
    return ResponseModel(
        data={
            "owner": UserOut.model_validate(db.get(User, kb.owner_id)),
            "items": [
                KBMemberOut(
                    user_id=member_user.id,
                    username=member_user.username,
                    display_name=member_user.display_name,
                    permission=KBMemberPermission(member.permission.value),
                )
                for member, member_user in rows
            ],
        }
    )


@router.post("/{kb_id}/members", response_model=ResponseModel)
def add_member(
    kb_id: int,
    body: KBMemberAdd,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """按用户名授权；重复调用即改权限（幂等），不需要单独的「改权限」接口。"""
    kb, _ = ensure_kb_access(db, kb_id, user, AccessLevel.OWNER)

    target = db.query(User).filter(User.username == body.username).one_or_none()
    if target is None:
        raise HTTPException(status_code=404, detail=f"用户不存在: {body.username}")
    if target.id == kb.owner_id:
        raise HTTPException(status_code=400, detail="拥有者不需要加入成员列表")

    member = db.get(KBMember, (kb_id, target.id))
    if member is None:
        db.add(
            KBMember(kb_id=kb_id, user_id=target.id, permission=KBPermission(body.permission.value))
        )
    else:
        member.permission = KBPermission(body.permission.value)
    db.commit()
    logger.info(
        f"[KB] 授权 kb={kb_id} user={target.username} perm={body.permission.value} "
        f"by={user.username}"
    )
    return ResponseModel(
        message="授权成功",
        data=KBMemberOut(
            user_id=target.id,
            username=target.username,
            display_name=target.display_name,
            permission=body.permission,
        ),
    )


@router.delete("/{kb_id}/members/{user_id}", response_model=ResponseModel)
def remove_member(
    kb_id: int,
    user_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    ensure_kb_access(db, kb_id, user, AccessLevel.OWNER)
    member = db.get(KBMember, (kb_id, user_id))
    if member is None:
        raise HTTPException(status_code=404, detail="该用户不在授权列表中")
    db.delete(member)
    db.commit()
    logger.info(f"[KB] 移除授权 kb={kb_id} user_id={user_id} by={user.username}")
    return ResponseModel(message="已移除授权")
