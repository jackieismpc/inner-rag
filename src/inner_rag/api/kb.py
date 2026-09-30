"""知识库管理 API：CRUD + 成员授权（知识库级 ACL）。

本层只做参数校验、判权与序列化：查询与写入都在 ``repositories/``（见 `docs/architecture.md`
§3.3），因此换关系库不需要改这个文件。
"""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger

from inner_rag.api.deps import ensure_kb_access, get_current_user, get_repositories
from inner_rag.core.access import AccessLevel, accessible_kb_ids, level_map
from inner_rag.core.config import settings
from inner_rag.models import KBPermission, KnowledgeBase, User
from inner_rag.repositories import Repositories
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
    repos: Repositories = Depends(get_repositories),
    user: User = Depends(get_current_user),
):
    """列出「我拥有或被授权」的知识库：过滤发生在 SQL 里，不是查完再筛。"""
    kb_ids = accessible_kb_ids(repos.kbs, user)
    items, total = repos.kbs.list_page(
        kb_ids=kb_ids, keyword=keyword, page=page, page_size=page_size
    )
    levels = level_map(repos.kbs, user, [item.id for item in items])
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
    repos: Repositories = Depends(get_repositories),
    user: User = Depends(get_current_user),
):
    # 建库时锁定 embedding 标识，之后换模型会被校验拦下，避免向量空间不一致
    kb = repos.kbs.create(
        name=body.name,
        description=body.description,
        icon=body.icon,
        embedding_model=body.embedding_model or settings.embedding_key,
        owner_id=user.id,
    )
    logger.info(
        f"[KB] 创建知识库 id={kb.id} name={kb.name!r} owner={user.username} "
        f"embedding={kb.embedding_model}"
    )
    return ResponseModel(data=_kb_out(kb, AccessLevel.OWNER))


@router.get("/{kb_id}", response_model=ResponseModel)
def get_kb(
    kb_id: int,
    repos: Repositories = Depends(get_repositories),
    user: User = Depends(get_current_user),
):
    kb, level = ensure_kb_access(repos.kbs, kb_id, user, AccessLevel.READ)
    data = _kb_out(kb, level).model_dump()
    data.update({"vector_count": vector_service.count(kb_id)})
    return ResponseModel(data=data)


@router.put("/{kb_id}", response_model=ResponseModel)
def update_kb(
    kb_id: int,
    body: KBUpdate,
    repos: Repositories = Depends(get_repositories),
    user: User = Depends(get_current_user),
):
    """改设置属于拥有者权限：成员能读能写文档，但不能改库的属性。"""
    kb, _ = ensure_kb_access(repos.kbs, kb_id, user, AccessLevel.OWNER)
    kb = repos.kbs.update(kb, body.model_dump(exclude_none=True))
    return ResponseModel(data=_kb_out(kb, AccessLevel.OWNER))


@router.delete("/{kb_id}", response_model=ResponseModel)
async def delete_kb(
    kb_id: int,
    repos: Repositories = Depends(get_repositories),
    user: User = Depends(get_current_user),
):
    kb, _ = ensure_kb_access(repos.kbs, kb_id, user, AccessLevel.OWNER)

    # 先删向量与上传文件，再删数据库记录，避免留下孤儿数据（成员授权由外键级联删除）
    await vector_service.delete_kb(kb_id)
    shutil.rmtree(Path(settings.UPLOAD_DIR) / f"kb_{kb_id}", ignore_errors=True)
    repos.kbs.delete(kb)
    query_cache.invalidate_kb_sync(kb_id)
    logger.info(f"[KB] 删除知识库 id={kb_id} owner={user.username}")
    return ResponseModel(message="删除成功")


# ── 成员授权（仅拥有者）───────────────────────────────────────────────


@router.get("/{kb_id}/members", response_model=ResponseModel)
def list_members(
    kb_id: int,
    repos: Repositories = Depends(get_repositories),
    user: User = Depends(get_current_user),
):
    kb, _ = ensure_kb_access(repos.kbs, kb_id, user, AccessLevel.OWNER)
    owner = repos.users.get(kb.owner_id)
    if owner is None:
        # 归属用户不存在意味着数据库被外部改坏了（FK 是 RESTRICT），显性报错而不是回一个空 owner
        raise HTTPException(status_code=500, detail="知识库归属用户不存在，请联系管理员修复数据")
    return ResponseModel(
        data={
            "owner": UserOut.model_validate(owner),
            "items": [
                KBMemberOut(
                    user_id=member_user.id,
                    username=member_user.username,
                    display_name=member_user.display_name,
                    permission=KBMemberPermission(member.permission.value),
                )
                for member, member_user in repos.kbs.list_members(kb_id)
            ],
        }
    )


@router.post("/{kb_id}/members", response_model=ResponseModel)
def add_member(
    kb_id: int,
    body: KBMemberAdd,
    repos: Repositories = Depends(get_repositories),
    user: User = Depends(get_current_user),
):
    """按用户名授权；重复调用即改权限（幂等），不需要单独的「改权限」接口。"""
    kb, _ = ensure_kb_access(repos.kbs, kb_id, user, AccessLevel.OWNER)

    target = repos.users.find_by_username(body.username)
    if target is None:
        raise HTTPException(status_code=404, detail=f"用户不存在: {body.username}")
    if target.id == kb.owner_id:
        raise HTTPException(status_code=400, detail="拥有者不需要加入成员列表")

    repos.kbs.upsert_member(kb_id, target.id, KBPermission(body.permission.value))
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
    repos: Repositories = Depends(get_repositories),
    user: User = Depends(get_current_user),
):
    ensure_kb_access(repos.kbs, kb_id, user, AccessLevel.OWNER)
    if not repos.kbs.remove_member(kb_id, user_id):
        raise HTTPException(status_code=404, detail="该用户不在授权列表中")
    logger.info(f"[KB] 移除授权 kb={kb_id} user_id={user_id} by={user.username}")
    return ResponseModel(message="已移除授权")
