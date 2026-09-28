"""知识库管理 API。"""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger
from sqlalchemy.orm import Session

from inner_rag.core.config import settings
from inner_rag.core.database import get_db
from inner_rag.models import KnowledgeBase
from inner_rag.schemas import KBCreate, KBOut, KBUpdate, PageData, ResponseModel
from inner_rag.services.cache import query_cache
from inner_rag.services.vector_store import vector_service

router = APIRouter(prefix="/api/kb", tags=["知识库"])


@router.get("", response_model=ResponseModel)
def list_kbs(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    keyword: str | None = None,
    db: Session = Depends(get_db),
):
    query = db.query(KnowledgeBase)
    if keyword:
        query = query.filter(KnowledgeBase.name.like(f"%{keyword}%"))
    total = query.count()
    items = (
        query.order_by(KnowledgeBase.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return ResponseModel(
        data=PageData(
            total=total,
            items=[KBOut.model_validate(item) for item in items],
            page=page,
            page_size=page_size,
        )
    )


@router.post("", response_model=ResponseModel)
def create_kb(body: KBCreate, db: Session = Depends(get_db)):
    payload = body.model_dump()
    # 建库时锁定 embedding 标识，之后换模型会被校验拦下，避免向量空间不一致
    payload["embedding_model"] = payload.get("embedding_model") or settings.embedding_key
    kb = KnowledgeBase(**payload)
    db.add(kb)
    db.commit()
    db.refresh(kb)
    logger.info(f"[KB] 创建知识库 id={kb.id} name={kb.name!r} embedding={kb.embedding_model}")
    return ResponseModel(data=KBOut.model_validate(kb))


@router.get("/{kb_id}", response_model=ResponseModel)
def get_kb(kb_id: int, db: Session = Depends(get_db)):
    kb = db.get(KnowledgeBase, kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")
    data = KBOut.model_validate(kb).model_dump()
    data.update(vector_service.get_kb_stats(kb_id))
    return ResponseModel(data=data)


@router.put("/{kb_id}", response_model=ResponseModel)
def update_kb(kb_id: int, body: KBUpdate, db: Session = Depends(get_db)):
    kb = db.get(KnowledgeBase, kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")
    for key, value in body.model_dump(exclude_none=True).items():
        setattr(kb, key, value)
    db.commit()
    db.refresh(kb)
    return ResponseModel(data=KBOut.model_validate(kb))


@router.delete("/{kb_id}", response_model=ResponseModel)
def delete_kb(kb_id: int, db: Session = Depends(get_db)):
    kb = db.get(KnowledgeBase, kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")

    # 先删向量与上传文件，再删数据库记录，避免留下孤儿数据
    vector_service.delete_kb(kb_id)
    shutil.rmtree(Path(settings.UPLOAD_DIR) / f"kb_{kb_id}", ignore_errors=True)
    db.delete(kb)
    db.commit()
    query_cache.invalidate_kb_sync(kb_id)
    logger.info(f"[KB] 删除知识库 id={kb_id}")
    return ResponseModel(message="删除成功")
