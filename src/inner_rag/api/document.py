"""文档管理 API：上传、路径导入、查询、删除、重新处理。

本层只做参数校验、判权与序列化；查询与写入都在 ``repositories/``，长任务都在任务队列里。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from loguru import logger

from inner_rag.api.deps import (
    ensure_doc_access,
    ensure_kb_access,
    get_current_user,
    get_repositories,
)
from inner_rag.core.access import AccessLevel
from inner_rag.models import DocStatus, Document, User
from inner_rag.repositories import NewDocument, Repositories
from inner_rag.schemas import DocOut, LocalPathImport, PageData, ResponseModel
from inner_rag.services.document import doc_service
from inner_rag.services.parser import parser

router = APIRouter(prefix="/api/doc", tags=["文档"])


@router.get("", response_model=ResponseModel)
def list_docs(
    kb_id: int = Query(...),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: str | None = None,
    keyword: str | None = None,
    repos: Repositories = Depends(get_repositories),
    user: User = Depends(get_current_user),
):
    ensure_kb_access(repos.kbs, kb_id, user, AccessLevel.READ)
    items, total = repos.docs.list_page(
        kb_id=kb_id, status=status, keyword=keyword, page=page, page_size=page_size
    )
    return ResponseModel(
        data=PageData(
            total=total,
            items=[DocOut.model_validate(item) for item in items],
            page=page,
            page_size=page_size,
        )
    )


@router.post("/upload", response_model=ResponseModel)
async def upload_files(
    kb_id: int = Form(...),
    files: list[UploadFile] = File(...),
    repos: Repositories = Depends(get_repositories),
    user: User = Depends(get_current_user),
):
    """上传文件到指定知识库（流式落盘 + 大小限制，随后交给任务队列处理）。"""
    ensure_kb_access(repos.kbs, kb_id, user, AccessLevel.WRITE)
    if not files:
        raise HTTPException(status_code=400, detail="未选择文件")

    items: list[NewDocument] = []
    try:
        for upload in files:
            filename = upload.filename or ""
            if not doc_service.is_supported_file(filename):
                raise HTTPException(status_code=400, detail=f"不支持的文件类型: {filename}")

            file_path, safe_name, size = await doc_service.save_upload(upload, kb_id)
            items.append(
                NewDocument(
                    kb_id=kb_id,
                    filename=safe_name,
                    file_path=file_path,
                    file_type=parser.get_file_type(safe_name),
                    file_size=size,
                    source_type="upload",
                )
            )
    except ValueError as exc:
        # 文件过大等校验失败：这一批文件全部不受理（磁盘上已写入的部分由 save_upload 自己清理）
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # 一次提交整批：要么全受理、要么全不受理，不留「半个批次」的文档记录
    doc_ids = repos.docs.add_many(items)

    # 任务只接收 doc_id，自己创建会话（请求级 Session 在响应返回后即关闭）
    for doc_id in doc_ids:
        await doc_service.enqueue_processing(doc_id)

    logger.info(f"[DOC] 上传 {len(doc_ids)} 个文件到 kb={kb_id} user={user.username}")
    return ResponseModel(
        message=f"成功上传 {len(doc_ids)} 个文件，正在后台处理中",
        data={"doc_ids": doc_ids},
    )


@router.post("/import-path", response_model=ResponseModel)
async def import_from_local_path(
    body: LocalPathImport,
    repos: Repositories = Depends(get_repositories),
    user: User = Depends(get_current_user),
):
    """从服务器本地路径导入文件（默认禁用，需 ALLOW_LOCAL_IMPORT=true）。"""
    ensure_kb_access(repos.kbs, body.kb_id, user, AccessLevel.WRITE)

    try:
        doc_service.validate_import_source(body.path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    await doc_service.enqueue_import(body.kb_id, body.path, body.recursive)
    return ResponseModel(message=f"正在导入路径: {body.path}")


@router.get("/{doc_id}", response_model=ResponseModel)
def get_doc(
    doc_id: int,
    repos: Repositories = Depends(get_repositories),
    user: User = Depends(get_current_user),
):
    doc = ensure_doc_access(repos.docs, repos.kbs, doc_id, user, AccessLevel.READ)
    return ResponseModel(data=DocOut.model_validate(doc))


@router.delete("/{doc_id}", response_model=ResponseModel)
async def delete_doc(
    doc_id: int,
    repos: Repositories = Depends(get_repositories),
    user: User = Depends(get_current_user),
):
    ensure_doc_access(repos.docs, repos.kbs, doc_id, user, AccessLevel.WRITE)
    await doc_service.delete_document(repos.docs, doc_id)
    return ResponseModel(message="删除成功")


@router.post("/{doc_id}/reprocess", response_model=ResponseModel)
async def reprocess_doc(
    doc_id: int,
    repos: Repositories = Depends(get_repositories),
    user: User = Depends(get_current_user),
):
    """重新处理文档（失败后重试，或强制重建索引）。"""
    doc: Document = ensure_doc_access(repos.docs, repos.kbs, doc_id, user, AccessLevel.WRITE)

    repos.docs.update_status(doc, DocStatus.PENDING)
    await doc_service.enqueue_processing(doc_id)
    return ResponseModel(message="已重新提交处理")
