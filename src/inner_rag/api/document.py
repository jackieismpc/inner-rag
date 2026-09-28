"""文档管理 API：上传、路径导入、查询、删除、重新处理。"""

from __future__ import annotations

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from loguru import logger
from sqlalchemy.orm import Session

from inner_rag.api.deps import ensure_doc_access, ensure_kb_access, get_current_user
from inner_rag.core.access import AccessLevel
from inner_rag.core.database import get_db
from inner_rag.models import DocStatus, Document, User
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
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    ensure_kb_access(db, kb_id, user, AccessLevel.READ)
    query = db.query(Document).filter(Document.kb_id == kb_id)
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
    background_tasks: BackgroundTasks,
    kb_id: int = Form(...),
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """上传文件到指定知识库（流式落盘 + 大小限制，随后交给后台任务处理）。"""
    ensure_kb_access(db, kb_id, user, AccessLevel.WRITE)
    if not files:
        raise HTTPException(status_code=400, detail="未选择文件")

    doc_ids: list[int] = []
    try:
        for upload in files:
            filename = upload.filename or ""
            if not doc_service.is_supported_file(filename):
                raise HTTPException(status_code=400, detail=f"不支持的文件类型: {filename}")

            file_path, safe_name, size = await doc_service.save_upload(upload, kb_id)
            doc = Document(
                kb_id=kb_id,
                filename=safe_name,
                file_path=file_path,
                file_type=parser.get_file_type(safe_name),
                file_size=size,
                source_type="upload",
                status=DocStatus.PENDING,
            )
            db.add(doc)
            db.flush()
            doc_ids.append(doc.id)
        db.commit()
    except HTTPException:
        db.rollback()
        raise
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # 后台任务只接收 doc_id，自己创建会话（请求级 Session 在响应返回后即关闭）
    for doc_id in doc_ids:
        background_tasks.add_task(doc_service.process_document, doc_id)

    logger.info(f"[DOC] 上传 {len(doc_ids)} 个文件到 kb={kb_id} user={user.username}")
    return ResponseModel(
        message=f"成功上传 {len(doc_ids)} 个文件，正在后台处理中",
        data={"doc_ids": doc_ids},
    )


@router.post("/import-path", response_model=ResponseModel)
async def import_from_local_path(
    background_tasks: BackgroundTasks,
    body: LocalPathImport,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """从服务器本地路径导入文件（默认禁用，需 ALLOW_LOCAL_IMPORT=true）。"""
    ensure_kb_access(db, body.kb_id, user, AccessLevel.WRITE)

    try:
        doc_service.validate_import_source(body.path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    background_tasks.add_task(doc_service.import_from_path, body.kb_id, body.path, body.recursive)
    return ResponseModel(message=f"正在导入路径: {body.path}")


@router.get("/{doc_id}", response_model=ResponseModel)
def get_doc(
    doc_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    doc = ensure_doc_access(db, doc_id, user, AccessLevel.READ)
    return ResponseModel(data=DocOut.model_validate(doc))


@router.delete("/{doc_id}", response_model=ResponseModel)
def delete_doc(
    doc_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    ensure_doc_access(db, doc_id, user, AccessLevel.WRITE)
    doc_service.delete_document(db, doc_id)
    return ResponseModel(message="删除成功")


@router.post("/{doc_id}/reprocess", response_model=ResponseModel)
async def reprocess_doc(
    doc_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """重新处理文档（失败后重试，或强制重建索引）。"""
    doc = ensure_doc_access(db, doc_id, user, AccessLevel.WRITE)

    doc.status = DocStatus.PENDING
    doc.error_msg = None
    db.commit()

    background_tasks.add_task(doc_service.process_document, doc_id)
    return ResponseModel(message="已重新提交处理")
