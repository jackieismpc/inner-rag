"""文档处理服务：解析 -> 分块 -> 向量化入库，以及删除。

约定：所有对外方法自行创建并关闭数据库会话。这样后台任务、脚本与 API 层都能安全调用，
不会出现「请求级 Session 在响应返回时已关闭，后台任务却还在使用它」的问题。
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import UploadFile
from loguru import logger
from sqlalchemy.orm import Session

from inner_rag.core.config import settings
from inner_rag.core.database import SessionLocal
from inner_rag.models import DocStatus, Document, KnowledgeBase
from inner_rag.services.cache import query_cache
from inner_rag.services.parser import parser
from inner_rag.services.vector_store import vector_service

UPLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MiB


class DocumentService:
    # ── 路径与校验 ─────────────────────────────────────────────────────

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """只保留文件名本身，去掉路径分隔符与控制字符，避免路径穿越。"""
        name = Path(filename.replace("\\", "/")).name.strip()
        cleaned = "".join(ch for ch in name if ch not in "\0\r\n\t")
        return cleaned or "unnamed"

    def get_upload_path(self, kb_id: int, filename: str) -> str:
        kb_dir = Path(settings.UPLOAD_DIR) / f"kb_{kb_id}"
        kb_dir.mkdir(parents=True, exist_ok=True)
        return str(kb_dir / self.sanitize_filename(filename))

    def is_supported_file(self, filename: str) -> bool:
        ext = Path(filename).suffix.lower().lstrip(".")
        return ext in settings.allowed_extensions_list

    async def save_upload(self, upload: UploadFile, kb_id: int) -> tuple[str, str, int]:
        """流式写盘并即时校验大小，返回 (存储路径, 原始文件名, 字节数)。

        旧实现先把整个文件读进内存再判断大小，大文件会直接吃掉内存。
        """
        original_name = self.sanitize_filename(upload.filename or "unnamed")
        stored_name = f"{uuid.uuid4().hex[:8]}_{original_name}"
        dest_path = self.get_upload_path(kb_id, stored_name)
        size = 0
        try:
            with open(dest_path, "wb") as out:
                while chunk := await upload.read(UPLOAD_CHUNK_SIZE):
                    size += len(chunk)
                    if size > settings.MAX_FILE_SIZE:
                        msg = (
                            f"文件过大: {original_name}"
                            f"（上限 {settings.MAX_FILE_SIZE // 1024 // 1024} MB）"
                        )
                        raise ValueError(msg)
                    out.write(chunk)
        except Exception:
            Path(dest_path).unlink(missing_ok=True)
            raise
        return dest_path, original_name, size

    # ── 处理流程 ───────────────────────────────────────────────────────

    async def process_document(self, doc_id: int) -> None:
        """后台任务入口：自建会话，处理单个文档。"""
        db = SessionLocal()
        try:
            await self._process(db, doc_id)
        finally:
            db.close()

    async def _process(self, db: Session, doc_id: int) -> None:
        doc = db.get(Document, doc_id)
        if doc is None:
            logger.error(f"[DOC] 文档不存在: {doc_id}")
            return

        doc.status = DocStatus.PROCESSING
        doc.error_msg = None
        db.commit()

        try:
            kb = db.get(KnowledgeBase, doc.kb_id)
            vector_service.ensure_embedding_matches(doc.kb_id, kb.embedding_model if kb else None)

            if not doc.file_path:
                msg = f"文档缺少服务端存储路径，无法解析: doc_id={doc.id}"
                raise ValueError(msg)
            documents, meta = parser.parse(doc.file_path, doc.filename)
            chunk_count = await vector_service.add_documents_async(
                kb_id=doc.kb_id, documents=documents, doc_id=doc.id, filename=doc.filename
            )

            doc.status = DocStatus.COMPLETED
            doc.chunk_count = chunk_count
            doc.char_count = meta.get("total_chars", 0)
            doc.meta_info = meta
            db.commit()
            self._update_kb_doc_count(db, doc.kb_id)

            # 新内容入库后必须让检索缓存失效，否则会持续返回旧结果
            await query_cache.invalidate_kb(doc.kb_id)
            logger.info(f"[DOC] 处理完成: {doc.filename}, chunks={chunk_count}")
        except Exception as exc:
            logger.error(f"[DOC] 处理失败 doc_id={doc_id}: {exc}")
            db.rollback()
            self._mark_failed(db, doc_id, exc)

    @staticmethod
    def _mark_failed(db: Session, doc_id: int, exc: Exception) -> None:
        row = db.get(Document, doc_id)
        if row is None:
            return
        row.status = DocStatus.FAILED
        row.error_msg = str(exc)[:500]
        db.commit()

    @staticmethod
    def _update_kb_doc_count(db: Session, kb_id: int) -> None:
        count = (
            db.query(Document)
            .filter(Document.kb_id == kb_id, Document.status == DocStatus.COMPLETED)
            .count()
        )
        db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).update({"doc_count": count})
        db.commit()

    # ── 本地路径导入 ───────────────────────────────────────────────────

    @staticmethod
    def validate_import_source(path: str) -> Path:
        """校验本地导入路径：开关、存在性、根目录限制。返回解析后的绝对路径。"""
        if not settings.ALLOW_LOCAL_IMPORT:
            msg = "服务端路径导入已禁用（安全考虑）；如需启用请设置 ALLOW_LOCAL_IMPORT=true"
            raise PermissionError(msg)

        source = Path(path).expanduser().resolve()
        if not source.exists():
            msg = f"路径不存在: {path}"
            raise FileNotFoundError(msg)

        if settings.LOCAL_IMPORT_ROOT:
            root = Path(settings.LOCAL_IMPORT_ROOT).expanduser().resolve()
            if root != source and root not in source.parents:
                msg = f"路径 {source} 不在允许导入的根目录 {root} 内"
                raise PermissionError(msg)
        return source

    async def import_from_path(self, kb_id: int, path: str, recursive: bool = True) -> list[int]:
        """从服务器本地路径导入文件（默认关闭，见 ALLOW_LOCAL_IMPORT）。"""
        source = self.validate_import_source(path)

        if source.is_file():
            files = [source]
        else:
            pattern = "**/*" if recursive else "*"
            files = sorted(item for item in source.glob(pattern) if item.is_file())

        db = SessionLocal()
        try:
            doc_ids: list[int] = []
            for file in files:
                if not self.is_supported_file(file.name):
                    logger.warning(f"[DOC] 跳过不支持的文件: {file.name}")
                    continue
                destination = Path(self.get_upload_path(kb_id, file.name))
                if file != destination:
                    shutil.copy2(file, destination)
                doc = Document(
                    kb_id=kb_id,
                    filename=file.name,
                    file_path=str(destination),
                    file_type=parser.get_file_type(file.name),
                    file_size=file.stat().st_size,
                    source_type="local_path",
                    status=DocStatus.PENDING,
                )
                db.add(doc)
                db.flush()
                doc_ids.append(doc.id)
            db.commit()
        finally:
            db.close()

        for doc_id in doc_ids:
            await self.process_document(doc_id)
        return doc_ids

    # ── 删除 ───────────────────────────────────────────────────────────

    def delete_document(self, db: Session, doc_id: int) -> bool:
        doc = db.get(Document, doc_id)
        if doc is None:
            return False

        kb_id = doc.kb_id
        vector_service.delete_documents(kb_id, doc_id)

        if doc.file_path:
            Path(doc.file_path).unlink(missing_ok=True)

        db.delete(doc)
        db.commit()
        self._update_kb_doc_count(db, kb_id)
        query_cache.invalidate_kb_sync(kb_id)
        return True


doc_service = DocumentService()
