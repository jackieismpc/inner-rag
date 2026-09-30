"""文档处理服务：解析 -> 分块 -> 向量化入库，以及删除。

约定：

* 所有对外方法自行创建并关闭数据库会话（后台任务与脚本都能安全调用，不会出现
  「请求级 Session 在响应返回时已关闭，后台任务却还在使用它」的问题）；会话只用来
  构造仓储，业务代码不直接拼查询；
* 处理入口 ``process_document`` 失败时**抛异常**（原因同时写进 ``document.error_msg``）：
  任务队列靠这个异常决定是否重试，脚本调用方也能看见失败而不是静默返回；
* 批量场景（HTTP 上传、路径导入）统一走 ``enqueue_processing`` / ``enqueue_import``，
  由它们把失败收敛成日志——一个文档失败不该让整批导入重跑（重跑会重复落库）。
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import UploadFile
from loguru import logger

from inner_rag.core.config import settings
from inner_rag.core.database import SessionLocal
from inner_rag.core.metrics import metrics
from inner_rag.core.observability import tracer
from inner_rag.models import DocStatus
from inner_rag.repositories import DocumentRepository, NewDocument, Repositories, build_repositories
from inner_rag.services.cache import query_cache
from inner_rag.services.embedding import ensure_embedding_matches
from inner_rag.services.lexical import lexical_index
from inner_rag.services.parser import parser
from inner_rag.services.task_queue import task_queue
from inner_rag.services.vector_store import vector_service

UPLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MiB


class DocumentProcessingError(RuntimeError):
    """文档处理失败。原因已写进 ``document.error_msg``（用户可见）与日志。"""


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

    # ── 任务提交 ───────────────────────────────────────────────────────

    async def enqueue_processing(self, doc_id: int) -> str | None:
        """把「处理该文档」交给任务队列。

        返回 task_id（``inline`` 后端返回的 id 没有查询意义）。失败原因已写进文档状态，
        这里只记日志：批量上传 / 导入的语义是「已受理」，不该因为其中一个文档失败而整体报错，
        更不该让整批导入被队列重跑（重跑会重复落库）。
        """
        try:
            return await task_queue.submit("doc.process", self.process_document, doc_id)
        except DocumentProcessingError as exc:
            logger.error(f"[DOC] 处理失败 doc_id={doc_id}: {exc}")
            return None

    async def enqueue_import(self, kb_id: int, path: str, recursive: bool) -> str | None:
        """把「按路径导入」交给任务队列。

        与 ``enqueue_processing`` 同理：导入是「已受理」语义，中途失败（路径消失、权限变化）
        只记日志——用户在文档列表里能直接看到哪些文档 ``failed``。
        """
        try:
            return await task_queue.submit(
                "doc.import_path", self.import_from_path, kb_id, path, recursive
            )
        except Exception as exc:
            logger.error(f"[DOC] 导入失败 kb={kb_id} path={path!r}: {exc}")
            return None

    # ── 处理流程 ───────────────────────────────────────────────────────

    async def process_document(self, doc_id: int) -> None:
        """任务入口：自建会话，处理单个文档；失败抛 ``DocumentProcessingError``。"""
        db = SessionLocal()
        try:
            completed = await self._process(build_repositories(db), doc_id)
        finally:
            db.close()
        if not completed:
            raise DocumentProcessingError(f"文档处理失败（详见文档状态）: doc_id={doc_id}")

    async def _process(self, repos: Repositories, doc_id: int) -> bool:
        """处理单个文档，返回是否完成（失败已写进 ``document.error_msg``）。"""
        docs = repos.docs
        doc = docs.get(doc_id)
        if doc is None:
            # 文档在排队期间被删掉是正常情况（删除文档与重新处理可以并发发生）：
            # 这里当作「无需处理」，既不该重试也不该报错
            logger.warning(f"[DOC] 文档已不存在，跳过处理: doc_id={doc_id}")
            return True

        docs.update_status(doc, DocStatus.PROCESSING)

        try:
            kb = repos.kbs.get(doc.kb_id)
            ensure_embedding_matches(doc.kb_id, kb.embedding_model if kb else None)

            if not doc.file_path:
                msg = f"文档缺少服务端存储路径，无法解析: doc_id={doc.id}"
                raise ValueError(msg)
            # 入库链路是一棵独立的 trace 树（kind=ingest）：它跑在后台任务里，
            # 没有 HTTP 响应可挂，只能靠 request_id 与上传请求对上。
            async with tracer.span(
                "ingest.document",
                kind="ingest",
                kb_id=doc.kb_id,
                doc_id=doc.id,
                filename=doc.filename,
                ocr_backend=settings.OCR_BACKEND,
            ) as root:
                async with tracer.span("parse", filename=doc.filename) as parse_span:
                    documents, meta = parser.parse(doc.file_path, doc.filename)
                    parse_span.set(chunk_count=len(documents), chars=meta.get("total_chars", 0))
                async with tracer.span(
                    "vector.ingest", kb_id=doc.kb_id, chunk_count=len(documents)
                ):
                    chunk_count = await vector_service.add_documents(
                        kb_id=doc.kb_id, documents=documents, doc_id=doc.id, filename=doc.filename
                    )
                root.set(chunk_count=chunk_count, chars=meta.get("total_chars", 0))

            docs.mark_completed(
                doc,
                chunk_count=chunk_count,
                char_count=meta.get("total_chars", 0),
                meta=meta,
            )
            docs.sync_kb_doc_count(doc.kb_id)

            # 新内容入库后必须让检索缓存失效，否则会持续返回旧结果
            # （词面索引同样由库内容派生，漏掉它会让 BM25 一路继续命中过期分块）
            await query_cache.invalidate_kb(doc.kb_id)
            lexical_index.invalidate(doc.kb_id)
            metrics.increment("rag_ingest_documents_total", labels={"status": "completed"})
            logger.info(f"[DOC] 处理完成: {doc.filename}, chunks={chunk_count}")
            return True
        except Exception as exc:
            # 失败原因来自解析 / 嵌入 / 向量写入，都不涉及数据库事务，因此直接做状态迁移即可
            # （不做 rollback：那会把上游错误掩盖成「会话已回滚」这类次生问题）
            logger.error(f"[DOC] 处理失败 doc_id={doc_id}: {exc}")
            metrics.increment("rag_ingest_documents_total", labels={"status": "failed"})
            docs.update_status(doc, DocStatus.FAILED, error_msg=str(exc)[:500])
            return False

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

        items: list[NewDocument] = []
        for file in files:
            if not self.is_supported_file(file.name):
                logger.warning(f"[DOC] 跳过不支持的文件: {file.name}")
                continue
            destination = Path(self.get_upload_path(kb_id, file.name))
            if file != destination:
                shutil.copy2(file, destination)
            items.append(
                NewDocument(
                    kb_id=kb_id,
                    filename=file.name,
                    file_path=str(destination),
                    file_type=parser.get_file_type(file.name),
                    file_size=file.stat().st_size,
                    source_type="local_path",
                )
            )

        db = SessionLocal()
        try:
            doc_ids = build_repositories(db).docs.add_many(items)
        finally:
            db.close()

        for doc_id in doc_ids:
            await self.enqueue_processing(doc_id)
        return doc_ids

    # ── 删除 ───────────────────────────────────────────────────────────

    async def delete_document(self, docs: DocumentRepository, doc_id: int) -> bool:
        doc = docs.get(doc_id)
        if doc is None:
            return False

        kb_id = doc.kb_id
        await vector_service.delete_document(kb_id, doc_id)

        if doc.file_path:
            Path(doc.file_path).unlink(missing_ok=True)

        docs.delete(doc)
        docs.sync_kb_doc_count(kb_id)
        query_cache.invalidate_kb_sync(kb_id)
        lexical_index.invalidate(kb_id)
        return True


doc_service = DocumentService()
