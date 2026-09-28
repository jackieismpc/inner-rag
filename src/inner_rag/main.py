"""inner-rag 知识库系统 — FastAPI 入口。"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from inner_rag.api import chat, document, kb, system
from inner_rag.core.config import settings
from inner_rag.core.database import init_db
from inner_rag.services.retrieval_log import setup_rag_loggers

logger.remove()
logger.add(
    sys.stdout,
    level=settings.LOG_LEVEL,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {message}",
)
logger.add(
    f"{settings.LOG_DIR}/app.log",
    rotation="10 MB",
    retention="7 days",
    level=settings.LOG_LEVEL,
    encoding="utf-8",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("启动 inner-rag 服务 ...")
    init_db()
    setup_rag_loggers()
    logger.info(f"LLM 模型: {settings.OLLAMA_LLM_MODEL}")
    logger.info(f"Embedding 模型: {settings.OLLAMA_EMBEDDING_MODEL}")
    logger.info(f"OCR 后端: {settings.OCR_BACKEND}")
    yield
    logger.info("服务已停止")


app = FastAPI(
    title="inner-rag Knowledge Base",
    description="可插拔模型后端的 RAG 知识库系统",
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    # 白名单化，避免 allow_origins=["*"] + allow_credentials=True 这种无效且不安全的组合
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"未处理异常 {request.method} {request.url.path}: {exc}")
    return JSONResponse(status_code=500, content={"code": 500, "message": str(exc), "data": None})


app.include_router(kb.router)
app.include_router(document.router)
app.include_router(chat.router)
app.include_router(system.router)


@app.get("/")
async def root():
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "inner_rag.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
    )
