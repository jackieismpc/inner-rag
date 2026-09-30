"""inner-rag 知识库系统 — FastAPI 入口。"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from inner_rag.api import auth, chat, document, kb, system
from inner_rag.core.config import settings
from inner_rag.core.context import IdentityContextMiddleware
from inner_rag.core.database import init_db
from inner_rag.core.logging import RequestIdMiddleware, configure_logging
from inner_rag.core.observability import tracer
from inner_rag.core.security import verify_production_secret
from inner_rag.providers import ProviderError, chat_health, embedding_health
from inner_rag.services.retrieval_log import setup_rag_loggers

# 日志格式（text / json）与 sink 由 core/logging.py 统一配置：
# 那里还要负责把 request_id 与 user 注入每条日志，放在入口处只是「什么时候配」的决定。
configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("启动 inner-rag 服务 ...")
    # 配置边界校验：生产用默认签名密钥直接拒绝启动
    verify_production_secret()
    init_db()
    setup_rag_loggers()
    tracer.configure()
    logger.info(
        f"追踪: {tracer.status()['backend']} | 日志格式: {settings.LOG_FORMAT} | "
        f"指标后端: {settings.METRICS_BACKEND}"
    )
    # 启动时只做配置校验（不发网络请求），配置有问题不阻断启动：
    # 服务照常起，/api/system/health 会给出可读原因
    for kind, health in (("LLM", chat_health(probe=False)), ("Embedding", embedding_health())):
        if health["ok"]:
            logger.info(f"{kind} provider: {health['provider']} / {health['model']}")
        else:
            logger.warning(f"{kind} provider 配置有误: {health['error']}")
    logger.info(f"OCR 后端: {settings.OCR_BACKEND}")
    yield
    logger.info("服务已停止")


app = FastAPI(
    title="inner-rag Knowledge Base",
    description="可插拔模型后端的 RAG 知识库系统",
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs" if settings.ENABLE_DOCS else None,
    redoc_url="/redoc" if settings.ENABLE_DOCS else None,
    openapi_url="/openapi.json" if settings.ENABLE_DOCS else None,
)

# 后添加的在外层：CORS 要先看到请求（含预检），request_id 必须在最外层
# ——否则被 CORS 拦下的预检请求就没有 request_id，访问日志会缺一条。
app.add_middleware(IdentityContextMiddleware)
app.add_middleware(
    CORSMiddleware,
    # 白名单化，避免 allow_origins=["*"] + allow_credentials=True 这种无效且不安全的组合
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestIdMiddleware)


@app.exception_handler(ProviderError)
async def provider_error_handler(request: Request, exc: ProviderError):
    """模型后端没配好（未知 provider / 缺 API Key）：503 + 可直接照做的文案。"""
    logger.error(f"provider 配置错误 {request.method} {request.url.path}: {exc}")
    return JSONResponse(status_code=503, content={"code": 503, "message": str(exc), "data": None})


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"未处理异常 {request.method} {request.url.path}: {exc}")
    return JSONResponse(status_code=500, content={"code": 500, "message": str(exc), "data": None})


app.include_router(auth.router)
app.include_router(kb.router)
app.include_router(document.router)
app.include_router(chat.router)
app.include_router(system.router)


@app.get("/")
async def root():
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs" if settings.ENABLE_DOCS else None,
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
