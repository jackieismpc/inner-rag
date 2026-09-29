# syntax=docker/dockerfile:1
# 后端镜像：uv 管理依赖，多阶段构建（依赖层与源码层分开，改代码不必重装依赖）
FROM python:3.13-slim AS base
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"
WORKDIR /app
# git 供部分依赖从 VCS 安装；libgomp1 是 chromadb/onnxruntime 运行期需要的
RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app

# 1) 只装依赖：这一层在只改源码时会命中缓存
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev

# 2) 再装项目本身
COPY src ./src
COPY alembic.ini ./
COPY migrations ./migrations
RUN uv sync --frozen --no-dev

# 3) 运行期目录（挂载卷时会覆盖）
ENV HOST=0.0.0.0 \
    PORT=8010 \
    UPLOAD_DIR=/data/uploads \
    ZVEC_PATH=/data/zvec_db \
    CHROMA_PERSIST_DIR=/data/chroma_db \
    LOG_DIR=/data/logs
RUN mkdir -p /data/uploads /data/zvec_db /data/chroma_db /data/logs \
    && chown -R app:app /data /app

# 以非 root 身份运行；具名卷首次挂载时会继承 /data 的属主
USER app
VOLUME ["/data"]

EXPOSE 8010
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import httpx,os; httpx.get(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/system/health', timeout=4)"

# 启动前先跑迁移，保证容器内 schema 永远是最新的
CMD ["sh", "-c", "alembic upgrade head && uvicorn inner_rag.main:app --host 0.0.0.0 --port ${PORT}"]
