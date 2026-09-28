#!/usr/bin/env bash
# 启动后端服务（开发模式）
#
#   ./scripts/start.sh            # 迁移数据库 + 热重载启动
#   SKIP_MIGRATE=1 ./scripts/start.sh
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
  echo "提示: 未找到 .env，正在从 .env.example 复制（请按需修改）"
  cp .env.example .env
fi

echo "同步依赖 (uv sync)..."
uv sync

if [ "${SKIP_MIGRATE:-0}" != "1" ]; then
  echo "应用数据库迁移 (alembic upgrade head)..."
  uv run alembic upgrade head
fi

PORT="$(uv run python -c 'from inner_rag.core.config import settings; print(settings.PORT)')"
echo "启动 FastAPI: http://127.0.0.1:${PORT}/docs"
exec uv run uvicorn inner_rag.main:app --host 0.0.0.0 --port "$PORT" --reload --log-level info
