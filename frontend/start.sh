#!/bin/bash
# 启动前端开发服务（默认 3000 端口，通过 Vite 代理访问后端）

set -e
cd "$(dirname "$0")"

echo "启动 inner-rag 前端..."

if [ ! -d "node_modules" ]; then
  echo "📥 安装 npm 依赖..."
  npm install
fi

echo "启动 Vue3 开发服务 http://localhost:3000"
npm run dev
