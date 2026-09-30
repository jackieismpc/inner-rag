#!/bin/bash
# 查看本地 LLM 推理服务状态
# 用法：bash scripts/llm_status.sh
PORT=8000
PID=$(ps aux | grep "[l]aunch_server" | awk "{print \$2}" | head -1)

if [ -n "$PID" ]; then
  ETIME=$(ps -o etime= -p "$PID" 2>/dev/null | tr -d " ")
  echo "[llm] 状态：运行中（PID $PID，已运行 ${ETIME:-未知}）"
  # GPU 显存占用（进程本体的 rss 不含显存，显存在 GPU 上）
  if command -v nvidia-smi >/dev/null 2>&1; then
    VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk "{s+=\$1} END {printf \"%.1f GB\", s/1024}")
    echo "[llm] GPU 显存总占用：$VRAM"
  fi
  if curl -s --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "[llm] 就绪：http://127.0.0.1:$PORT/health 返回 200 ✅"
  else
    echo "[llm] 进程在，但 health 尚未就绪（可能仍在加载）⚠️"
  fi
else
  echo "[llm] 状态：未运行 ❌（用 bash scripts/llm_start.sh 启动）"
fi
