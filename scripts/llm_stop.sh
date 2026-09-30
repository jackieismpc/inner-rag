#!/bin/bash
# 关闭本地 LLM 推理服务（SGLang）
# 用法：bash scripts/llm_stop.sh
set -e
# 用 [l]aunch_server 避免 pkill 匹配到本脚本自身
PIDS=$(ps aux | grep "[l]aunch_server" | awk "{print \$2}")
if [ -z "$PIDS" ]; then
  echo "[llm] 没有正在运行的推理服务"
  exit 0
fi
echo "[llm] 关闭推理服务进程：$PIDS"
echo "$PIDS" | xargs -r kill
sleep 2
echo "[llm] 已关闭"
