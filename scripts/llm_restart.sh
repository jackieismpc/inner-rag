#!/bin/bash
# 一键重启本地 LLM 推理服务
# 用法：bash scripts/llm_restart.sh [GPU_ID]
set -e
GPU_ID="${1:-2}"
echo "[llm] 重启：先关闭..."
bash "$(dirname "$0")/llm_stop.sh"
echo "[llm] 再启动..."
bash "$(dirname "$0")/llm_start.sh" "$GPU_ID"
