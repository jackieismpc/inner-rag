#!/bin/bash
# 启动本地 LLM 推理服务（SGLang 加载 Qwen2.5-32B-Instruct-GPTQ-Int4）
# 用法：bash scripts/llm_start.sh [GPU_ID]
set -e
GPU_ID="${1:-2}"
SGLANG_PY="/data/users/mapengcheng/anaconda3/envs/sglang/bin/python"
MODEL_DIR="$HOME/rustproject/qwen25-32b-gptq"
LOG="$HOME/rustproject/inner-rag/logs/sglang-qwen32b.log"

mkdir -p "$(dirname "$LOG")"

if curl -s --max-time 2 http://127.0.0.1:8000/health >/dev/null 2>&1; then
  echo "[llm] 服务已在运行（http://127.0.0.1:8000/health 返回 200）"
  exit 0
fi

echo "[llm] 启动 SGLang（GPU $GPU_ID，模型 $MODEL_DIR）..."
# 注意：nohup env CUDA_VISIBLE_DEVICES=... 的写法，env 必须在 nohup 之后，否则变量被 nohup 吞掉
nohup env CUDA_VISIBLE_DEVICES="$GPU_ID" "$SGLANG_PY" -m sglang.launch_server \
  --model-path "$MODEL_DIR" \
  --quantization gptq_marlin \
  --trust-remote-code \
  --host 127.0.0.1 --port 8000 \
  --dtype float16 \
  --attention-backend triton \
  --sampling-backend pytorch \
  > "$LOG" 2>&1 &

echo "[llm] 已后台启动，日志：$LOG"
echo "[llm] 等待模型加载（约 1-2 分钟），可用 curl http://127.0.0.1:8000/health 探测"
