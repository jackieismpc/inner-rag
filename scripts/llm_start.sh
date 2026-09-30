#!/bin/bash
# 启动本地 LLM 推理服务（SGLang 加载 Qwen2.5-32B-Instruct-GPTQ-Int4）
# 用法：bash scripts/llm_start.sh [GPU_ID]
set -e
GPU_ID="${1:-2}"
SGLANG_PY="/data/users/mapengcheng/anaconda3/envs/sglang/bin/python"
MODEL_DIR="$HOME/rustproject/qwen25-32b-gptq"
LOG="$HOME/rustproject/inner-rag/logs/sglang-qwen32b.log"
PORT=8000
MAX_WAIT=300   # 最长等待秒数（32B 首载约 1-2 分钟，留足余量）

mkdir -p "$(dirname "$LOG")"

if curl -s --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "[llm] 服务已在运行（http://127.0.0.1:$PORT/health 返回 200）"
  exit 0
fi

echo "[llm] 启动 SGLang（GPU $GPU_ID，模型 $MODEL_DIR）..."
# 注意：nohup env CUDA_VISIBLE_DEVICES=... 的写法，env 必须在 nohup 之后，否则变量被 nohup 吞掉
nohup env CUDA_VISIBLE_DEVICES="$GPU_ID" "$SGLANG_PY" -m sglang.launch_server \
  --model-path "$MODEL_DIR" \
  --quantization gptq_marlin \
  --trust-remote-code \
  --host 127.0.0.1 --port "$PORT" \
  --dtype float16 \
  --attention-backend triton \
  --sampling-backend pytorch \
  > "$LOG" 2>&1 &

echo "[llm] 已后台启动，日志：$LOG"
echo "[llm] 等待模型加载（约 1-2 分钟）..."

# 轮询 health 直到就绪或超时
elapsed=0
while [ "$elapsed" -lt "$MAX_WAIT" ]; do
  if curl -s --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "[llm] ✅ 服务已就绪（${elapsed}s）"
    exit 0
  fi
  # 若进程已退出，说明启动失败，直接报错
  if ! ps aux | grep -q "[l]aunch_server.*--port $PORT"; then
    echo "[llm] ❌ 启动失败，进程已退出，最后几行日志："
    tail -n 20 "$LOG"
    exit 1
  fi
  sleep 5
  elapsed=$((elapsed + 5))
done

echo "[llm] ❌ 等待 ${MAX_WAIT}s 仍未就绪，请查看日志：$LOG"
exit 1
