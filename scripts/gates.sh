#!/usr/bin/env bash
# 门禁脚本：把 docs/DEVELOPMENT_PLAN.md §5 的 G0 / G1 / G2 固化成可执行命令。
#
#   ./scripts/gates.sh g0      # 每次提交：静态检查 + 评测离线自检
#   ./scripts/gates.sh g1      # push 前 / 阶段收尾：G0 + mypy + 离线全量 + 迁移自检 + 冒烟 + changelog / 密钥自检
#   ./scripts/gates.sh g2      # 真实链路验收：需要 .env 里的 Key，会产生模型调用费用
#   ./scripts/gates.sh all     # g0 → g1 → g2
#
# 设计取舍：
# - 用 `uv run` 而不是「先激活 venv」：与 README / 计划的命令逐字一致，可直接复制，避免两套口径。
# - G1 的迁移自检要改 DATABASE_URL / 目录到临时路径：必须证明「干净库从零建表」也成立，
#   不能依赖开发者本机 ./data 里已有的库（否则换机器就炸，而这正是要防的失败）。
# - changelog 检查放在 G1：AGENTS.md §1.1 把「变更留痕」定为 push 前硬门禁，只有放进脚本才拦得住。
# - 冒烟用独立端口 8011 并落临时目录：不污染开发中的 8010 与 ./data。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LEVEL="${1:-g0}"
GATE_TMP="${GATE_TMP:-/tmp/ir-gate}"
GATE_BASE="${GATE_BASE:-origin/main}"

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
ok()  { printf '\033[1;32m[ok]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }

g0() {
  log "G0 · 静态检查"
  uv run ruff check .
  uv run ruff format --check .
  ok "ruff / ruff format"

  log "G0 · 评测集与指标算法离线自检（mock provider，不联网、不写 README）"
  uv run python -m benchmark.run_bench --mode fixtures
  ok "G0 通过"
}

g1() {
  g0

  log "G1 · 类型检查（mypy）"
  uv run mypy
  ok "mypy"

  log "G1 · 全量离线测试（不依赖网络与外部服务）"
  uv run pytest -q
  ok "pytest（离线）"

  # 迁移与冒烟都跑在临时环境上：干净库 + 临时目录，证明不依赖本机既存数据。
  rm -rf "$GATE_TMP"
  mkdir -p "$GATE_TMP"
  export DATABASE_URL="sqlite:///$GATE_TMP/app.db" \
         UPLOAD_DIR="$GATE_TMP/uploads" \
         ZVEC_PATH="$GATE_TMP/zvec" \
         CHROMA_PERSIST_DIR="$GATE_TMP/chroma" \
         LOG_DIR="$GATE_TMP/logs"

  log "G1 · 迁移自检（干净库：upgrade → check → downgrade → upgrade）"
  uv run alembic upgrade head
  uv run alembic check
  uv run alembic downgrade base
  uv run alembic upgrade head
  ok "Alembic 迁移可建、无漂移、可回退、可重放"

  log "G1 · 冒烟启动 + 探活（临时端口 8011）"
  uv run uvicorn inner_rag.main:app --port 8011 >"$GATE_TMP/uvicorn.log" 2>&1 &
  local srv=$!
  trap 'kill "$srv" 2>/dev/null || true' EXIT
  for _ in $(seq 1 20); do
    curl --noproxy '*' -sf localhost:8011/api/system/health >/dev/null 2>&1 && break
    sleep 1
  done
  curl --noproxy '*' -sf localhost:8011/api/system/health >"$GATE_TMP/health.json" \
    || die "探活失败，日志见 $GATE_TMP/uvicorn.log"
  curl --noproxy '*' -sf -o /dev/null localhost:8011/openapi.json \
    || die "openapi.json 不可用，日志见 $GATE_TMP/uvicorn.log"
  kill "$srv" 2>/dev/null || true
  trap - EXIT
  ok "冒烟启动与 /api/system/health、/openapi.json 正常"

  unset DATABASE_URL UPLOAD_DIR ZVEC_PATH CHROMA_PERSIST_DIR LOG_DIR

  log "G1 · 变更留痕自检（AGENTS.md §1.1：本分支改动必须在 CHANGELOG.md 有条目）"
  local changed
  if git rev-parse --verify --quiet "$GATE_BASE" >/dev/null; then
    changed="$(git --no-pager diff --name-only "$GATE_BASE"...HEAD)"
  else
    changed="$(git --no-pager diff --name-only HEAD)"
  fi
  if [ -z "$changed" ]; then
    ok "相对 $GATE_BASE 无改动"
  else
    printf '%s\n' "$changed"
    printf '%s\n' "$changed" | grep -qx 'CHANGELOG.md' \
      || die "本分支改动未写入 CHANGELOG.md（对照上面的文件清单补条目，AGENTS.md §1.1）"
    ok "CHANGELOG.md 已更新（请人工确认条目覆盖上面每一项改动）"
  fi

  log "G1 · 密钥与敏感文件自检"
  if git ls-files --error-unmatch .env >/dev/null 2>&1; then
    die ".env 已被 git 跟踪，必须从版本库移除"
  fi
  ok ".env 未被跟踪"
  git check-ignore -q .env || die ".env 未被 .gitignore 忽略"
  ok ".env 已被忽略"
  if git --no-pager diff HEAD | grep -qE '(API|SECRET|TOKEN)_KEY=.{20,}|sk-[A-Za-z0-9_-]{20,}'; then
    die "diff 里疑似出现明文密钥"
  fi
  ok "diff 未发现疑似密钥"

  log "G1 通过（推送前请再跑一次 git status -sb 确认无未跟踪的敏感文件）"
}

g2() {
  log "G2 · 真实 provider 联网验收（需要 .env 里的 Key，会产生少量费用）"
  uv run pytest -m live -q
  ok "联网用例通过"

  cat <<'MANUAL'

手动部分（G2 的另一半，无法脚本化）：
  1) uv run uvicorn inner_rag.main:app --reload --port 8010
  2) curl --noproxy '*' -s localhost:8010/api/system/health
  3) 上传 docs/samples/acceptance.txt → 提问 → 检查 sources 与 done 的答案确实来自文档
  4) Phase 6 起：小库评测（见 docs/evaluation.md 的三级执行）
MANUAL
  ok "G2 通过"
}

case "$LEVEL" in
  g0) g0 ;;
  g1) g1 ;;
  g2) g2 ;;
  all) g0; g1; g2 ;;
  *) die "用法: $0 g0|g1|g2|all" ;;
esac
