# inner-rag 执行路线与测试门禁

本文档是阶段开发的**唯一入口**：既是「接下来做什么」的路线，也是「什么时候才允许上传（push）」的判定
标准。每完成一个阶段，先按第 4 节的**门禁**跑测试，再按第 6 节清单提交与推送。

## 1. 开发策略（已确认）

排序原则是**先跑通、再打磨、最后交付**：

- **API 优先**：模型侧默认走云端 HTTP API（OpenRouter / DeepSeek / 任意 OpenAI 兼容端点），一条
  `curl` 就能验证链路，迭代最快。Ollama 保留为可选离线路径，不再是唯一后端。
- **Linux 本地优先**：当前主力环境是 Linux + `uv run` + SQLite 单文件。docker 与 PostgreSQL 留到
  Phase 5 收尾（本机 docker socket 无权限，过早引入会拖慢迭代）。
- **部署能力不丢**：所有阶段都必须保持「只改环境变量即可切换 provider 与数据库」的性质——这是
  Phase 5 能顺利部署的前提，也是不能为了开发方便而写死的红线。
- **SQLite 优先，PostgreSQL 可达**：同一套 SQLAlchemy 模型与 Alembic 迁移必须对两者都成立。
- **密钥只在 `.env`**：`.env` 已被 `.gitignore` 忽略，只提交 `.env.example` 的空占位；任何 key 都不得
  出现在源码、测试、文档与 commit 里。
- **小步提交**：每阶段末尾按主题拆分 commit（含 `Co-Authored-By: Warp <agent@warp.dev>`），先跑门禁
  再 push。

## 2. 当前状态（Phase 2 结束）

已经具备：

- 分层清晰、可测试的后端：`src/inner_rag/{api,core,models,providers,schemas,services}`，`uv` 管理依赖
- **模型后端可插拔**：Chat 与 Embedding 各自独立选型（`ollama` / `openrouter` / `deepseek` /
  `openai` / `mock`），只改 `.env` 两个变量；instance 在工厂内按配置缓存复用
- 文档解析（PDF / Word / Excel / 文本）、Chroma 向量库（cosine 空间、按 KB 分 collection）、
  `similarity` / `mmr` / `hybrid` 三策略检索与**真实相关度**、SSE 流式问答、引用溯源、会话历史
- Alembic 迁移（SQLite 与 PostgreSQL 通吃）、`AUTO_CREATE_TABLES=false`、SQLite 开启 WAL 与
  `foreign_keys=ON`
- 68 个完全离线的 pytest 用例（fake / mock provider，无网络）+ 4 个 `-m live` 真实 API 用例（默认 deselect）、
  `ruff` 与 `mypy` 干净
- 前端（Vue 3 + Vite）与后端端口 8010 打通，SSE 手写解析

尚未具备（这就是后续阶段要解决的）：

1. **OCR 只有 `none` / `paddle`**：扫描件与图片型 PDF 走不通，且 `paddle` 依赖重（Phase 3）
2. **检索质量未量化**：没有评测集、没有 rerank，`hybrid` 只是启发式合并，无法回答「改进了多少」
3. **工程化欠账**：后台解析依赖进程内 `BackgroundTasks`（无重试、无进度），缓存是进程内 LRU
   （多 worker 失效），无结构化日志与 trace
4. **前端未做本地构建验证**：本机没有 Node/npm，只在静态层面修过 bug
5. **部署产物未验证**：Dockerfile / compose 已写好但从未真正构建过
6. **真实云端链路仅在 G2 手工跑过一次**：没有稳定性/成本监控，也没有多 provider 回归矩阵

## 3. 阶段总览

| 阶段 | 主题 | 关键交付 | 状态 |
| --- | --- | --- | --- |
| Phase 0 | uv 项目与 git 基线 | src 布局、`uv.lock`、依赖清理 | ✅ |
| Phase 1 | 陈旧 API 全量升级 + 必修缺陷 | LangChain 1.x、Alembic、44 个离线用例 | ✅ |
| Phase 1.5 | SQLite 优先 + 密钥外置 | SQLite 引擎加固、`.env`/`.env.example` | ✅ |
| Phase 2 | 多 Provider（API 优先） | `providers/` 层、云端 API 跑通问答 | ✅ |
| Phase 3 | OCR 迁移到视觉大模型 | VLM OCR 后端、扫描件可检索 | 待开始 |
| Phase 4 | 检索质量与工程化 | 评测集、Rerank、任务队列、结构化日志 | 待开始 |
| Phase 5 | 交付 | Docker 镜像、PostgreSQL、CI、部署文档 | 待开始 |

## 4. 测试门禁（Gate）

门禁分三级：**G0 每次提交**、**G1 阶段上传前**、**G2 阶段人工验收**。任何一级失败都不提交、不推送；
不满足的门禁项必须在提交说明里写明原因与补救计划（例如「本机无 docker，Phase 5 的镜像验证待 CI」）。

### G0 — 每次提交（秒级，改一行也要跑）

```bash
uv run ruff check .
uv run ruff format --check .
```

期望：两条命令都输出无错误（`All checks passed!` / `N files already formatted`）。

### G1 — 阶段上传前（必须全绿）

```bash
# 1) 静态检查与类型
uv run ruff check .
uv run ruff format --check .
uv run mypy

# 2) 全量离线测试（不依赖 Ollama / 数据库服务 / 网络）
uv run pytest -q

# 3) 迁移自检：干净库从零建表 → 无漂移 → 可回退 → 可重放
rm -rf /tmp/ir-gate && mkdir -p /tmp/ir-gate
export DATABASE_URL="sqlite:////tmp/ir-gate/app.db" \
       UPLOAD_DIR=/tmp/ir-gate/uploads \
       CHROMA_PERSIST_DIR=/tmp/ir-gate/chroma \
       LOG_DIR=/tmp/ir-gate/logs
uv run alembic upgrade head
uv run alembic check
uv run alembic downgrade base
uv run alembic upgrade head

# 4) 冒烟启动 + 探活（临时端口 8011，不污染 ./data）
uv run uvicorn inner_rag.main:app --port 8011 &
SRV=$!
sleep 5
curl --noproxy '*' -sf localhost:8011/api/system/health
curl --noproxy '*' -sf -o /dev/null localhost:8011/openapi.json && echo "openapi ok"
kill $SRV

# 5) 密钥与敏感文件自检（只应有 ok 行）
git ls-files | grep -E '(^|/)\.env$' && echo "ERROR: .env 被跟踪" || echo "ok: .env 未被跟踪"
# 注意：.env 被 .gitignore 忽略，因此 `--others --exclude-standard` 不会列出它，要用 check-ignore
git check-ignore -q .env && echo "ok: .env 已被 gitignore 忽略" || echo "WARN: .env 未被忽略"
# 启发式：只统计“像真 key 的值”的条数，不回显匹配内容；文档里的 KEY=... 占位不算
FOUND=$(git --no-pager diff HEAD | grep -cE '(API|SECRET|TOKEN)_KEY=[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9_-]{20,}')
[ "$FOUND" = "0" ] && echo "ok: diff 无密钥" || echo "ERROR: diff 里疑似密钥 $FOUND 处"
```

期望：
- `mypy` → `Success: no issues found`
- `pytest` → 全绿（用例数只增不减；任何 skip 都要在提交说明里解释）
- 迁移四步全部成功，`alembic check` 输出无 drift
- `/api/system/health` 返回 JSON（模型侧未启动时 `status: degraded` 属正常），`openapi.json` 可访问
- 密钥自检没有 `ERROR` 行（该检查是启发式的，命中时人工确认是否为真密钥）

### G2 — 阶段人工验收（真实链路，每阶段至少一次）

门禁只证明「代码没坏」，不证明「功能好用」。每个阶段收尾时用真实 provider 跑一次端到端，并把结果
（`curl` 输出片段 / 前端截图 / `/api/system/stats` 的延迟与命中率）写进提交说明或 `docs/` 记录：

```bash
# 需要 .env 中已填好 OPENROUTER_API_KEY（或 DEEPSEEK_API_KEY）
uv run pytest -m live -q          # 真实 API 冒烟：chat / 流式 / embedding 维度 / 端到端验收
uv run uvicorn inner_rag.main:app --reload --port 8010

curl --noproxy '*' -s localhost:8010/api/system/health
KB=$(curl --noproxy '*' -s -X POST localhost:8010/api/kb \
  -H 'Content-Type: application/json' -d '{"name":"验收库"}' | python -c 'import sys,json;print(json.load(sys.stdin)["data"]["id"])')
curl --noproxy '*' -s -X POST localhost:8010/api/doc/upload -F kb_id=$KB -F 'files=@docs/samples/acceptance.txt'
curl --noproxy '*' -s "localhost:8010/api/doc?kb_id=$KB"          # status 应为 completed
curl --noproxy '*' -N -X POST localhost:8010/api/chat/stream \
  -H 'Content-Type: application/json' -d "{\"kb_id\":$KB,\"question\":\"验收用的测试句子是什么？\"}"
```

验收通过的最低标准：`sources` 指向刚上传的文件且相关度合理，`done` 的答案确实来自文档内容（而不是
模型自由发挥）。若真实 API 不可用（无 key / 余额耗尽），可以退化为 `LLM_PROVIDER=mock` 演示链路，
但必须在提交说明里写明「G2 未完成，原因：…」。

## 5. 分阶段计划

每个阶段固定包含：目标 → 主要改动 → 阶段专属测试 → 完成定义（DoD）。门禁沿用 G1，另有专项要求。

### Phase 2 — 多 Provider 抽象（API 优先）✅

目标：把模型后端从「写死 Ollama」变成配置驱动的可插拔工厂，云端 API 成为默认路径。

主要改动：
- 新增 `src/inner_rag/providers/`：`chat.py`、`embeddings.py`、`factory.py`
- Chat provider：`ollama` / `openrouter` / `deepseek` / `openai`（兼容任意 OpenAI 风格端点）/ `mock`
- Embedding provider：`ollama` / `openrouter` / `openai` / `mock`
  （注意：**DeepSeek 官方没有 embeddings 接口**，嵌入必须走 Ollama 或 OpenAI 兼容端点，README 与
  `.env.example` 都要写清）
- 配置项：`LLM_PROVIDER`、`EMBEDDING_PROVIDER`、各 provider 的 `*_BASE_URL` / `*_MODEL` / `*_API_KEY`；
  旧 `OLLAMA_*` 保持兼容，缺失 key 时启动即给出「请在 .env 填写 X」的明确报错
- `Settings.embedding_key` 改为 `provider:model`（例如 `openrouter:qwen/qwen3-embedding-8b`），复用
  Phase 1 已落地的知识库一致性校验
- API：`/api/system/providers`（逐 provider 状态）、`/api/system/health` 改为不写死 Ollama、
  `/api/system/models` 按 provider 返回

阶段专属测试（除 G1 外必须新增并通过）：
- **工厂单测**（无网络）：未知 provider 名、缺 key、别名与大小写、Chat 与 Embedding 解耦组合
- **参数映射单测**：monkeypatch 断言工厂构造出的对象确实带上配置的 `base_url` / `model`（不发请求）
- **mock provider 契约测试**：`/api/chat/stream`、`/api/chat/send`、上传入库全链路跑在 mock 上
- **降级测试**：缺 key 时 API 返回可读错误（不是 500 堆栈）
- **live 冒烟（手工、默认跳过）**：新增 `pytest -m live`（`addopts = "-m 'not live'"`），需要 `.env`
  里的真实 key；跑之前确认 embedding 缓存命中（同一文本不重复调用），控制费用

DoD：`.env` 里填 `LLM_PROVIDER=openrouter` + `OPENROUTER_API_KEY` 后，`/api/system/health` 全绿，
上传 → 提问 → 引用来源全链路走通；README 更新 provider 配置表与「DeepSeek 无 embeddings」说明。

落地结果（实际交付相对计划的差异）：
- 除计划中的 `chat.py` / `embeddings.py` / `factory.py` 外，多出一个 `specs.py`：provider 元数据
  （base_url、模型、key 环境变量名、文档链接、备注）集中一处，校验错误能直接引用环境变量名
- 默认向量模型选 `openrouter` 的 `liquid/lfm-2.5-embedding-350m:free`（免费但只有 **512 token**
  上下文，因此新增 `EMBEDDING_MAX_INPUT_CHARS` 显式截断开关）
- `/health` 不因配置错误返回 5xx，而是 `status: degraded` + `llm.error` / `embedding.error`；
  provider 配置错误在 API 层统一映射为 **503**，且文案指向该改的变量
- `LangChain` 侧统一改用 `langchain-openai` 接管全部云端 OpenAI 兼容端点，Ollama 仍用
  `langchain-ollama`；不再手写 `httpx` 探活（`/models` 发现逻辑收进 `factory._discover_models`）

### Phase 3 — OCR 迁移到视觉大模型（VLM）

目标：扫描件与图片型 PDF 能进索引，且不引入本地重依赖。

主要改动：
- `services/ocr.py` 增加 `vlm` 后端（OpenRouter / 任意 OpenAI 兼容 chat completions 传 base64 图片）
- 扫描版 PDF：PyMuPDF 渲染页面为图 → 降采样 → VLM 识别（替代从未真正生效的旧 `fitz` 兜底）
- `OCR_BACKEND=none|paddle|vlm`；送图前控制分辨率以约束 token 成本；失败时文档标记 `failed` 并给出原因

阶段专属测试：
- 用 `tests/fixtures/` 里的小图（几十 KB 的合成图片）跑 VLM 后端的**假响应**单测（monkeypatch HTTP）
- `none` 后端对图片显式报错的回归测试（Phase 1 已有，需保持）
- 真实 VLM live 冒烟：一张含已知文字的图片，断言识别结果包含该文字（`-m live`）

DoD：上传一张扫描件图片，文档变成 `completed` 且能检索到图中文字。

### Phase 4 — 检索质量与工程化

目标：让「检索好不好」可量化，并补上工程化欠账。

主要改动：
- 评测：`scripts/eval_retrieval.py` + 一个小型标注集（问题 → 应命中的分块/文件），输出召回率、
  MRR、引用准确率与平均延迟，结果落 `docs/` 便于对比
- Rerank：cross-encoder 或 provider 侧 rerank，配置开关，评测集上要有可复现的提升
- 工程化：`BackgroundTasks` → 独立任务队列（先做可插拔抽象 + 本地 worker）、进程内 LRU → Redis
  （无 Redis 时自动降级为内存实现）、结构化日志与请求 trace

阶段专属测试：
- 评测脚本的**离线回归用例**：用假 embedding 构造已知答案，断言指标计算正确（防止「评测脚本自己错」）
- 任务队列：入队 / 重试 / 失败标记的状态机单测；无 Redis 时的降级单测
- 缓存：命中、TTL、按 kb_id 失效、多进程语义（至少在文档中说明边界）

DoD：`scripts/eval_retrieval.py` 能在固定数据集上跑出稳定指标，且 README 给出改动前后的对比数字。

### Phase 5 — 交付（Docker + PostgreSQL + CI）

目标：换一台干净机器，按 README 能在 10 分钟内跑起来。

主要改动：
- 用真实环境验证 `Dockerfile` 与 `docker compose --profile app up -d --build`
- PostgreSQL 路径实测：`DATABASE_URL` 切 PostgreSQL 后 `alembic upgrade head` 与全链路回归
- CI（GitHub Actions）：`ruff` + `mypy` + `pytest` + 镜像构建
- 部署文档：环境变量清单、数据卷与备份、反向代理与前端静态托管

阶段专属测试：
- `docker build` 成功 + 容器内 `/api/system/health` 通过（compose 自带 healthcheck）
- PostgreSQL 下的 G1 全流程（迁移 upgrade/downgrade/check + pytest 指向 PG 跑一遍）
- CI 在 PR 上全绿

DoD：新机器按 README 从零跑通，且 CI 绿。

## 6. 上传前清单（每个阶段照做）

- [ ] G0 通过；G1 五步全绿（把关键输出贴进提交说明）
- [ ] G2 至少用真实 provider 验证一次；无法验证则写明原因
- [ ] 文档同步：README / `docs/` / `.env.example`（新增配置项必须进模板）
- [ ] 模型或表结构有变动 → 必须有 Alembic revision，且 `alembic check` 无漂移
- [ ] 密钥自检通过：`.env` 未被跟踪，diff 里无 key
- [ ] 提交按主题拆分，信息说清「改了什么 / 为什么 / 怎么验证的」，附
      `Co-Authored-By: Warp <agent@warp.dev>`
- [ ] push 后 `git status -sb` 显示与 `origin/main` 同步

## 7. 风险与缓解

| 风险 | 影响 | 缓解 |
| --- | --- | --- |
| 云端 API 费用/限流 | 迭代中断 | 默认走缓存与 mock provider；live 测试默认跳过并人工触发；先用便宜/免费档模型 |
| 密钥泄漏（进仓库或日志） | 账号被盗刷 | 只从 `.env` 读；门禁含密钥扫描；日志脱敏；泄漏后立即吊销 |
| SQLite 并发写入 | `database is locked` | 已开 WAL + `busy_timeout`；上传并发受限；真实并发场景按门禁切 PostgreSQL |
| 本机无 Node/npm | 前端无法构建验证 | Phase 2 起把前端改动限制在最小范围；构建验证放到有 Node 的环境或 CI |
| 本机无 docker socket 权限 | 无法本地验证镜像 | Phase 5 前不阻塞主流程；镜像验证交给 CI 或具备权限的机器 |
| 测试全用假 provider | 真实 API 行为差异漏测 | 每阶段一次 G2 + `-m live` 用例覆盖真实调用 |
| 免费/小上下文 embedding 模型 | 输入超上下文（如 512 token）、上游可能留存数据训练 | 用 `EMBEDDING_MAX_INPUT_CHARS` 显式截断（可配合调小 `CHUNK_SIZE`）；有合规要求时换付费模型；换 embedding 模型后必须重建索引 |
| provider 配置写错（错名 / 缺 key / DeepSeek 当 embedding） | 问答骤报 503 | `/api/system/providers` 与 `/health` 的 `llm.error` / `embedding.error` 直接给出变量名与可选值；`.env.example` 与 README 同步说明 |
| 迁移漂移（模型改了没生成 revision） | 部署时炸 | G1 固定跑 `alembic check`，新增字段必须带 revision |

## 8. 进度记录

- **Phase 0** ✅ uv 项目骨架、依赖锁定、源码迁到 `src/inner_rag/`
- **Phase 1** ✅ LangChain 1.x 全量升级 + 必修缺陷修复；Alembic、Docker 资产、44 个离线用例
- **Phase 1.5** ✅ 开发默认 SQLite（WAL + 外键 + 等锁超时）、密钥外置到 `.env`、本路线文档
- **Phase 2** ✅ `providers/` 抽象层（`specs` / `chat` / `embeddings` / `factory`）：Chat 与 Embedding
  独立选型（ollama / openrouter / deepseek / openai / mock）、`ProviderError` → 503、`/api/system/providers`、
  `embedding_key` 改为 `provider:model`、模型实例缓存；68 个离线用例 + 4 个 `-m live` 用例；
  G1 五步全绿（ruff / mypy / pytest / 迁移四步 / 冒烟启动 + 密钥扫描）
- **Phase 3** 待开始（OCR 迁移到视觉大模型）

## 9. 已修复的关键缺陷（工程记录）

这些是 Phase 1 修复的行为变更，README 不再展开（README 只讲系统能力），细节留在这里备查：

| 问题 | 原行为 | 现行为 |
| --- | --- | --- |
| 相关度算错 | `similarity_search_with_relevance_scores` 返回的其实是原始距离；MMR 项被虚构为 0.8/1.0 | 统一用 `similarity_search_with_score` + `1 - 距离` 换算并 clamp 到 `[0,1]`；MMR 项如实返回 `null` |
| 会话历史取错 | 取**最早** 20 条消息 | 取**最近** `HISTORY_MAX_MESSAGES` 条 |
| 后台任务用坏 Session | 把请求级 `Session` 传进 `BackgroundTasks`（响应返回时已关闭） | 后台任务自行创建/关闭 Session |
| 上传无大小前置校验 | 先整体读进内存再判断大小 | 流式落盘 + 边写边校验，超限立即中断并清理 |
| 文件名未净化 | 直接用上传的文件名拼路径（可路径穿越） | 只取 basename + 随机前缀，落地在该知识库目录内 |
| 检索缓存粒度粗 / 不失效 | 按 query 全局缓存，入库/删除后仍返回旧结果 | key 带 `kb_id`，入库、删除、重建后精确失效该知识库 |
| 静默吞异常 | 检索异常被吞掉后返回空结果 | 异常向上抛出，由 API 层给出明确错误 |
| 解析失败无提示 | 扫描件/图片写入占位文本污染索引 | 显式抛错，提示启用 OCR |
| CORS 不安全 | `allow_origins=["*"]` + `allow_credentials=True`（浏览器实际会拒绝） | 白名单化，由 `CORS_ORIGINS` 配置 |
| embedding 换模型无校验 | 静默检索到不同向量空间的向量 | 知识库记录 embedding 标识，写入/检索时校验并给出重建指引 |
| 前端代理失效 | `baseURL` 写成 `' http://localhost:8000/api'`（含前导空格且绕过 Vite 代理） | 改为相对路径 `/api`，代理目标统一为后端 `8010` |

另外两项工程决策保留在代码与 `.env.example` 里：LangChain 1.x 已移除的旧 API
（`langchain.text_splitter`、`ChatOllama(streaming=...)` 等）全部换成新写法；建表从运行时
`create_all` 改为 Alembic 迁移（`AUTO_CREATE_TABLES` 仅留给测试/一次性库）。
