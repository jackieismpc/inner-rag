# 🤖 inner-rag

> 🏗️ **多 Provider · 可插拔 · 好部署** 的企业内部知识库问答系统
>
> `FastAPI` + `LangChain 1.x` + `zvec` + `SQLite / PostgreSQL`
>
> 🔐 内置登录与知识库级 ACL：账号由运维脚本发放，成员按知识库授权 `只读 / 可写`

---

## 🌱 背景

企业内部的资料大多散在共享盘、邮件附件和各人的电脑里：产品手册是 PDF，规范是 Word，参数表是 Excel，还有一堆扫描件。想查一句话往往要翻半天，关键词搜索又答不了「这个参数是什么意思」这类问题。把资料直接丢给公网 SaaS 又过不了合规——内部文档不能出内网。

自己搭一套 RAG 问答，真正的门槛不在「调用一个模型」，而在工程侧的 **四件事**：

| # | 挑战 | 说明 |
| --- | --- | --- |
| 1 | 🔄 **模型会换** | 今天用云端便宜模型，明天换本地开源模型，或中间隔一层公司网关；模型调用散落在业务代码里，每换一次就是一次重构 |
| 2 | 🗄️ **向量库会换** | 嵌入式向量库省运维，但要横向扩容时得换远程服务；而多数 RAG 代码把向量库 API 直接写进了检索逻辑 |
| 3 | 🔐 **权限必须是真的** | 企业知识库天然不是「所有人都能看所有库」，必须落到「谁能看哪个库、谁只能看不能改」，且每次请求实时判定 |
| 4 | 📊 **改完要能证明变好了** | 调分块参数、加精排、改阈值，凭感觉只会越调越玄学，需要可复现的评测集与指标，把「改动前后」变成可对照的表 |

> `inner-rag` 就是围绕这四件事做的：**把会变的部分全部收敛成插件点，把「好不好」变成可量化的数字。**

---

## ✨ Introduction

`inner-rag` 把企业里散落的文档（PDF / Word / Excel / 纯文本，扫描件与图片走 OCR）解析、分块、向量化入库，再基于「向量检索 + 引用溯源 + 流式问答」回答问题。

Chat 模型与 Embedding 模型是两个**互相独立的可插拔后端**：同一条链路既能跑本地推理服务（vLLM / SGLang / Ollama，全离线），也能直接接 OpenRouter / DeepSeek / 任意 OpenAI 兼容网关，改两个环境变量即可切换，业务代码、接口与数据库都不用动。

### 🔌 七个插件点，同一套契约

Chat provider、Embedding provider、向量库、缓存、任务队列、精排（rerank）、查询改写（query rewrite）——每个插件点都有 **「接口 + 内置实现 + 配置项 + 探活 + 契约测试」五件套**，统一由 `plugins/registry.py` 的 `Registry[T]` 登记，第三方包可以用 `importlib.metadata` 的 entry point 注册实现而**不改本项目源码**；`GET /api/system/plugins` 把「有哪些实现、当前用哪个」变成可查询的运行时数据。

> 内置向量库有 `zvec`（默认）/ `chroma` / `memory` 三种实现，跑的是同一套契约测试——这是「换后端不改业务代码」这句话的**实证**，而不是口号。

### 🔍 检索链路（Phase 8 起）

```
查询改写（默认关闭）
   → 逐条查询做「向量召回 ∪ 词面 BM25 召回」
   → 按分块融合打分
   → 阈值过滤
   → 精排（默认关闭）
   → 组装 Prompt
   → LLM 流式输出
```

hybrid 的融合规则是 `max(向量相关度, 权重 × 归一化 BM25)`，阈值在融合**之后**统一生效；词面这一路带启用门槛（`HYBRID_MIN_SPARSE_SCORE`），避免一两次巧合的字面匹配摧毁「答不出来就拒答」的能力。

### 🧩 其余值得一提

- 🔐 **身份与访问控制**：本地账号 + JWT + argon2id 口令哈希，知识库级 ACL 分 `read` / `write` / `owner` 三级，权限每次请求实时判定；
- 📡 **可观测性**：「一个 `request_id` 贯穿响应头、日志与 trace」，检索 / 问答 / 入库全链路 span 计时，可选上报 LangSmith（默认关闭，零网络零费用）；
- ⚡ **成本与性能**：嵌入缓存（按 `provider:model` 隔离）+ 检索缓存（按库精确失效）、批量嵌入真并发 + 信号量限流、有界并发的后台入库队列。

### 🧰 技术选型

| 层次 | 选型 |
| --- | --- |
| 🐍 语言 / 包管理 | Python 3.13（uv 管理）、`uv.lock` 锁定依赖 |
| 🌐 Web 框架 | FastAPI 0.141+、Uvicorn 0.54+、SSE 流式响应 |
| 🤖 LLM 编排 | LangChain 1.x + `langchain-ollama` / `langchain-openai`（云端 + 本地 vLLM/SGLang）/ `langchain-deepseek` |
| 🗄️ 向量库 | **zvec 0.7.0**（[Alibaba 开源](https://github.com/alibaba/zvec)，HNSW + cosine，默认）；ChromaDB 1.5+ 与 `memory`（零依赖进程内）为兼容实现 |
| 🗃️ 关系库 | SQLite（开发默认）+ PostgreSQL 16（部署可选）+ SQLAlchemy 2.1 + Alembic 1.20 |
| 🔑 认证与权限 | JWT（PyJWT，HS256）+ argon2id（argon2-cffi）+ 知识库级 ACL |
| 📄 文档解析 | pypdf、PyMuPDF、python-docx、docx2txt、openpyxl、xlrd、Pillow、chardet |
| 🎨 前端 | Vue 3 + Vite + Pinia + Tailwind CSS 3 |
| ✅ 质量 | ruff、pytest（+ pytest-asyncio）、mypy |

> 💡 选用 **zvec** 的理由是「零运维」：进程内嵌入、无需独立服务、WAL 持久化，与 SQLite 单文件开发模型一致。代价是写锁按 collection 目录独占，因此**必须单进程部署**（不要 `uvicorn --workers`）。

### 📚 文档导航

| 文档 | 内容 |
| --- | --- |
| [`docs/architecture.md`](docs/architecture.md) | 架构分层、插件点契约、错误与降级策略、怎么加新后端 |
| [`docs/usage.md`](docs/usage.md) | 使用指南与 API 一览 |
| [`docs/configuration.md`](docs/configuration.md) | 配置项逐项说明 |
| [`docs/operations.md`](docs/operations.md) | 运维脚本、开发与部署 |
| [`docs/observability.md`](docs/observability.md) | 可观测性 |
| [`docs/testing.md`](docs/testing.md) | 测试与门禁策略 |
| [`docs/DEVELOPMENT_PLAN.md`](docs/DEVELOPMENT_PLAN.md) | 阶段计划 |

---

## 🚀 Quick Start

### 📋 前置条件

Linux / macOS（Windows 建议 WSL2）+ [uv](https://docs.astral.sh/uv/)（会自行安装 Python 3.13）。

**模型三选一**：
- ☁️ 云端 API（需 OpenRouter / DeepSeek 的 key）
- 💻 本地推理服务（vLLM / SGLang 跑 Qwen2.5-32B，默认，见 `docs/operations.md` 3.5）
- 🧪 `mock`（零依赖，仅演示与验收链路）

数据库开发默认 SQLite 单文件，**无需任何安装**。

### 🛠️ 五步启动

```bash
# 1️⃣ 安装依赖
git clone https://github.com/jackieismpc/inner-rag.git
cd inner-rag
uv sync                       # 创建 .venv 并按 uv.lock 安装依赖（含 Python 3.13）
# uv sync --extra ocr-paddle  # 需要本地 OCR（PaddleOCR）时追加

# 2️⃣ 配置环境变量与密钥
cp .env.example .env          # .env 已被 .gitignore 忽略，不会进仓库
# 通常只需要改：模型 provider 与 key、PORT、CORS_ORIGINS

# 3️⃣ 建表并启动后端（表结构由 Alembic 管理，首次启动前必须先迁移）
uv run alembic upgrade head
uv run uvicorn inner_rag.main:app --reload --port 8010
# 或者一步到位（自动 uv sync + 迁移 + 热重载）：
./scripts/start.sh

# 4️⃣ 创建第一个账号（系统不提供注册接口，账号由管理员发放）
uv run scripts/create_user.py admin          # 密码交互式输入、不回显

# 5️⃣ 启动前端
cd frontend && npm install && npm run dev    # http://localhost:3000
```

### ⚙️ 模型侧最少配置

模型侧最少只需要在 `.env` 里指定两个独立开关。默认向量侧走**本机部署的开源权重**（不发外部请求；原因是云端网关的免费额度按请求数限流，一次全量建库就会超）：

```bash
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=<your-key>
DEEPSEEK_CHAT_MODEL=deepseek-flash

EMBEDDING_PROVIDER=sentence_transformers          # 默认值
SENTENCE_TRANSFORMERS_MODEL=Qwen/Qwen3-Embedding-0.6B
# SENTENCE_TRANSFORMERS_DEVICE=auto               # 无 GPU 时自动退回 CPU
# HF_ENDPOINT=https://hf-mirror.com               # 出网受限时换镜像
```

> 📌 本地嵌入需要可选依赖：`uv sync --extra local-embed`（会拉 `torch`，首次运行还要下载权重）。
> 不想装就用云端或 Ollama：`EMBEDDING_PROVIDER=openrouter` + `OPENROUTER_API_KEY`，或 `EMBEDDING_PROVIDER=ollama` + `ollama pull qwen3-embedding:8b`；换 embedding = 换向量空间，必须重建索引（`uv run scripts/reindex_kb.py <kb_id>`）。

### 🩺 健康检查

打开 <http://localhost:8010/docs> 看交互式 API 文档，`/api/system/health` 检查后端与模型连通性：

```bash
curl -s http://localhost:8010/api/system/health
# {"status":"healthy","version":"0.3.0",
#  "llm":{"provider":"deepseek","model":"deepseek-flash","ok":true,...},
#  "embedding":{"provider":"sentence_transformers","model":"Qwen/Qwen3-Embedding-0.6B","ok":true,...}}
```

> ⚠️ `status: degraded` 说明模型侧有问题，看 `llm.error` / `embedding.error`——文案会直接指出该去 `.env` 改哪个变量（例如漏填 API Key、忘了 `uv sync --extra local-embed`）或哪个服务连不上；此时检索与问答会失败，但知识库、文档等管理接口仍可用。

> 🧪 想完全离线：`LLM_PROVIDER=mock` + `EMBEDDING_PROVIDER=mock`，不需要任何模型服务，也能跑通「上传 → 检索 → 带引用回答」的完整链路。其余三步（建库、上传、提问）的 curl 示例与全部接口清单见 [`docs/usage.md`](docs/usage.md)。

---

## 📊 Benchmark

改检索、改分块、改 Prompt 之后必须能回答一个问题：**到底变好了没有**。`benchmark/` 下的脚本把评测集（`docs/datasets/dragon_king/` 下的真实语料评测题）跑一遍，输出 RAG 指标与延迟，并把结果写成下面这段表格的一行；指标定义、门禁阈值与判读方式见 [`docs/evaluation.md`](docs/evaluation.md)。

```bash
# 🧪 离线自检：mock provider + 短片段 fixture，不联网、不花钱（提交前跑这个）
uv run --extra local-embed python -m benchmark.run_bench --mode fixtures

# 📈 真实知识库：检索指标 + 延迟，写入下面这张表
uv run --extra local-embed python -m benchmark.run_bench --mode kb --kb-id 3 --update-readme

# 💬 再加回答指标（要点命中率 / 引用精度 / 拒答正确率，会调用 LLM 产生费用）
uv run --extra local-embed python -m benchmark.run_bench --mode kb --kb-id 3 --answer --update-readme

# 🏗️ 建评测库（从 data/uploads/龙族.pdf 按页窗口构建，产出 manifest）
# 被打断就直接重跑：默认复用同名知识库续跑，已入库的分块会跳过
uv run --extra local-embed python -u scripts/build_eval_kb.py --profile small --name dragon_king_small --owner admin
uv run --extra local-embed python -u scripts/build_eval_kb.py --profile full  --name dragon_king_full  --owner admin
# 评测集覆盖全书，所以**主场地是全库**；小库用于快速回归与拒答验证（见 docs/evaluation.md 2.1）

# 🤖 回答侧评测：judge 正确性 / 忠实度 / token / 失败归因 → docs/reports/eval-<日期>-<label>.md
uv run --extra local-embed python scripts/eval_answer.py --from-result benchmark/results/<上面的结果 json>
```

- 🧪 `fixtures` 模式只验证脚本与指标算法（mock embedding 没有语义能力），**不写表也不落盘**，避免把自检数字当成成绩；
- 📈 `kb` 模式用真实知识库并绕过 QueryCache 直接查向量库，召回与延迟都是真值，`--answer` 按题计费；
- 📚 评测集默认取 `docs/datasets/dragon_king/eval_v3.jsonl`（146 条：28 条人工题 + 118 条从语料自动生成，锚点覆盖全书），用 `--dataset` 可切回 v1/v2 做同口径对比；

<!-- BEGIN BENCHMARK -->
| 日期 | 配置 | 题数 | Recall@k | MRR | 页命中率 | 要点命中率 | 引用精度 | 拒答正确率 | 检索 p50 | 检索 p95 | 端到端 p50 | 结果文件 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-09-30 | kb1/openrouter:liquid/lfm-2.5-embedding-350m:free/hybrid/k=8+answer | 9 | 75.0% | 0.688 | 62.5% | 75.0% | 27.5% | 100.0% | 1238.6 ms | 2176.1 ms | 1442.6 ms | `benchmark/results/2026-09-30-kb-kb1-openrouter-liquid-lfm-2-5-embedding-350m-free-hybrid-k-8-answer.json` |
| 2026-09-30 | kb1/openrouter:liquid/lfm-2.5-embedding-350m:free/hybrid/w=0.6/k=8+answer | 9 | 100.0% | 0.745 | 87.5% | 100.0% | 20.0% | 100.0% | 1259.0 ms | 1637.7 ms | 1363.2 ms | `benchmark/results/2026-09-30-kb-kb1-openrouter-liquid-lfm-2-5-embedding-350m-free-hybrid-w-0-6-k-8-answer.json` |
| 2026-09-30 | full/qwen3-0.6b-local/hybrid/k=8 | 146 | 79.3% | 0.565 | 78.5% | — | — | — | 123.8 ms | 186.5 ms | — | `benchmark/results/2026-09-30-kb-full-qwen3-0-6b-local-hybrid-k-8.json` |
<!-- END BENCHMARK -->

> ⚙️ 表格由 `--update-readme` 写入，**不要手工编辑标记之间的区域**；kb 模式每次还在 `benchmark/results/` 留一份含逐题明细的 JSON，便于回溯。指标口径与注意事项见 [`benchmark/README.md`](benchmark/README.md)。

### 📖 读表注意两件事

1. ⚠️ **`引用精度` 不可跨配置比较**。它的分母是「本题引用了多少条来源」，而召回变好会让原本一条都没引用的题（记 0.0）变成引满 5 条（记 1/5），分母随配置漂移。同期可比口径「引用的来源里至少一条命中期望页」的题占比是 **75.0% → 87.5%**，该口径已落成 `metrics.spans_citation_hit`（逐题结果里的 `citation_hit`）。

2. 🔀 **`hybrid` 起词面融合（Phase 8.1）**：`w=0` 那一行是关闭词面的纯向量配置，用于对照；词面这一路有启用门槛，细节与标定数据见 [`docs/evaluation.md`](docs/evaluation.md)。

> 📄 回答侧的 judge 正确性 / 忠实度 / 失败归因在 `docs/reports/eval-*.md`，建库的页窗口与配置快照在 `docs/reports/eval-kb-*.json`。**引用里的页码是源 PDF 的物理页号**（不是子 PDF 的局部页号），建库脚本不做任何重编号——这一点踩过坑：重编号会让引用翻不到原文、评测锚点全部对不上。

---

## 📜 License

MIT
