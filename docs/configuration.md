# 配置说明

全部配置项见 [`.env.example`](../.env.example)（按 应用 / 认证 / 服务端 / 数据库 / 密钥 / 模型 / 向量库 /
上传 / OCR / 检索 / 缓存 / CORS / 日志 分组，每项都有注释）；本地实际生效的值写在 `.env`
（已被 gitignore，不入库）。本文只列出容易踩坑的关键项。

## 1. 关键配置项

| 配置 | 默认值 | 说明 |
| --- | --- | --- |
| `PORT` | `8010` | 后端端口（`8000` 在共享机器上常被占用），需与 `frontend/vite.config.js` 代理一致 |
| `DATABASE_URL` | `sqlite:///./data/inner_rag.db` | 开发默认 SQLite；部署切 `postgresql+psycopg://...` 后重跑 `alembic upgrade head` |
| `LLM_PROVIDER` | `ollama` | 对话侧 provider；向量侧是**另一个独立开关**（向量侧没有 `deepseek`，官方无 embeddings 接口） |
| `EMBEDDING_PROVIDER` | `sentence_transformers` | 向量侧 provider，默认走**本机部署的开源权重**（见 §3.1）；可选 `ollama` / `openrouter` / `openai` / `mock` |
| `SENTENCE_TRANSFORMERS_MODEL` | `Qwen/Qwen3-Embedding-0.6B` | 本地嵌入的模型（HuggingFace id 或本地目录）；**换了就必须重建索引** |
| `SENTENCE_TRANSFORMERS_DEVICE` | `auto` | `auto` / `cuda` / `cpu`；`auto` 在无 GPU 的机器上自动退回 CPU |
| `SENTENCE_TRANSFORMERS_BATCH_SIZE` | `32` | 单次喂给模型的条数；显存吃得下可以调大 |
| `SENTENCE_TRANSFORMERS_NORMALIZE` | `true` | 输出归一化；cosine 度量下**不要关** |
| `SENTENCE_TRANSFORMERS_ALLOW_DOWNLOAD` | `true` | 允许从 hub 拉权重；离线部署改 `false`，只用本地已缓存的权重 |
| `HF_ENDPOINT` | 空 | HuggingFace 端点（如 `https://hf-mirror.com`）；出网受限的环境用它换镜像，空 = 官方站 |
| `OPENROUTER_API_KEY` / `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` | 空 | 云端 API 密钥，只写在 `.env`，不要提交 |
| `EMBEDDING_MAX_INPUT_CHARS` | `0` | 单条输入的字符上限（0 = 不截断）；小上下文模型建议设 `400` |
| `EMBEDDING_TIMEOUT` | `60.0` | 单次 HTTP embedding 请求超时（秒），**只对云端 / Ollama 生效**。云端**必须显式设置**：OpenAI SDK 默认 600s，一旦某批挂起会让整个建库看起来「静默卡死」；本地权重走进程内推理，不受它影响 |
| `EMBED_MAX_ATTEMPTS` / `EMBED_RETRY_BACKOFF` | `3` / `2.0` | 单批嵌入的重试次数与退避基数（秒）。本地权重几乎不会失败，但保留这层退避不影响它 |
| `EMBED_BATCH_SIZE` / `EMBEDDING_CONCURRENCY` | `20` / `3` | 每批发给 embedding 的条数与在途批次数；入库是「逐批落库」，中断最多丢在途的那几批 |
| `VECTOR_STORE` | `zvec` | 向量库后端：`zvec`（嵌入式，默认）/ `chroma`（兼容旧数据）/ `memory`（零依赖进程内，重启即丢）；切换后用 `scripts/reindex_kb.py <kb_id>` 重建 |
| `ZVEC_PATH` | `./data/zvec_db` | zvec 数据目录，每知识库一个 `kb_<id>/` collection；写锁目录独占，**单进程部署** |
| `CACHE_BACKEND` | `memory` | 缓存后端：`memory`（每 namespace 一份有界 LRU）；换 Redis 只需加实现 + 注册 |
| `QUERY_CACHE_MAX_SIZE` / `QUERY_CACHE_TTL` | `500` / `300` | 检索缓存容量与有效期（秒，0 = 不过期）；入库后仍会按 `kb_id` 精确失效 |
| `EMBEDDING_CACHE_MAX_SIZE` | `2000` | 嵌入缓存容量；按 `provider:model` 隔离，跨模型不会命中 |
| `TASK_QUEUE_BACKEND` | `inprocess` | 后台任务队列：`inprocess`（协程池）/ `inline`（同步执行，测试用） |
| `TASK_QUEUE_CONCURRENCY` | `2` | 后台并发上限；免费 embedding 路由限流严，别调大 |
| `TASK_QUEUE_MAX_RETRIES` / `TASK_QUEUE_RETRY_BACKOFF` | `1` / `5.0` | 失败重试次数与退避系数（`backoff × attempt` 秒） |
| `TASK_QUEUE_HISTORY` | `200` | 任务历史环形缓冲大小（`/api/system/stats` 展示） |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | 分块参数，影响召回粒度；改动后建议重建索引并跑基准 |
| `RETRIEVAL_SCORE_THRESHOLD` | `0.3` | 相关度阈值；Phase 8.1 起作用在**融合后**的分数上（`max(向量相关度, 权重 × 归一化 BM25)`） |
| `HYBRID_SPARSE_WEIGHT` | `0.6` | hybrid 策略里词面检索（BM25）的权重，`0` = 退回纯向量 + MMR；定标见 `evaluation.md` 4.5 |
| `HYBRID_MIN_SPARSE_SCORE` | `20.0` | 词面这一路的启用门槛（原始 BM25）；低于它且稠密侧无过阈值候选时不启用，避免巧合匹配破坏拒答 |
| `RERANK_BACKEND` | `none` | 精排后端：`none`（默认关闭）/ `lexical`（免模型，按候选集内 IDF 覆盖率重排）/ `llm`（listwise）；契约见 `architecture.md` 3.9 |
| `RERANK_LLM_TOP_N` / `RERANK_LLM_SNIPPET_CHARS` | `12` / `500` | 仅 `llm` 精排生效：参加排序的前 N 条、每条截断的字符数（控 token） |
| `QUERY_REWRITE_BACKEND` | `none` | 查询改写后端：`none`（默认关闭）/ `alias`（查别名表）/ `keywords`（剥疑问框架词）/ `llm`；契约见 `architecture.md` 3.10 |
| `QUERY_REWRITE_MAX_QUERIES` | `3` | 一次提问最多展开出的查询条数（含原查询）；每条都会做一次召回再融合 |
| `QUERY_ALIASES` | 空 | `alias` 改写的别名表：`别名=正式名`，逗号分隔（如 `诺诺=陈墨瞳,凯撒=凯撒·加图索`） |
| `AUTO_CREATE_TABLES` | `false` | 表结构交给 Alembic；仅测试/一次性库设为 `true` |
| `AUTH_SECRET_KEY` | `dev-only-insecure-...` | JWT 签名密钥（HS256）；`DEBUG=false` 时用默认值或短于 32 字节会**拒绝启动** |
| `AUTH_TOKEN_TTL_MINUTES` | `720` | 令牌有效期（分钟）；JWT 无状态、无法单独撤销，短 TTL 是泄漏后的唯一收敛手段 |
| `ENABLE_DOCS` | `true` | 是否开放 `/docs` `/redoc` `/openapi.json`；生产环境建议关闭 |
| `OCR_BACKEND` | `none` | `none` 或 `paddle`；关闭时图片/扫描件会明确报错而不是写入占位文本 |
| `ALLOW_LOCAL_IMPORT` | `false` | 是否允许 `import-path`（服务端文件系统读取能力），配合 `LOCAL_IMPORT_ROOT` 限定目录 |
| `CORS_ORIGINS` | `http://localhost:3000,...` | 前端来源白名单 |
| `LOG_FORMAT` | `text` | `text`（人读）/ `json`（一行一个 JSON，供采集端解析） |
| `LANGSMITH_TRACING` | `false` | LangSmith 追踪总开关，**默认关闭**；关闭时 span 照常计时并落本地日志 |
| `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT` | 空 / `inner-rag` | 追踪上报的目标；Key 为空时只降级告警，不影响请求 |
| `METRICS_BACKEND` / `METRICS_TOKEN` | `none` / 空 | `/api/system/metrics` 的输出格式与可选门禁（非空时需 `X-Metrics-Token`） |

## 2. 两条必须记住的规则

1. **换 embedding 模型 = 换向量空间**：库内旧向量在新空间里没有意义，必须
   `uv run scripts/reindex_kb.py <kb_id>` 重建索引，不做原地格式转换。
2. **密钥只走环境变量**：`.env` 已 gitignore，但**追踪 / 评测类密钥（如 `LANGSMITH_API_KEY`）
   建议直接用 `export` 注入**，优先级高于 `.env`，既不用改文件也不会被误提交。

## 3. Model Provider 选型

Chat 与 Embedding 是**两个独立开关**，改 `.env` 即可，代码无需改动：

| Provider | 可用于 | 说明 |
| --- | --- | --- |
| `sentence_transformers` | **仅 embedding** | 本机跑 HuggingFace 开源权重（默认 `Qwen/Qwen3-Embedding-0.6B`，1024 维），需要 `uv sync --extra local-embed`；详见 §3.1 |
| `ollama` | chat + embedding | `http://localhost:11434`，本地、无需 key |
| `openrouter` | chat + embedding | `https://openrouter.ai/api/v1`，OpenAI 兼容聚合网关，一个 key 用数百个模型 |
| `deepseek` | **仅 chat** | `https://api.deepseek.com/v1`，官方集成（`langchain-deepseek`）；模型名以[官方文档](https://api-docs.deepseek.com)为准（默认 `deepseek-flash`，另有 `deepseek-v4-pro`）；官方**没有 embeddings 接口**，写成 `EMBEDDING_PROVIDER=deepseek` 会得到明确报错 |
| `openai` | chat + embedding | `https://api.openai.com/v1`；也可指向任何 OpenAI 兼容的自建网关 |
| `mock` | chat + embedding | 不联网、不需要 key、输出确定性；用于本地演示 / CI / 降级验收（只有词面相似度，不能用来评估检索效果） |

`GET /api/system/providers` 会列出全部可选项、当前选择与 key 是否已配置。

### 3.1 本地嵌入（默认，推荐）

默认就是这一条：**向量侧在本机跑开源权重**，不发外部请求。选它当默认是运维结论——云端网关的
免费额度按**请求数**限流（一整天 1000 次），而一次全库重建要 500+ 次请求，重试一次就超额度；
本地权重只受 GPU 时间约束，且文档不出机器。

```bash
# 依赖（可选分组，不进默认安装）
uv sync --extra local-embed          # sentence-transformers + torch

# .env（也是默认值，写出来只是显式化）
EMBEDDING_PROVIDER=sentence_transformers
SENTENCE_TRANSFORMERS_MODEL=Qwen/Qwen3-Embedding-0.6B
SENTENCE_TRANSFORMERS_DEVICE=auto
HF_ENDPOINT=https://hf-mirror.com    # 出网受限时用它换镜像；能直连官方站就留空
```

要点：

- **首次运行会下载权重**（0.6B 约 1.2GB）。之后从本地缓存加载，`SENTENCE_TRANSFORMERS_ALLOW_DOWNLOAD=false`
  可以强制「只用本地缓存」，适合离线部署；
- **首次入库前先单独试一次模型加载**，别把它和建库混在一起——权重下载慢/失败会表现为「建库卡住」；
- **换了 `SENTENCE_TRANSFORMERS_MODEL` 就是换了向量空间**，必须 `uv run scripts/reindex_kb.py <kb_id>` 重建。
- **CUDA 构建不能比宿主机驱动新**。`local-embed` 里的 torch 走 `pyproject.toml` 里
  `pytorch-cu124` 这条索引，不是因为 12.4 特殊，而是**wheel 内置的 CUDA runtime 不能新于驱动支持的版本**：
  实测 驱动 535（CUDA 12.2）+ PyPI 默认的 cu13x 构建时，`torch.cuda.is_available()` 会是 `False`，
  而 PyTorch 只在 stderr 上打一次 UserWarning——很容易被误当成「这台机器本来就没 GPU」。
  换机器时只改 `[[tool.uv.index]]` 的 URL（驱动 ≥580 用 cu130，≥560 用 cu129，≥550 用 cu128/cu126），
  `[tool.uv.sources]` 不用动。一条命令自检：

  ```bash
  uv run --extra local-embed python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
  ```

  末位是 `False` 时后端会**退回 CPU 并打一条 WARNING**（写明「torch 的 CUDA 构建=X」），
  检索仍能跑完，只是明显变慢——看到这条告警就先回来查本节。

- **多卡机器默认落 GPU 0**。共享机器上 GPU 0 可能被别的任务占满，本地嵌入会 `CUDA out of memory`。
  用 `CUDA_VISIBLE_DEVICES=<空闲卡号>` 选卡（后台跑长任务时注意 `nohup` 会吞掉前缀赋值，须用
  `nohup env CUDA_VISIBLE_DEVICES=3 uv run ...`，见 [operations.md §3](operations.md#3-依赖与模型)）。

### 3.2 云端 API（最快起量）

```bash
# .env —— Chat 走 DeepSeek，Embedding 走 OpenRouter（分属不同厂商完全可以）
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=<your-key>
DEEPSEEK_CHAT_MODEL=deepseek-flash     # 也可用 deepseek-v4-pro
# LLM_REASONING_EFFORT=high            # 可选：minimal/low/medium/high（仅 DeepSeek 生效）

EMBEDDING_PROVIDER=openrouter
OPENROUTER_API_KEY=<your-key>
EMBEDDING_MAX_INPUT_CHARS=400    # 见下方「小上下文模型」说明
```

> ⚠️ OpenRouter 的免费路由是**按请求数**限流的（`free_model_daily_requests`，默认 1000/天），
> 不是按 token。全库建库要发 500+ 次请求，再叠加重试就很紧张；查额度用
> `curl -s https://openrouter.ai/api/v1/key -H "Authorization: Bearer $OPENROUTER_API_KEY"`。
> 免费的 `liquid/lfm-2.5-embedding-350m:free`（1024 维）只有 **512 token** 上下文，建议同时设
> `EMBEDDING_MAX_INPUT_CHARS=400`；免费路由的数据可能被上游留存，有合规要求时换付费模型或改本地嵌入。

### 3.3 本地 Ollama（全离线对话 + 向量）

```bash
ollama pull qwen3:14b            # 对话模型
ollama pull qwen3-embedding:8b   # 向量模型（必须与建库时的 embedding 保持一致）
```

### 3.4 完全离线

`LLM_PROVIDER=mock` + `EMBEDDING_PROVIDER=mock`，不需要任何模型服务，也能跑通
「上传 → 检索 → 带引用回答」的完整链路。
