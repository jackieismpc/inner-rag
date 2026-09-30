# 测试策略

对应 `docs/DEVELOPMENT_PLAN.md` 第 5 节的门禁。三条核心原则：

1. **默认离线、可复现、不花钱**：`uv run pytest` 不联网、不依赖开发者本机 `.env`。
2. **覆盖准确 > 用例数量**：每个用例对应一个**唯一且真实的失效场景**；没有失效场景的断言不写。
3. **禁止重复用例**：同一行为只测一次；变体用 `parametrize`，不复制粘贴。

## 1. 测试分层（按失效场景划分，不按数量）

| 层 | 覆盖的失效场景 | 网络 | 命令 | 频率 |
| --- | --- | --- | --- | --- |
| **L1 单元** | 纯函数与单模块的错误：解析、分块边界、距离→相关度换算、缓存失效、provider spec 解析、基准指标算法 | 否 | `uv run pytest tests/test_parser.py tests/test_cache.py tests/test_benchmark_metrics.py …` | 每次改动 |
| **L2 离线集成** | 接口契约与状态机：建库 → 上传 → 入库（mock provider）→ 检索 → 问答 → 系统接口，401/403、阈值过滤、SSE 事件序列 | 否 | `uv run pytest -q` | 每次提交 |
| **L3 真实联网** | 只有真实 provider 才能暴露的问题：Key / 模型名、鉴权、真实 embedding 维度、LangSmith 上报 | 是 | `uv run pytest -m live -q` | 阶段收尾 / 改 provider 时 |
| **L4 评测** | 质量回归：检索与回答指标在真实语料上是否变差 | fixture 否，小库/全库 是 | `uv run python -m benchmark.run_bench --mode fixtures`（离线）/ `--mode kb`（+ Phase 6 的 `scripts/eval_answer.py`） | fixture 每次提交；小库见 `docs/evaluation.md` |

不设用例数量目标；README 里的用例数只是**现状快照**，不是 KPI。

现状（Phase 0–8.3，用例数是快照不是目标）：

- `pyproject.toml` 里 `addopts = "-m 'not live'"`，即**默认只跑离线用例**；
- `markers` 已注册 `live`；`live` 用例必须显式 `-m live` 才执行；
- `tests/conftest.py` 强制把 `LLM_PROVIDER` / `EMBEDDING_PROVIDER` 钉成 `mock`、设置
  `EMBEDDING_MAX_INPUT_CHARS`、并把 `TASK_QUEUE_BACKEND` 设为 `inline`
  （后台任务在请求内同步跑完，用例不必等队列），保证**不受开发者本机 `.env` 影响**；
- 用例数快照：**337 个离线用例**（Phase 8.3 后；`uv run pytest --collect-only -q` 报 `337/345`，
  差掉的 8 个是 `-m live` 联网验收用例），其中登录 / 鉴权 / ACL 在 `tests/test_auth.py`
  （多为参数化路由表，例如「11 条受保护路由全部 401」是一条用例的参数化而不是 11 条用例）；
- **契约测试参数化跑所有实现**，这是 Phase 7 的核心验收方式：
  - 向量库 `tests/test_vector_store.py`：`store` fixture 参数化跑 `zvec / chroma / memory`，
    同一份用例（相关度口径、阈值计数、MMR 无分数、元数据白名单、删除可见性）覆盖三个实现；
    另有一组**断点续跑**用例（见下）；
  - 缓存 `tests/test_cache.py`：`BACKEND_FACTORIES` 参数化跑后端，业务缓存用例单独一组；
  - 任务队列 `tests/test_task_queue.py`：并发上限、退避重试后的 attempts、永久失败的归因、历史有界；
  - 关系库 `tests/test_repositories.py`：直接对仓储断言（含 `history` 取最近 N 条、
    `reset_kb_for_reprocess` 不跨库误伤、删库级联），验证用**新开会话**读回，避免只验证身份映射缓存；
  - 插件注册表 `tests/test_plugins.py`：撞名抛错、entry point 失败跳过、同名保留内置，
    以及一个第三方 provider 只靠注册 + 改配置就跑通「建库 → 上传 → 提问」的端到端演练；
- 检索组合层与词面检索（Phase 8.1，两路都要有失效场景才写用例）：
  - `tests/test_lexical.py`：分词（中文 bigram 跨换行也成立、拉丁词小写、标点忽略）、
    BM25 排序与**分数无上界**、`filter_doc_ids` 过滤、索引按库缓存与 `invalidate` 语义、
    归一化把最大值映到 1.0；
  - `tests/test_retrieval.py`：融合去重取较大分、`None`（MMR 补充项）排在最后、
    **阈值在融合之后生效**（向量分低于阈值但词面完全匹配的分块必须能被救回）、
    **词面启用门槛**（两路都无实质证据时不采信词面，尤其是负样本题的形状）、
    以及三种策略各自是否调用词面（`similarity` 绝不调用、权重 0 = 显式关闭）；
    用假后端 + 假索引，不需要真向量库或网络；
- 精排与查询改写（Phase 8.2 / 8.3，同样是「先有失效场景再写用例」）：
  - `tests/test_rerank.py`：契约本身（`none` / `lexical` 都**保持条数与分数不变**）、
    `candidate_idf`（查询词人人皆有时权重趋近 0）、`parse_order`（去重 / 丢越界 / 垃圾输入）、
    LLM 后端解析与异常兜底、`RERANK_LLM_TOP_N` 之外的 tail 保持原序；
    以及**接进检索层的时机**——关闭时绝不调用、在阈值过滤**之后**才调用、
    重排结果真的决定谁进 context、实现违约（改条数）时退回原顺序；
  - `tests/test_query_rewrite.py`：契约「**首条恒为原查询**」参数化跑三个实现、
    `strip_stopwords` 只替换不删除（保留词边界）、`parse_aliases` 坏行告警跳过、
    alias 大小写不敏感且无匹配时保持惰性、模块级入口的截断 / 去重 / 实现违约时补回原查询、
    以及多查询确实「一个变体跑一次召回」（`seen == ["Sakura", "路明非"]`）；
  - `tests/test_embedding.py`：并发分批的**真并发**（用假后端记录 `max_inflight` 断言并发上限、
    按「批首文本」记账以区分重试与首次）、单批瞬时失败重试后成功、永久失败在耗尽尝试次数后抛出、
    **硬失败时取消兄弟批次**（不留下还在烧配额的请求）、批内保序、空输入不发请求、
    `EMBEDDING_TIMEOUT` 真的传给了 OpenAI 兼容后端；
  - 同文件末尾一组**本地后端的设备解析**（用假 `torch` 注入 `is_available()` 与 `version.cuda`）：
    有 CUDA 时选 `cuda`、显式 `cuda:1` 原样保留、显式 `cpu` 时**根本不去 import torch**、
    CUDA 不可用时**退回 CPU 并把成因写进 WARNING**（「构建比驱动新」与「CPU 版 torch」两种文案要能区分）、
    没装 torch 的机器上构造实例也不炸、解析结果**只算一次**（否则每取一次 device 就重探一次、告警重复刷屏），
    外加「构造器不做重活」（返回后 `_model is None`，权重未被加载）；
- 入库的**幂等与断点续跑**（Phase 8.4，`tests/test_vector_store.py`）——它是一组契约用例，
  因为「重跑等于续跑」是写入路径的**基本要求**而不是某个后端的优化：
  - `test_resume_after_interruption_fills_the_gap`：用「第 N 次批量嵌入抛异常」制造**真实中断**，
    断言中断前已落库（partial > 0）、重跑只补缺口（`written == total - partial`）、
    总数与源页数一致（不遗漏）、分块键集合恰好是 0..total-1（不重复）、再跑一次返回 0；
  - `test_interruption_cancels_remaining_batches`：在 `embed_batches_in_order` 层面验证「失败即取消兄弟批次」。
    用**20ms 对 5s 的时差**做确定性判定（第 2 批慢到「没被取消就一定会跑完」），
    而不是靠假后端嵌入的调度运气——否则这条用例会间歇性假绿；
  - `test_stored_chunk_keys_reports_only_that_document` / `.._is_empty_for_unknown_kb`：
    进度查询按文档隔离、未入库的库返回空集而不是抛异常；
  - `test_normalized_chunk_key_stringifies_chunk_index`：`chunk_index` 在内存后端是 int、
    在 zvec schema 里是 STRING，不统一字符串化会把「已入库」判成「没有」，续跑时白跑一整份文档；
  - 配套断言写入语义的变化：`add_documents` 返回「**本次新写入**」的分块数，因此
    `test_zvec_reingest_does_not_accumulate` / `test_memory_reingest_does_not_accumulate`
    的第二次数值从「分块总数」改成了 **0**（行为契约变了，用例跟着改）；
- **答案保密**（Phase 8.4，`tests/test_eval_pipeline.py`）：
  `test_answer_llm_never_sees_reference_answer` 跑一次真实的 `run_bench.run_kb(--answer)`，
  在模型边界上**捕获每一个 prompt**，断言参考答案 / 关键词 / 禁止词一个都没出现，
  同时断言语料正文**在** prompt 里（否则这条用例只是在证明「什么都没发」）；
  `test_answer_llm_prompt_still_contains_retrieved_context` 是它的反面（prompt 必须含召回内容）。
  这两条把「不给模型看答案」从约定变成了断言——参见 `docs/evaluation.md` 2.4；
- 测试库与向量库都用临时目录，不写 `./data`；跑完即清理；
- `benchmark/` 的指标与评测集校验也有离线用例（`tests/test_benchmark_metrics.py`，
  含 `citation_hit` 与 `citation_precision` 的口径差异、以及缺字段时聚合成 `None` 而不是 0 分）；
  该文件还钉住三条容易悄悄退化的不变量：
  - **页文本与入库解析同源**：`test_page_text_is_the_same_as_what_ingest_parsed` 用**真实
    `DocumentParser` 解析单页**，断言结果与 `benchmark.dataset.PageText` 完全一致。这条守的是
    「锚点校验读的文本」＝「入库时读的文本」——早先是两套提取器（校验用 PyMuPDF、入库用 pypdf），
    同一页给出的文本不同，正确的题会被判成锚点错误；
  - **fixture 片段必须留在上限内**：`test_excerpt_fits_under_the_limit_on_a_long_page` 用 1000 字的页
    验证「窗口扩到 199 字又被 `≥199` 否掉」这类 off-by-one 不会回归（旧实现下长页一律裁不出片段）；
  - **按库内证据筛题 / 改判**：`test_filter_by_evidence_skips_incomplete_items`（小库上引用页不在
    库中的题被跳过，「只进来部分引用页」的题也跳过——避免把拒答算成检索失败）、
    `test_filter_by_evidence_can_turn_absent_items_into_refusals`（`refuse` 模式把它们改判成
    拒答题参与 `refusal_accuracy`，且必须**复制**而不能原地改，否则污染调用方的 items）、
    `test_filter_by_evidence_keeps_everything_when_the_kb_is_complete`（全库上一条都不许跳，
    否则「筛掉难例涨分」就成了捷径）；
  `--mode fixtures` 的评测自检不在 pytest 里，要单独跑（见第 6 节）。

## 2. 目录与命名

```
tests/
├── conftest.py              # 环境钉死 + 公共 fixture（临时库、已登录 client、建号/登录辅助、假 provider）
├── test_auth.py             # L2：登录 / 鉴权 / ACL（401·403 路由表、伪造 Token、成员权限）
├── test_api.py              # L2：kb / document / chat / system 接口
├── test_cache.py            # L1/L2：缓存引擎 + CacheBackend 契约（参数化跑后端）+ query/embedding 缓存语义
├── test_task_queue.py       # L1/L2：并发上限、退避重试、失败归因、历史有界、未知后端报错
├── test_repositories.py     # L1/L2：仓储契约（事务边界、排序、作用域、级联）
├── test_plugins.py          # L1/L2：注册表语义 + 第三方 provider / 插件状态接口的端到端演练
├── test_parser.py           # L1/L2：解析与 OCR 后端行为
├── test_providers.py        # L1：spec 解析、错误文案、健康检查状态机
├── test_vector_store.py     # L1/L2：写入、检索策略、阈值、相关度换算（参数化跑 zvec / chroma / memory）
├── test_lexical.py          # L1：中文 bigram 分词、BM25 排序、索引按库缓存与失效、归一化
├── test_retrieval.py        # L1/L2：融合去重取较大分、阈值时机、词面启用门槛、三策略调用面
├── test_rerank.py           # L1/L2：精排契约（只改顺序）、candidate_idf、parse_order、接入时机与违约兜底
├── test_query_rewrite.py    # L1/L2：改写契约（首条恒为原查询）、停用词剥离、别名表、多查询召回
├── test_embedding.py        # L1：并发分批真并发、单批重试、硬失败取消兄弟批次、批内保序、超时下发、设备解析
├── test_eval_pipeline.py    # L4：建库 manifest、页区间命中判定、回答侧评测的聚合与对比
├── test_benchmark_metrics.py # L1/L4：基准指标算法、评测集 schema 与锚点校验
├── test_observability.py    # L1/L2：request_id、结构化日志、span 树、指标与 Prometheus 导出
├── test_langsmith_live.py   # L3：LangSmith 上报读回（-m live）
└── test_live_providers.py   # L3：真实联网（-m live）
```

命名与编写规范：

- 文件名 `test_<被测模块>.py`，用例名 `test_<行为>_<条件>`（例：`test_chat_stream_returns_sources`）；
- 断言失败信息要能自解释（带上关键数值），不要只写 `assert x == y`；
- **一个用例一个失效场景**：写完先问「它红了说明什么坏了」，答不上来就删掉；
- **不允许重复**：同一行为只保留一个用例，参数组合用 `parametrize`（多 provider / 多策略等）；
- **每个 bug fix 必须附一个先红后绿的回归用例**（没有回归用例的修复视为未完成）；
- **合并 / 删除用例**：必须在提交说明里写理由（通常意味着行为契约变了）；
- **不写凑覆盖率的空断言**：`assert response is not None` 这类没有失效含义的断言不算覆盖。

## 3. 离线集成测试的 fixture 约定

- **环境隔离**：`conftest.py` 的 `Settings` 覆盖必须在 import `inner_rag` 之前生效，否则
  `settings` 单例已经读走 `.env`；
- **临时资源**：SQLite 用临时文件、向量库（zvec / Chroma）与上传目录用 `tmp_path`，
  绝不共用 `./data`；
- **假 provider**：`LLM_PROVIDER=mock` / `EMBEDDING_PROVIDER=mock`（确定性哈希词袋向量），
  断言只依赖「词面相似」这种可控特性；
- **账号与登录**：`conftest.py` 暴露 `create_user` / `set_user_active` / `login` / `bearer` 与
  `TEST_PASSWORD`；`client` 与 `other_client` 是「甲 / 乙」两个已登录用户（后者专用来断言跨库 403
  与列表隔离），`anonymous_client` 不带 Token。账号走与 `scripts/create_user.py` 同一条代码路径创建
  ——系统没有注册接口，测试也不该绕过鉴权塞数据；
- **假时钟 / 假网络**：需要 TTL 的缓存用例用 monkeypatch 时间，不用 `sleep`；
  任何 HTTP 调用（Phase 5 的 tracing 上报、Phase 9 的 VLM）都要能 monkeypatch——
  `tests/test_observability.py` 就是把 `httpx` 的两个 transport 换成「一调用即抛」来断言零网络的；
- **严禁真实 Key**：测试进程里不许出现真实 Key；`live` 用例从环境变量读，缺失时 `skip`
  并给出「缺少 X_API_KEY」的明确原因。

## 4. L3 联网用例规范

- 一律 `pytest.mark.live`，并放在 `test_live_providers.py`；
- **只覆盖关键路径**，不为「看起来完整」给每个 provider 堆用例；缺 Key 就
  `pytest.skip("缺少 DEEPSEEK_API_KEY")` 并说明缺哪个变量，不静默跳过；
- 断言要**稳健**：不比对模型自由文本的全文，只断言结构（非空、含引用标记、
  关键实体出现），避免模型换版本就红；
- 有成本意识：单次运行控制在个位数请求；不使用全库做联网测试；
- 不把 Key 写进 `pytest.ini` / `conftest.py` / CI 明文；CI 用仓库 Secrets。

## 5. L4 评测用例规范

- 评测数据集与 fixture 的 schema、锚点校验见 `docs/evaluation.md`；
- L4 的自动化入口是 `uv run python -m benchmark.run_bench --mode fixtures`：必须能在**无 PDF、无网络**下
  跑通（用 `docs/datasets/dragon_king/fixtures/`），全程 mock provider；
- 评测代码本身要有单测（指标算得对、schema 校验能拦住坏数据）：`tests/test_benchmark_metrics.py`；
- 涉及检索 / 分块 / Prompt 的改动，还要跑一次 `--mode kb` 并把结果写进 README 基准表（`--update-readme`）；
- 报告落 `benchmark/results/*.json`（kb 模式每次运行都有；fixtures 自检不落盘）与 `docs/reports/`（Phase 6 起），
  里程碑数字更新 `docs/evaluation.md` 的基线表。

## 6. 门禁与命令速查

```bash
# 每次提交（G0）
uv run ruff check . && uv run ruff format --check .
uv run python -m benchmark.run_bench --mode fixtures   # 评测自检（离线，不写 README、不落盘）

# 阶段收尾（G1）
uv run mypy
uv run pytest -q                    # 离线全量（含 benchmark 单测）
uv run pytest -q tests/test_auth.py # 只看登录 / 鉴权 / ACL
uv run alembic upgrade head && uv run alembic check

# 真实链路（G2）
uv run pytest -m live -q

# 评测（G3）
uv run python -m benchmark.run_bench --mode kb --kb-id <小库> --answer --update-readme
uv run scripts/eval_answer.py --kb-id <小库>   # Phase 6 起：judge 类指标与成本
```

补充规则：

- **覆盖准确 > 数量**：新增能力要补「真实失效场景」的用例；补不出失效场景，说明改动可能没有可观测行为；
- **新增用例前先查重**：已有用例已覆盖同一行为时，用 `parametrize` 扩展它，而不是新建；
- **删 / 并用例要有理由**：提交说明写清「为什么不再需要」，通常意味着行为契约变了；
- **修 bug 先补用例**：没有回归用例的修复视为未完成；
- **不跳过离线用例**：`xfail` / `skip` 必须写明原因与解除条件；
- **README 里的用例数**只是现状快照，变更时同步更新，不作为 KPI。

## 7. CI 计划（Phase 10 落地）

| 作业 | 触发 | 内容 |
| --- | --- | --- |
| `lint` | 每次 push / PR | `ruff check` + `ruff format --check` + `mypy` |
| `test-offline` | 每次 push / PR | `pytest -q`（离线）+ `benchmark.run_bench --mode fixtures`（评测自检）+ Alembic 迁移自检 |
| `build` | 每次 push / PR | 构建 Docker 镜像（不推送） |
| `live-smoke` | 每日定时 / 手动 | `pytest -m live -q`（用 Secrets 里的 Key） |
| `eval-small` | 每日定时 / 手动 | 小库评测，产出趋势报告与 LangSmith experiment |

约束：

- `live-*` 与 `eval-small` **不允许**在 PR 上自动跑（成本与密钥权限），只能定时或手动触发；
- CI 里的评测结果要留档（artifact），便于对比与回溯；
- CI 失败时优先看是否「本机通过、CI 失败」——通常是依赖了本机 `.env` 或本地文件，
  这类问题必须在测试里当成 bug 修，而不是加环境变量绕过。
