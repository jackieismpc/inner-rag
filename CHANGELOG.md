# Changelog

> 本文件记录**每一次改动**，用于回溯与面试复述：每个阶段**为什么做、怎么做、效果如何**。
> 规则（强制）见 `AGENTS.md` 第 1.1 节：**每次 push 前必须先在顶部新增一条记录**，否则不推。
>
> 阅读方式：最新在上。历史阶段（Phase 0 起）与后续阶段统一用同一套字段。

## 记录格式

```
## [阶段] YYYY-MM-DD — 标题
- 类型：新增功能 | 优化 | 修复 | 重构 | 文档
- 目的：为什么做（要解决的问题 / 面试时可讲的动机）
- 方案：怎么做（关键取舍与取舍理由）
- 效果：可验证的结果（命令 / 用例 / 指标 / 结论），禁止「应该没问题」
- 涉及提交：<sha 或分支/PR>
```

字段说明：

- **类型**固定五选一，便于区分「增加功能」与「优化」——面试官常问"这是新功能还是优化"，在这里就要能一眼回答。
- **目的**回答"为什么"，**效果**回答"有没有用"；两者缺一，这条记录就失去价值。
- **效果**必须能被复现或引用（跑过的命令、通过的用例数、评测指标、trace 链接等）。
---

## [Phase 8.2/8.4] 2026-09-30 — 本地 GPU 嵌入落地：Qwen3-Embedding 替换 OpenRouter，全库 11138 分块全覆盖

- 类型：优化 + 修复
- 目的：云端 OpenRouter 免费路由按**请求数**限流（1000/天），全库建库要发 550+ 次请求，反复调优时
  根本不够用；且免费 350m 模型只有 512 token 上下文，检索召回被截断拖累。换成本地部署的
  Qwen3-Embedding-0.6B（1024 维、32k 上下文、无配额无费用），并让它在 A100 上跑 GPU 推理。
- 方案：
  - 新增 `SentenceTransformerEmbeddings`（`providers/embeddings.py`）：懒加载（构造器不碰权重）、
    `_load()` 内 import（sentence-transformers 是可选 extra）、`threading.Lock` 串行化 GPU 推理；
    query 侧在模型声明 `prompts` 时传 `prompt_name="query"` 激活 instruction 前缀；
  - **CUDA 构建兼容性（本次最深的坑）**：PyPI 默认 torch 已是 cu13x 构建，而驱动 535（CUDA 12.2）
    不支持 → `torch.cuda.is_available()=False`，且 PyTorch 只在 stderr 打一次 UserWarning。
    修法：`pyproject.toml` 加 `[[tool.uv.index]] pytorch-cu124` + `[tool.uv.sources] torch = { index = ... }`
    + pin `torch>=2.6,<2.7`。`_resolve_device()` 在 CUDA 不可用时**退回 CPU 并告警写清华因**，
    而不是抛错（「任何单点问题都不该让整体流程失败」）；
  - **多卡选卡**：共享机器上 GPU 0 可能被占满，本地嵌入默认落 GPU 0 会 `CUDA out of memory`。
    用 `CUDA_VISIBLE_DEVICES=<空闲卡>` 选卡；坑在 `nohup` 会吞掉前缀赋值，须写
    `nohup env CUDA_VISIBLE_DEVICES=3 uv run ...`（见 operations.md §3）；
  - **修复全量覆盖 bug**：`_parse_meta` 的 `page_count`（段落数 11138）被误当「最大页号」切 full 窗口，
    导致末尾 27 页正文（页号 11139-11165，路山彦决战/尾声/校长等核心剧情）漏入库。拆出 `max_page`
    （最大物理页号 11165）供 `resolve_windows('full', ...)` 使用——空页被解析跳过、页号稀疏，
    窗口必须覆盖到最大页号而非段落数。补回归用例 `test_parse_meta_distinguishes_max_page_from_page_count`。
- 效果：
  - `torch 2.6.0+cu124`、`cuda.is_available()=True`、4×A100 可见；语义区分度 +0.326（相关 0.5955 vs 无关 0.2698）；
  - 全库重建 **11138 分块 / 66.5s**（覆盖全部非空页，末尾 27 页核验已入库），小库 227 分块 / 21s；
  - **新基线（全库 / 本地 Qwen / hybrid / k=8 / 146 题）**：Recall@8=79.3%、MRR=0.565、页命中率 78.5%、
    **检索 p50=123.8ms**（对比旧 openrouter 350m 小库 p50=1238.6ms，快 10 倍且免费）；
  - 质量门禁：ruff check/format 通过、mypy 59 源文件无问题、pytest **368 passed**；
    修正过时用例 `test_resolve_windows_small_covers_all_anchor_pages` → `_covers_v1_anchor_pages`
    （评测集演进到 v3 后，小库本就不该覆盖全库锚点）；
  - 清理：删除 macOS `tar` 推送带入的 8 个 AppleDouble 残留（`._*.py`/`._*.md`，ruff 报 E902），
    以及根目录 3 个历史推送失误副本（`evaluation.md`/`operations.md`/`usage.md`）。
- 涉及提交：本条目所在提交（Phase 8.2/8.4：本地 Qwen 嵌入 + 全库全覆盖修复 + 评测集 v3 扩容 + 失败题定位）

---

## [Phase 8.1] 2026-09-30 — 检索质量提升：阈值不是瓶颈，补一路词面召回（向量 ∪ BM25）

- 类型：优化 + 新增功能
- 目的：Phase 6 把 baseline-0 的失败题归因成「阈值过严」，本阶段先用数据把这条假设证伪，再修真正的瓶颈。
  小库（kb1，227 页）上用新加的 `--threshold-sweep` 反推整条阈值曲线：**Recall@8 在阈值 0 → 0.30 之间
  完全不变（75.0%）**，把阈值降到 0 也一道题都救不回来。逐题看，两道失败题是同一个形状——问题里的
  关键词在语料里**字面就存在**（「诺诺/绘梨衣/言灵」），但免费 350m embedding + 512 token 截断的
  稠密召回没把答案页放进 Top-8。所以缺的不是更好的 embedding，而是**缺一路稀疏检索**。
- 方案：
  - 新增 `services/lexical.py`：BM25（`k1=1.5` / `b=0.75`）+ 中文**字 bigram** 分词（零依赖，先取字序列
    再组 bigram，所以正文里「陈墨\n瞳」这种跨行排版仍能匹配）；倒排索引按知识库缓存在进程内，
    由新增的 `VectorStore.iter_chunks` 构建（离线读路径，不进检索热路径），内容变化时 `invalidate(kb_id)`，
    调用点与 `query_cache.invalidate_kb*` 成对出现；
  - 新增 `services/retrieval.py`（检索组合层）：`fuse()` 按 `chunk_key` 去重、取两侧较大分；
    阈值**只在融合之后**生效（传给向量库的 `score_threshold` 恒为 0，否则「向量分低但词面完全匹配」
    的候选会被提前丢掉，而补这一路正是融合的目的）；`rag_service.retrieve` 与 `benchmark/run_bench.py`
    都改调这一层——评测跑的就是线上那条路；
  - `strategy` 对外语义不变（`similarity` / `mmr` 仍是纯向量；`hybrid` = 向量 ∪ 词面），
    新增 `HYBRID_SPARSE_WEIGHT`（默认 0.6）与 `HYBRID_MIN_SPARSE_SCORE`（默认 20.0）；
  - **前置门**：BM25 归一化后词面第一名恒为权重值，任何字面重叠都必然过阈值。实测负样本题
    「巴黎在哪个国家？」因正文里有一款香槟叫「**巴黎之花**美丽时光」而被命中、注入 3 条 context，
    模型随即放弃拒答（**拒答正确率 100% → 0%**）。因此词面这一路只在「两路里至少一路有实质证据」时
    启用：稠密侧有过阈值候选 **或** 词面最强匹配 ≥ 门槛。判据落在 query 级信号上是有原因的——
    按目标页分数分不开（14.01 vs 13.63），按 query 最强匹配差 1.8 倍（24.83 vs 13.63）；
  - 评测侧补齐 `--threshold-sweep`（一次未过滤召回反推任意阈值下的指标，不必逐阈值重跑）、
    逐题落盘 `scores` / `expected_pages`、结果 JSON 里记 `hybrid_sparse_weight`（否则 JSON 无法自解释），
    并新增**引用命中率** `metrics.spans_citation_hit`（见下）。
- 效果（可复现，同一 config 快照，小库 kb1 / hybrid / k=8 / th=0.3）：

  | 指标 | 改动前（`HYBRID_SPARSE_WEIGHT=0`） | 改动后（w=0.6 + 门槛） |
  | --- | --- | --- |
  | Recall@8 | 75.0% | **100.0%** |
  | 引用命中率 | 75.0% | **87.5%** |
  | 要点命中率 | 75.0% | **100.0%** |
  | 拒答正确率 | 100.0% | **100.0%** |
  | MRR / 页命中率 | 0.688 / 62.5% | 0.745 / 87.5% |
  | 检索 p50 | 1290.2ms | 1222.4ms |

  - 三项 gated 指标全部上升或持平，**无 >2pp 回归**；DoD 目标（Recall@8 ≥ 0.8、引用命中率 ≥ 0.8、
    拒答正确率 ≥ 0.9）**全部达成**；
  - 关闭词面（`w=0`）重跑**逐项复现**了 baseline-0（75.0% / 0.688 / 62.5% / 75.0% / 100.0%），
    证明词面这一路是纯增量、没有改动原有得分；
  - 过程中被门禁挡下两次，证据都留在本地 `benchmark/results/`（按既有约定该目录不入库）：①无门槛的融合 → 拒答正确率掉到 0%；②只用「稠密侧有过阈值候选」
    当门 → `nonno-real-name` 又被挡掉（该题稠密侧本来就没有过阈值候选，正是要靠词面救的那一类）。
    结论：门禁必须同时含检索指标与拒答指标，只看 Recall 会把破坏拒答的改动放进主干；
  - **发现并修掉一个指标缺陷**：`citation_precision` 的分母是「本题引用了多少条来源」，召回变好会让
    原本一条都没引用的题（记 0.0）变成引满 5 条（记 1/5），分母随配置漂移（4.1 → 5.0 条），
    27.5% → 20.0% 看着像 7.5pp 回归、其实不可比。新增可比口径 `citation_hit_rate`
    （引用的来源里至少一条命中期望页的题占比，分母恒为题数），同期 75.0% → 87.5%；
    终端逐题表改打命中率，避免再被精度的两个极端误导；
  - 已知边界（写在 `docs/evaluation.md` 4.5，不藏）：`HYBRID_MIN_SPARSE_SCORE` 是**绝对** BM25 量级、
    随语料规模增长，只有 1 正 1 负样本支撑，上全库前必须重标；更稳的替代判据是「命中多少语料内罕见的
    query 词元」（与语料规模无关），已列入待办。fixtures 自检（10 个短分块）量级远低于门槛、
    词面基本被门挡掉，因此其 Recall@8 仍是 75.0%——自检数字不代表检索质量，但正说明门槛对语料规模敏感；
  - 离线用例 **234 → 279**（新增 `test_lexical.py` 与 `test_retrieval.py`，以及引用命中率的 4 条）；
  - 门禁全绿：`ruff check` / `ruff format --check`（102 files）/ `mypy`（57 source files，0 error）/
    `pytest -q` → **279 passed, 8 deselected**；`benchmark --mode fixtures` 通过。
- 涉及提交：本条目所在提交（Phase 8.1：词面检索 + 融合 + 前置门 + 指标与文档同步）

---

## [Phase 7] 2026-09-30 — 可插拔深化：五个插件点统一走注册表

- 类型：重构 + 新增功能
- 目的：Phase 0–6 已经把「能跑、能控权限、能排障、能算准」做完了，但「可插拔」还只在向量库这一处成立：
  provider 靠 `if name == "xxx"` 分支，关系库的 SQLAlchemy 会话散落在路由 / 服务 / 脚本里，
  缓存与后台任务直接写死了进程内实现。结果是**换任何后端都要改业务代码**，
  而「可插拔」一旦不能兑现，第三方就无法接入、招聘方也无法验证——本阶段要把它变成**可被外部验证的能力**。
- 方案：五个插件点（provider / 向量库 / 缓存 / 队列 / 关系库）各自补齐「接口 + 内置实现 + 配置项 +
  探活 + 契约测试」五件套，名单统一由 `plugins/registry.py` 的 `Registry[T]` 维护。
  - `Registry[T]`：名字 → 实现，支持 entry point 发现（`inner_rag.chat_providers` 等五个组名）；
    同名重复注册直接抛错（静默覆盖会让「到底加载了哪个实现」变成谜）；第三方 entry point 加载失败
    只告警并跳过，与内置同名时保留内置实现。
  - provider：`spec` 与 `build` **成对注册**（避免注册一半：探活能过但实例建不出来）；
    注册改为惰性（`specs._ensure_builtins`），解开 `specs` ↔ `chat` / `embeddings` 的循环 import。
  - 缓存：`CacheBackend` 契约 + 每 namespace 一份有界 LRU；通配失效对齐 Redis `SCAN MATCH` 语义
    （`1:*` 不误删 `11:*`）。
  - 队列：`TaskQueue` 契约 + `inprocess`（并发上限 + 退避重试 + 有界历史）与 `inline`（同步执行），
    取代 FastAPI `BackgroundTasks`；`process_document` 失败改为**抛异常**，让队列能决定是否重试。
  - 关系库：`repositories/` 作为唯一入口，四个聚合契约（用户只读 / 知识库+成员 / 文档+状态机 /
    会话+消息），事务边界写在仓储里；`core/access.py` 改依赖 `ACLReader` 协议，`core/` 不再反向
    import 实现；`api/`、`services/`、`scripts/` 全部收敛（`create_user.py` 是显式例外：口令写入属运维动作）。
  - 替换演练：新增 `memory` 向量库（零依赖进程内实现）+ MMR / 余弦 / 混合检索合并三处语义从 zvec
    适配器提到 `base.py` 共用 + `GET /api/system/plugins` 暴露插件状态。
- 效果（可复现）：
  - **替换演练成立**：新增 `memory` 后端只改了 2 个文件（实现类 + 一行 `vector_stores.register`）
    加 1 行测试参数，`services/rag.py` 与 `api/*.py` **零改动**，
    `tests/test_vector_store.py` 的 `store` fixture 从 2 个后端扩到 3 个后**全部契约用例直接通过**；
  - **业务层已无裸会话**：`grep -rn "SessionLocal\|db\.query" src/inner_rag/{api,services,core}` 无匹配；
  - `GET /api/system/plugins` 返回五个插件点的 `configured` / `active` / `available` / `third_party`，
    `/api/system/health` 返回同一份报告并据此降级（`status=degraded`）；
  - 离线用例 **189 → 234**（新增 `test_repositories.py` 28 条、`test_plugins.py` 注册表语义与
    插件状态接口、`test_vector_store.py` 的 memory 参数与后端边界）；
  - 门禁全绿：`ruff check` / `ruff format --check`（98 files）/ `mypy`（55 source files，0 error）/
    `pytest -q` → **234 passed, 8 deselected**。
- 涉及提交：`c84f83e`（provider 注册表）、`5820568`（缓存抽象）、`614c8df`（任务队列）、
  `65d795e`（关系库仓储）+ 收尾的 7.5 替换演练与文档同步

---


## [Phase 5/6] 2026-09-30 — LangSmith 真实连通性打通（trace 与 feedback 双向验证）

- 类型：修复 + 新增功能
- 目的：Phase 5 的追踪与 Phase 6 的评测回写此前**只在离线侧验证过调用序列**，真实 LangSmith 从未连过。
  结果是三处「静默失败」：脚本入口不调 `tracer.configure()`（追踪压根没开）、`run_bench` 不记
  `trace_id`（feedback 回写恒为 0）、`run.post()` 不抛异常被当成上报成功（其实只进了本地队列）。
  三个问题都不报错，只是什么都不写——不验证就永远发现不了。
- 方案：
  - 密钥只从环境变量读（`export LANGSMITH_API_KEY=...`），项目 `.env` 只留非密钥项，避免密钥被复制出去；
  - `run_bench --answer` 给每题套 `eval.item` span 当树根，把根 run id 写进结果的 `trace_id`；
  - `eval_answer.py` 打完 judge 分数后按 `trace_id` 回写 `correctness` / `citation_precision` / `faithfulness`；
  - `create_feedback` 补 `session_id`（不带会走已废弃路径），读回改用 `client.runs.retrieve(..., project_id=)`；
  - 新增 `scripts/check_langsmith.py`：建 trace → **服务端读回同一条 run** → 同步 dataset → 回写并读回 feedback。
- 效果（2026-09-30 实跑）：
  - `uv run scripts/check_langsmith.py --dataset` → `PASS`，`tracing_errors_total=0`，
    `project_id=bcf62371-...`，读回 run id 与本地一致；
  - 真实小库 9 题评测回写 **25 条 feedback**，抽查 `sakura-who` 单条 run 读回 3 个分数
    （correctness=1.0 / citation_precision=0.4 / faithfulness=1.0）；
  - `uv run pytest -m live -q` → **8 passed**（新增 3 条：trace 往返、feedback 挂到指定 trace、dataset 幂等）；
  - 离线侧 `uv run pytest -q` → 172 passed，未受影响。
- 涉及提交：见本次推送（check_langsmith.py / langsmith_sync.py / run_bench.py / eval_answer.py / tests + docs）

---

## [Phase 6] 2026-09-30 — 评测体系与准确性基线（龙族真实语料）

- 类型：新增功能
- 目的：Phase 0–5 解决了「能跑、能换后端、能控权限、能排障」，但「答得准不准」始终是一句主观判断。
  改检索、改分块、改 Prompt 之后没有任何数字能回答「到底变好了没有」，调参就只能靠感觉。
  本阶段把这件事变成可复现的数字：同一套语料、同一套题、同一份配置快照，两次改动直接对比。
- 方案：
  - **建库脚本化**（`scripts/build_eval_kb.py`）：从 `data/uploads/龙族.pdf` 按页窗口建小库/全库，
    产出 `docs/reports/eval-kb-<日期>-<profile>.json` 记录页窗口、分块数、embedding identity、
    PDF sha256 与耗时——评测结论要可复现，就必须知道「这个库到底是哪几页」。
  - **回答评测**（`scripts/eval_answer.py`）：LLM-as-judge 正确性 + 忠实度 + token + 失败归因 +
    与上次报告的差值对比 → `docs/reports/eval-<日期>-<label>.md`。judge 模型与 prompt 版本写进报告
    （换任一都要重跑基线）。
  - **引用可核对**：分块元数据补 `page_start` / `page_end`（进 zvec schema 与 SSE `sources`），
    命中判定从「单页号」改为「页区间」（`metrics.spans_*`，单点口径保留且与区间口径一致）。
  - **LangSmith 联动**（`benchmark/langsmith_sync.py`）做成可选 sink：关闭时是 no-op，
    评测不许依赖跟踪后端。
  - 两个关键取舍：① **页码必须是源 PDF 物理页号**——早期把窗口页抽成子 PDF 再入库，库里存成了
    局部页号（1..227），引用翻不到原文、评测锚点（5904）全对不上，Recall 直接归零而回答其实是对的；
    现在改成解析源 PDF 后「只筛选、不重编号」。② **成本不内置价格表**：单价属计费域，
    默认 0 并在报告标注「未配置价格表」，要算钱用 `--price-prompt/--price-completion`。
- 效果：
  - 小库 `dragon_king_small`（kb_id=1）：227 页 / 227 分块，构建 312.8s，manifest 已落盘；
    无关章节自检 0 命中（噪声段确实不含评测证据原文）。
  - 基线（9 题：8 正样本 + 1 负样本，hybrid / k=8 / th=0.3，lfm-2.5-350m + deepseek-flash）：
    Recall@8 **75.0%**、MRR 0.688、页命中率 62.5%、要点命中率 75.0%、引用精度 **27.5%**、
    judge 正确率 75.0%、忠实度 87.5%、拒答正确率 **100.0%**、误拒率 12.5%，检索 p50 1,641ms。
    报告：`docs/reports/eval-2026-09-30-small-hybrid-k-8-th-0.3.md`。
  - 失败归因给出了 Phase 8 的两个不同方向：`nonno-real-name` 是阈值过严（8 条召回全被 th=0.3 挡掉），
    `erie-lingyan` 是检索失败（召回页与证据页无关，改阈值救不了）。
  - 离线用例 149 → **172**（新增 `tests/test_eval_pipeline.py` 23 例）；`ruff check` / `mypy` 48 文件
    全绿；`uv run python -m benchmark.run_bench --mode fixtures` 自检通过。
  - 同时纠正了文档里的事实错误：源 PDF 是 11,138 非空页 / 平均 211 字符每页，且**一个分块就是一整页**
    （不是原以为的「1000 字符≈5 页、全库约 2,900–3,000 分块」，实际约 11,138 分块）。
- 明确未做：全库未建（11,138 分块的耗时与费用只适合里程碑跑，脚本已支持 `--profile full`）；
  成本估算留空（Phase 10）；LangSmith 真实连通性未验证（本环境无 Key，仅用假客户端覆盖调用序列）。
- 涉及提交：见本阶段提交（建库脚本 / 回答评测 / 页区间与指标 / 测试 / 文档与基线）

---

## [Phase 5] 2026-09-30 — 可观测性：request_id / span 树 / LangSmith 追踪 / 指标端点

- 类型：新增功能
- 目的：Phase 0–4 把「能跑、能换后端、能控权限」做完了，但线上出问题时只有一个 500 和几行文本日志——
  跨步骤耗时无法归因（检索慢还是生成慢？），同一次请求的日志串不起来（谁的、哪一次？），
  效果与成本也没有量化口径。Phase 5 的目标是把「看日志猜」变成「拿 request_id 串日志、
  拿 session_id 看 trace、拿 metrics 定阈值」。这也是 M2 里程碑的验收内容。
- 方案：
  - **request_id 与日志**：`core/logging.py` 把 loguru sink 统一为 `text`（人读）/ `json`（一行一个 JSON）
    两种格式，结构化字段走 `logger.bind(event=..., ...)`——一处埋点两种格式都成立。
    `RequestIdMiddleware` 透传或生成 `X-Request-ID`、回写响应头、发访问日志并记请求指标；
    生成与访问日志放在**同一个**中间件（两者都要包一层 `send`，拆开会让每个响应多一层包装）。
  - **Tracer 门面**：`core/observability.py` 对外只有 `tracer.span(name, **metadata)` 一个入口，
    span 树用 contextvar 串父子（调用方不用传 parent）。LangSmith 是**可选 sink**：
    默认关闭（测试与 CI 零网络零费用）；未配 Key / Client 初始化失败 / 运行中上报失败
    一律降级为本地计时日志并计 `tracing_errors_total`，**绝不抛给请求**。
    采样只在根 span 判定（半棵树的 trace 没法排障），失败请求 100% 记录。
    trace 只记元数据与统计，不记 Prompt 与回答正文。
  - **埋点**：问答 `rag.request → retrieve → (cache.query | vector.search → embed.query) →
    prompt.build → llm.generate`；入库 `ingest.document → parse / vector.ingest`（`embed.documents`
    嵌在后者内，由 `services/embedding.py` 自埋，因此不需要改 `VectorStore` 契约）。
    为拿 token 用量，模型调用从「一条链 `ainvoke` 出字符串」改成「渲染消息 → `ainvoke` 拿 AIMessage →
    `StrOutputParser` 取文本」——`usage_metadata` 只在消息对象上。
  - **指标**：`core/metrics.py`（counter + 有界蓄水池直方图，p50/p95/p99）+
    `GET /api/system/metrics`（JSON 或 Prometheus 文本，免登录，`METRICS_TOKEN` 非空时用
    `compare_digest` 校验）。
- 效果：
  - `uv run pytest -q` → **149 passed, 5 deselected**（Phase 4 为 139）：新增
    `tests/test_observability.py` 10 条，逐条对应 `docs/observability.md` 第 6 节的 DoD。
  - `uv run mypy` → Success: no issues found in **48 source files**；`uv run ruff check .` /
    `ruff format --check .` 全绿；`./scripts/gates.sh g1` 全绿（含迁移
    upgrade→check→downgrade→upgrade、冒烟探活 + openapi、changelog 与密钥自检）。
  - DoD 逐项结论（细节见 `docs/observability.md` 第 6 节表格）：
    ① 默认本地后端、关闭时**零网络调用**（把 `httpx` 两个 transport 换成「一调用即抛」，
    跑完整问答仍 200）；② 上报失败请求仍 200 且 `tracing_errors_total` 计数；
    ③ `LOG_FORMAT=json` 每行可 `json.loads`，`request_id` 在响应头与日志里同值、上游传来的原样透传；
    ④ span 父子层级（含 `vector.search → embed.query`）由用例断言；
    ⑤ `.env.example` / README「可观测性」小节 / `docs/observability.md` 三者一致。
  - 真实 LangSmith trace（DoD 第 1 条的联网部分）待 G2 执行：需要 `LANGSMITH_API_KEY`，
    缺 Key 时不阻塞交付，已记入 `docs/observability.md` 的验证表。
- 明确不做（已写进文档，避免被当成遗漏）：成本指标 `rag_llm_cost_usd_total`
  （token 单价属计费域，不写没有来源的价格表，留到 Phase 10 的成本看板）；
  入库链路拆到 `chunk` / `vector.write`（会改 `VectorStore` 契约，留到 Phase 7）；
  Prometheus 远程写与 OTLP 导出（Phase 10）。
- 涉及提交：371e4da（core 基建：日志 / 指标 / Tracer / 配置与入口）、6056a2a（问答与入库埋点）、
  f549c9b（`/api/system/metrics` 端点）、9cb3981（可观测性用例 + conftest 钉死追踪开关）、
  0c85a7e（文档同步）、本次提交（changelog 条目）

---

## [Phase 4] 2026-09-29 — 向量库统一到 zvec：VectorStore 契约 + 默认后端切换

- 类型：新增功能
- 目的：项目的向量库选型是 zvec（ADR 见 `docs/DEVELOPMENT_PLAN.md` 第 9 节），但代码一直落在 ChromaDB 上——
  「在 Chroma 上写、以后再在 zvec 上重验」意味着分块、阈值、MMR、删除语义都要做两遍，越晚迁移返工越大。
  本阶段把向量能力统一到 zvec，并在切换之前先抽出 `VectorStore` 契约，让两个后端跑同一套契约测试。
- 方案：
  - 配置与依赖：`zvec==0.7.0` 锁版本进必装依赖（上游 0.x 迭代快，升级要单独提交并重跑契约测试）；
    Chroma 暂时保留为必装依赖（兼容后端）。新增 `VECTOR_STORE`（默认 `zvec`）与 `ZVEC_PATH`，
    `CHROMA_*` 标注为「仅 `VECTOR_STORE=chroma` 时生效」；Dockerfile 的 `/data` 卷同步加 `ZVEC_PATH`。
  - 拆包 `services/vector_store/`：`base.py`（契约 + 相关度换算 + 分块元数据白名单 + MMR 常量）、
    `zvec_store.py`、`chroma_store.py`（原实现迁移）、`__init__.py`（`build_vector_store` 工厂 +
    `vector_service` 单例）。业务层 import 路径不变，换后端不改调用点。
  - 契约方法：`add_documents / search / delete_kb / delete_document / count`，并把两个诊断方法
    （`count_chunks_by_filename` / `list_doc_ids`）纳入契约；语义（相关度 `1 - distance`、MMR 无分数、
    阈值过滤计数、元数据白名单、写入幂等边界）写在 `docs/architecture.md` 3.2。
  - zvec 适配器按 zvec 0.7.0 的**实测行为**实现（先用探针脚本跑通再写代码）：schema 在首次写入时按向量维度
    懒建；分块 id 用 `kb-doc-chunk` 且统一 `upsert`（`insert` 撞 id 只返回错误码）→ 重跑入库幂等；
    正文显式存 `content` 字段（zvec 不保存原文）；MMR 由适配器自实现；`delete_document` 先数后
    `delete_by_filter`；`delete_kb` 用 `destroy()`；写操作只在返回值里报错，统一 `_ensure_ok` 显性化失败。
  - embedding 身份校验迁到 `services/embedding.py::ensure_embedding_matches`（向量空间一致性属于 embedding
    身份，不是某个向量库后端的属性）；`DocumentService.delete_document`、`api/kb.delete_kb`、
    `api/document.delete_doc` 随之改为 `async`。
  - 迁移方式：不做原地格式转换，新建 zvec 库 + 重跑建库（`scripts/reindex_kb.py <kb_id>`）。
- 效果：
  - `uv run pytest -q` → **139 passed, 5 deselected**（Phase 3 为 128）：新增 `tests/test_vector_store.py` 用
    `store` fixture 参数化跑 zvec / chroma 的同一份契约，另补 zvec 独有的幂等失效场景；`tests/conftest.py`
    固定 `VECTOR_STORE=zvec` 与 `ZVEC_PATH`，测试不受本机 `.env` 影响。
  - `uv run mypy` → Success: no issues found in **45 source files**；`uv run ruff check .` 全绿。
  - 默认 `VECTOR_STORE=zvec` 跑通「上传 → 检索 → 问答」全链路（`tests/test_api.py` 的 chat / SSE 用例）；
    `uv run python -m benchmark.run_bench --mode fixtures` 两后端指标**完全一致**
    （Recall@8 50.0% / MRR 0.500 / 页命中率 43.8%），zvec 更快（检索 p50 1.8ms vs chroma 3.5ms，
    p95 3.2ms vs 7.3ms）。
  - `./scripts/gates.sh g1` 全绿（ruff / mypy 45 文件 / 139 离线用例 / 迁移 upgrade→check→downgrade→upgrade /
    冒烟探活 + openapi / changelog 与密钥自检）。
  - 已知限制（已记入风险登记簿）：内嵌 zvec 的写锁按 collection 目录独占、跨进程互斥 → 必须**单进程部署**，
    不要 `uvicorn --workers`（README 部署章节已写明）；分块元数据收窄为白名单（解析器附带的
    `source` / `sheet` / `ocr` 不再进向量库），旧 Chroma 库需重建以对齐 schema。
- 涉及提交：本次提交（配置与依赖、拆包与适配器、调用点改名、契约测试、文档与 README/.env.example 同步）

---

## [Phase 3] 2026-09-28 — 容器镜像构建：本机验证依赖安装与启动链路（真实 Dockerfile 仍待 CI）

- 类型：文档
- 目的：Phase 3 遗留的最后一项「未验证」是 Docker 镜像构建（本机无 docker socket 权限）。
  全推给 Phase 10 的 CI，意味着 Dockerfile / `uv.lock` 的问题要到很晚才暴露；
  本阶段先回答一个更小的问题：**镜像里能不能装出可用环境**。
- 方案：
  - 用 conda 装 podman 5.8.3 到独立环境（`~/anaconda3/envs/podman`，不动 base，`conda env remove -n podman` 可回退）；
    以 vfs 存储（本机无 `fuse-overlayfs`）+ 用户级 `~/.config/containers/{storage.conf,registries.conf,policy.json}`
    跑 rootless 构建。
  - 实测结论：本机**无法完整构建仓库 Dockerfile**，原因在环境而非 Dockerfile——
    rootless 需要 setuid root 的 `newuidmap`/`newgidmap`（Debian `uidmap` 包），只有管理员能装；
    退到 podman 的「单 ID 映射」兜底（`USER` 指向在 `/etc/subuid` 无条目的用户名）后只映射容器 uid 0，
    于是第 5 步必然失败：apt 要降权到 `_apt`（uid 42）、`useradd --uid 10001 app` 与 `chown -R app:app` 都不可映射。
  - 因此改用一份**仅本地使用、不提交**的变体 Dockerfile：只替换上述三处环境不可行点（apt 关闭沙箱降权、去掉
    `useradd` 与 `chown`），其余步骤（基础镜像、uv 安装、`uv sync --frozen` 两层、迁移与启动）与仓库版本逐行一致。
- 效果：
  - `podman build` 成功产出镜像（`Successfully tagged localhost/inner-rag:verify`）；
    依赖层 `uv sync --frozen --no-install-project --no-dev` 在 CPython 3.13.15 上按 `uv.lock` **冻结安装 129 个包**
    （无版本漂移、无解析失败），项目层装上 `inner-rag==0.3.0 (from file:///app)`。
  - 镜像可实跑：`alembic upgrade head` 执行 0001→0002 → uvicorn 启动 →
    `curl http://127.0.0.1:18010/api/system/health` 返回 **HTTP 200**
    （`degraded` 仅因本机未运行 Ollama，响应里 `llm.error` 明确指出连不上 11434，符合预期）。
  - 基础镜像可达性：`docker.io/library/python:3.13-slim` 与 `ghcr.io/astral-sh/uv:latest` 均能经本机代理拉取；
    `apt-get install git libgomp1` 在容器内成功执行。
  - 仍未验证（写入风险登记簿）：仓库 Dockerfile 的 `apt-get`（默认 apt 沙箱降权）、`useradd --uid 10001 app`、
    `chown -R app:app /data /app` 与 `USER app` 的运行期权限行为——这三行加非 root 运行需要真 Docker 或
    Phase 10 的 CI 逐字节验证。
  - 环境回收：删除 11G 容器存储（vfs 每层全量拷贝），保留 conda podman 环境与配置以便后续复用。
- 涉及提交：本次提交（CHANGELOG、风险登记簿、README 部署说明）

---

## [Phase 3] 2026-09-28 — 前端构建验证：本机补齐 Node 工具链并跑通 npm run build

- 类型：优化
- 目的：Phase 3 的前端改动（登录页 / 路由守卫 / 按权限渲染 / SSE 带 Token）当时只做了静态检查，
  提交说明里写明「本机无 Node/npm，构建未验证」——这是本阶段唯一没有任何门禁覆盖的交付物，
  留着就是把风险推给 CI。
- 方案：在无 sudo 的前提下装 Node：官方 tarball（Node LTS v24.21.0 linux-x64）解到 `~/.local/node`
  并加入 PATH，不写 `/usr/local`、不动 conda base，不想要了 `rm -rf ~/.local/node` 即可回退；
  然后在 `frontend/` 跑 `npm install`（走本机代理，139 个包）+ `npm run build`。
  顺手同步 `package-lock.json` 里过期的 name/version（rag-frontend@1.0.0 →
  inner-rag-frontend@0.2.0）——这个不一致会让 `npm ci` 直接失败。
- 效果：`npm run build` 成功，**94 modules transformed**，2.72s 产出 `dist/`；
  登录页与各视图 chunk 全部生成（LoginView 2.67 kB / KbList 7.76 / KbDetail 7.87 /
  DocList 11.28 / ChatView 51.02 / index 153.10 kB，gzip 后 59.86 kB）。
  至此 Phase 3 的 DoD 里不再有「未验证」项；Docker 镜像构建仍属未验证（无 docker socket 权限），
  风险登记簿已同步收窄。
- 涉及提交：c103bb8（锁文件 name/version）、本次提交（CHANGELOG 与风险登记簿）

---

## [Phase 3] 2026-09-28 — 身份与访问控制平面：本地登录 + 知识库级 ACL

- 类型：新增功能
- 目的：在此之前系统完全匿名开放——任何人都能列库、删库、问任何库，知识库级越权读
  （尤其是会话 ID 只校验存在、不校验归属）在实际使用中是真实漏洞。企业场景里
  「谁能看哪个库」是刚需，而且它会改数据模型与检索链路，必须早期做（见
  `docs/DEVELOPMENT_PLAN.md` ADR）；拖到后面做，前面所有接口都要返工。
- 方案：
  - 认证：`users` 表 + **argon2id** 口令哈希（`argon2-cffi`，自带随机盐）；`POST /api/auth/login`
    换 **HS256 JWT**（载荷只有 `sub`/`iat`/`exp`，**不装权限快照**，所以改权限立即生效）；
    `GET /api/auth/me` 供前端刷新恢复；账号由 `scripts/create_user.py` 发放，**不做注册接口**。
  - 身份注入：`core/context.py` 的 ContextVar + 纯 ASGI 中间件把 `user_id` 注入请求上下文，
    日志格式统一带 `user=`；middleware 用纯 ASGI 是为了能整体包在 CORS 外层。
  - 授权：`knowledge_bases.owner_id`（NOT NULL + FK RESTRICT + 索引，迁移 `0002` 回填历史库到
    一个不可登录的 bootstrap 账号）+ `kb_members`；三级 `read` / `write` / `owner`，策略写在
    `core/access.py`、HTTP 映射集中在 `api/deps.py`（core 不抛 HTTPException，策略可被脚本复用）。
  - 边界：所有涉及 kb 的路由先过 `ensure_kb_access`（**判定在服务层之前**，不做「检索后再过滤」）；
    未登录 / 过期 / 篡改 / 停用 → 401，越权 → 403，不存在或属于别的库 → 404；
    `/api/system/health` 与登录是唯一免鉴权白名单。
  - 加固：`DEBUG=false` 时用默认或 <32 字节签名密钥**拒绝启动**；`ENABLE_DOCS=false` 时
    不暴露 `/docs` `/redoc` `/openapi.json`；登录失败文案统一，不泄露账号是否存在。
  - 前端：登录页 + 默认私有的路由守卫 + 按 `my_permission` 渲染；SSE 走原生 fetch，
    补上 Bearer 头与 401/403/404 映射（不在 axios 拦截器作用域内，不补就是匿名请求）。
  - 明确不做：多租户、部门隔离、文档级权限、SSO/LDAP、审计、注册接口、logout 接口。
- 效果：
  - `tests/test_auth.py` 覆盖越权与放行两侧：11 条受保护路由未登录全 401、跨库全 403、
    `/health` 免鉴权且 200、伪造 Token 五类（过期 / 错密钥 / 篡改载荷 / alg=none / 缺 exp）、
    停用账号后已签发 Token 立即失效、哈希加盐与 `"!"` 哨兵、弱密钥拒绝启动、列表隔离、
    跨库会话 ID 回归（先红后绿）、只读 / 可写成员行为、成员管理仅 owner。
  - `uv run pytest -q` → **128 passed, 5 deselected**（Phase 2.1 为 89）；
    `./scripts/gates.sh g1` 全绿（ruff / mypy 42 文件 / pytest / Alembic upgrade→check→downgrade→upgrade /
    临时端口冒烟 + `/health` + `/openapi.json` / changelog 与密钥自检）。
  - 本机开发库已迁移并建号：`admin`/`admin` 与 `p1`/`123456` 均可登录取 token 并访问 `/api/kb`（200），
    匿名访问同接口 401、口令错误 401「用户名或密码错误」。
  - 已知未验证：前端构建（本机无 Node/npm），仅做静态检查与契约人工核对；已记入风险登记簿。
- 涉及提交：d91077a（模型与 0002 迁移）、7ff082f（口令哈希 / JWT / 身份上下文 / 配置）、
  10b3c7b（ACL 判定与鉴权依赖）、372e9fc（登录接口 + 路由接入 ACL + 跨库会话修复）、
  a1603e9（建号脚本）、3ce0415（前端登录与按权限渲染）、4a2f2f7（鉴权 / ACL 用例）、
  ccc701e（架构与计划文档同步 + README 精简）

---

## [Phase 3] 2026-09-28 — README 精简：删掉互相复述的内容，保留可执行路径

- 类型：文档
- 目的：README 长到 541 行且大量内容互相复述（「设计目标」与「特性」重复、常见问题与配置表
  重复解释同一件事、特性列表 11 条平铺）。结果是需要找「怎么建号 / 怎么切 provider」的人
  反而扫不到重点，而文档越长越容易与代码脱节。
- 方案：按「同一件事只说一次、能指向单一出处就指向它」删：删掉「设计目标」（已在开头与特性里）；
  特性 11 条合并为 6 条（文档与检索 / 流式问答与会话 / 身份与访问控制 / 模型可插拔 /
  性能与成本 / 可观测与工程化）；常见问题 15 条删到 7 条（保留 degraded、入库 failed、
  空召回、换 embedding 后乱、401、403、账号相关）；配置表 24 行压到 14 行，其余项指向 `.env.example`
  （那里有逐项注释）；前置条件与基准测试段落去冗。
- 效果：541 → 499 行，且 AGENTS.md §1.2 要求同步的内容（配置表、API 一览、认证与 ACL 说明、
  目录结构、建号与 provider 步骤）全部保留。验证：逐条对照 §1.2 清单人工核对；
  grep 确认无指向已删章节的交叉引用。
- 涉及提交：ccc701e

---

## [Phase 2.2] 2026-09-28 — 修复：fixtures 自检不再落盘（G0 不再脏化工作区）

- 类型：修复
- 目的：`scripts/gates.sh g0` 一加进来就暴露了旧行为：`--mode fixtures` 会把结果写到
  `benchmark/results/`，且这个文件**已被提交进仓库**。后果是每次跑门禁都会把工作区弄脏
  （`git status` 出现修改），而里面只有 mock 分数与每次不同的延迟——既不是成绩，也不是证据。
- 方案：与「fixtures 拒绝写 README」保持同一口径——**fixtures 模式不再落盘**，只打印终端结果；
  只有 `--mode kb` 才写 `benchmark/results/*.json`；同时删除那个被跟踪的自检 JSON（`results/` 只留 `.gitkeep`）；
  同步更新 `benchmark/README.md`、`docs/evaluation.md`、`docs/testing.md`、`docs/DEVELOPMENT_PLAN.md`、README。
- 效果：`uv run python -m benchmark.run_bench --mode fixtures` 运行后 `git status --porcelain` 无新增/修改
  的 results 文件；`benchmark/results/` 只剩 `.gitkeep`。验证命令：
  `uv run python -m benchmark.run_bench --mode fixtures && git status --porcelain`。
- 涉及提交：fe65019

---

## [Phase 2.2] 2026-09-28 — 新增 scripts/gates.sh，把 G0/G1/G2 变成可执行门禁

- 类型：新增功能
- 目的：`docs/DEVELOPMENT_PLAN.md` §5 的门禁一直只是文档里的命令块，靠人脑记、手抄、可能漏步；
  尤其「push 前必须更新 CHANGELOG.md」是行为纪律，不落到脚本就拦不住。
- 方案：新增 `scripts/gates.sh`，支持 `g0|g1|g2|all`：
  - G0：`ruff check` + `ruff format --check` + `benchmark --mode fixtures`（离线自检，不写 README）；
  - G1：G0 + `mypy` + `pytest -q` + Alembic 迁移自检（临时干净库 `upgrade→check→downgrade→upgrade`，
    不依赖本机 `./data`）+ 临时端口 8011 冒烟探活 + **changelog 检查** + 密钥/敏感文件自检；
  - G2：`pytest -m live` 联网验收，并打印无法脚本化的手动步骤。
- 效果：`./scripts/gates.sh g0` 已实测通过（ruff 全绿、63 文件格式正常、fixtures 自检正常输出指标）；
  脚本通过 `bash -n` 语法检查。
- 涉及提交：c93034f

---

## [Phase 2.2] 2026-09-28 — 测试策略改为「覆盖准确 > 数量」

- 类型：文档
- 目的：原 `docs/testing.md` 把用例数量当基线（「用例只增不减」「数量是基线数字」），容易引导出
  为了数字而堆用例、重复测试同一行为、为不可能的分支写测试的反模式；真实（`-m live`）用例也需要
  明确「按需、只盖关键路径」的边界，而不是追求看起来很全。
- 方案：把分层表从「范围 + 数量级」改为「覆盖的失效场景」；核心原则明确为三条（离线可复现、
  覆盖准确 > 数量、禁止重复）；新增编写规范（一个用例一个失效场景、先查重再用 `parametrize`、
  不写空断言）；L3 联网用例明确只盖关键路径与缺 Key 自动 skip；删除「用例只增不减/数量是基线」两条；
  同步修正阶段编号引用（evaluation Phase 6、contracts Phase 7、tracing Phase 5、VLM Phase 9、CI Phase 10）。
- 效果：测试策略与 `AGENTS.md` §4 一致，可直接据此判定「这个用例该不该存在」。
  验证：与 `AGENTS.md` §4、`docs/DEVELOPMENT_PLAN.md` §5 门禁逐条对照无矛盾。
- 涉及提交：9eca4ec

---

## [Phase 2.2] 2026-09-28 — 架构文档补充「关系库 vs 向量库」与身份/权限契约

- 类型：文档
- 目的：两个存储（SQLite/PostgreSQL 与 zvec）在文档里只有零散的插件点描述，新人（或面试官）
  看不出「为什么要两个库、各自存什么、不一致时以谁为准」；同时新增的 Phase 3 权限平面还没有契约，
  实现者没有可依据的边界。
- 方案：在 `docs/architecture.md` 新增 §4「关系库 vs 向量库」对比表 + 对账口径（`count(kb_id)`
  必须等于 `completed` 文档分块数之和；向量是可重建的派生物，重建而非修补），新增 §3.8 身份与访问控制
  契约（密码强哈希、`user_id` 进 contextvar、kb 级 ACL、401/403 语义、权限判定在服务层之前）；
  插件点总表新增身份/权限行，错误契约新增 401/403 两行；同步修正全文阶段编号（Phase 4/5/6/7/9）。
- 效果：存储分工与一致性边界有明确文档；权限实现的边界（做什么、不做什么）可直接引用。
  验证：章节编号重排后修正 `docs/observability.md` 对「第 6 节错误契约」的引用；人工核对链接有效。
- 涉及提交：7283026

---

## [Phase 2.2] 2026-09-28 — 重排阶段路线：前置「身份与访问控制」与「向量库统一 zvec」

- 类型：文档
- 目的：原计划把向量库迁移放在中后段、且完全没有身份/权限阶段，会导致两处「后期重构债」：
  「先在 Chroma 上写新逻辑、以后再在 zvec 上重验」的重复成本，以及权限能力缺失下先做检索调优
  （调优结果在带权限边界后可能失效）。因此把这两类**会反向影响其它阶段**的基础设施前置。
- 方案：
  - 新增 **Phase 3 身份与访问控制平面**（单租户 + 本地账号登录 + 知识库级 ACL；明确不做多租户 / SSO / 审计）；
  - 新增 **Phase 4 向量库统一到 zvec**（抽 `VectorStore` 接口 + zvec 适配器，Chroma 降为兼容实现）；
  - 原 Phase 3–8 顺延为 **Phase 5–10**（可观测 / 评测 / 可插拔深化 / 质量提升 / OCR·VLM / 交付），
    并同步更新门禁、里程碑（新增 M0 权限、M1 向量库统一）、风险登记簿、ADR 与全部交叉引用；
  - 新增两条不变量（生产级但不做过度设计、变更留痕）与范围说明（开发期可用云端 API、不做真实企业合规）。
- 效果：`docs/DEVELOPMENT_PLAN.md` 阶段编号与 DoD 覆盖 Phase 3–10；全仓库 `Phase N` 引用已对齐
  （`grep -rn "Phase [0-9]" docs/ README.md` 无遗漏）。验证：G0（ruff）与文档链接人工核对。
- 涉及提交：b60a9ac

---

## [Phase 2.2] 2026-09-28 — 建立变更纪律（AGENTS.md）与 Changelog 制度

- 类型：文档
- 目的：随着进入 Phase 3 之后的开发，改动会越来越多；需要一个**强制机制**保证每次改动的
  「目的 / 方案 / 效果」都被记录，避免只有代码没有动机、面试时讲不清每一阶段在做什么。
  同时把"不做无效防御、注释规范、测试重覆盖而非数量、禁止累积重构债、生产级标准"等纪律固化成
  Agent 可自动读取的约束，而不是散落在对话里。
- 方案：
  - 新增仓库根 `AGENTS.md`（Warp 项目规则文件），把纪律写成可执行条款，含 push 前自检清单；
  - 新增 `CHANGELOG.md`（本文件），定义统一字段与"最新在上"的阅读方式；
  - changelog 更新作为 **push 前置门禁**：push 前用 `git diff --name-only origin/main...HEAD`
    逐项核对是否已记录。
  - 权衡：未把"阶段重排（zvec / 权限平面前置）"一并写入 `docs/DEVELOPMENT_PLAN.md`——
    该改动影响阶段顺序，另行确认后单独提交，避免一次改动混入设计与流程两类内容。
- 效果：`AGENTS.md` 共 102 行，覆盖 7 条纪律（变更纪律 / 禁止无效防御 / 注释命名 / 测试 /
  禁止重构债 / 提交安全 / push 自检）；`CHANGELOG.md` 建立并回溯记录 Phase 0–2.1。
  验证：`wc -l AGENTS.md`；后续每次 push 通过 `AGENTS.md §7` 清单自检。
- 涉及提交：8d00977

---

## 历史回溯（Phase 0 – 2.1）

> 以下条目依据 git 历史与 `docs/DEVELOPMENT_PLAN.md` §2「现状基线」整理，
> 目的是让面试时能按阶段讲清演进脉络。**目的 / 效果为归纳**，细节以对应文档与提交为准。

## [Phase 0] — 建立重构基线

- 类型：重构
- 目的：原项目是一个能跑但结构松散的 RAG demo，需要一个可控的重构起点。
- 方案：导入 legacy rag 作为基线；引入 `uv` 管理依赖并锁定（`uv.lock`），固定 Python 版本。
- 效果：依赖可复现安装；后续所有阶段都建立在锁定依赖之上。

## [Phase 1] — 工程化骨架

- 类型：重构
- 目的：把目录结构、迁移、部署与测试补成"工程"而不是"脚本"。
- 方案：后端迁到 `src/inner_rag/` 并升级到 LangChain 1.x；用 **Alembic** 迁移替代运行时
  `create_all`；新增 Dockerfile 与 docker compose；补齐离线 pytest 用例与运维脚本；
  修正前端 API 基址与相关度展示；README 重写。
- 效果：表结构可迁移、可回退；有了离线可复现的测试基础；容器化资产就位。

## [Phase 1.5] — 开发默认零依赖

- 类型：重构
- 目的：降低本地开发门槛，并让密钥管理有统一出口。
- 方案：开发默认库从 PostgreSQL 换成 **SQLite**（单文件、零外部依赖）；
  密钥统一走 `.env`（gitignore），代码只从 `core/config.py` 读取。
- 效果：`uv sync` + 一次迁移即可跑通；密钥不再散落在源码/命令里。

## [Phase 2 / 2.1] — 模型后端可插拔

- 类型：新增功能
- 目的：把"绑死某一家模型"变成"改两个环境变量就能换后端"，这是可插拔目标的起点。
- 方案：新增 `providers/` 抽象层（`specs` 元数据 + `chat` / `embeddings` 构造 + `factory` 门面），
  Chat 与 Embedding 各自独立选型；DeepSeek 改用官方集成；错误统一为 `ProviderError` → 可读的 503；
  补齐离线 provider 用例与 `-m live` 联网验收套件；README 改写为产品视角。
- 效果：`ollama / openrouter / deepseek / openai / mock` 可自由组合，业务代码与接口不变；
  配置错误能给出"改哪个变量"的明确提示。

## [文档与评测基建] — 可评估的前置准备

- 类型：新增功能
- 目的：让"答得准不准"可量化、可回归，而不是靠感觉。
- 方案：新增 `benchmark/` 指标脚本与龙族真实评测集（`docs/datasets/dragon_king/`），
  支持离线 fixture 与真实知识库两种模式，结果写入 README 基准表；重建开发文档体系
  （`docs/DEVELOPMENT_PLAN.md` / `architecture.md` / `observability.md` / `evaluation.md` / `testing.md`），
  向量库选型明确为 zvec。
- 效果：提交前可离线自检指标算法（`--mode fixtures`）；README 有基准表与阶段路线图；
  后续阶段（可观测 / 评测 / 可插拔深化 / 质量提升 / 交付）有了明确的 DoD 与门禁。
