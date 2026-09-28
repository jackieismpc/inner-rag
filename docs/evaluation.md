# 评测体系：怎么量化「答得准不准」

对应 `docs/DEVELOPMENT_PLAN.md` 的 **Phase 4**（基线）与 **Phase 6**（提升）。
评测语料是真实中文长篇小说 `data/uploads/龙族.pdf`（11,165 页 / 2,362,491 字符），
评测集与离线短片段放在 `docs/datasets/dragon_king/`。

## 1. 为什么要评测

RAG 的失败只有两种，而且必须分开看：

1. **没找到**（检索失败）——正确内容不在召回结果里，任何模型都答不对；
2. **找到了但没答对**（生成失败）——上下文里有答案，模型没用好或没引用。

只跑「看起来对」的人工验收会同时掩盖这两种失败。所以评测分成**检索指标**与**回答指标**两组，
并额外记录**引用是否正确**（防止「答对了但依据是编的」）。

## 2. 语料与知识库

### 2.1 双库策略

| 库 | 规模 | 用途 | 能否入库仓库 |
| --- | --- | --- | --- |
| **小库**（`dragon_king_small`） | 目标 ≤ 1,200 页 / ≤ 250 分块 | 自动化评测 + CI 回归 + 日常调参 | 否（PDF 本地，构建脚本生成） |
| **全库**（`dragon_king_full`） | 全量 11,165 页 / 约 2,900–3,000 分块 | 里程碑人工验收、规模性能验证 | 否 |
| **离线 fixture** | 9 段、每段 <200 字 | 无 PDF / 无网络时跑通评测逻辑（CI） | **是**，随仓库提交 |

小库的页窗口由 `scripts/build_eval_kb.py --profile small` 生成，必须包含：

- 命中全部评测条目的锚点页（含前后各 ±2 页的上下文，避免分块边界切掉证据）；
- 一组与评测问题**无关**的章节（约 10–15% 篇幅），用于验证「阈值能挡住不相关内容」；
- 覆盖多个人物线（路明非 / 绘梨衣 / 恺撒 / 楚子航 / 诺诺），避免小库退化成单主题语料。

### 2.2 为什么不全用全库

- 全库建库成本与耗时高（约 2,900+ 分块 × 每块一次嵌入调用），不适合每次改动都跑；
- 全库的噪声更大，指标波动会掩盖小改动的收益，回归判断不稳定。

全库只用于：里程碑验收、规模相关的性能测试（检索延迟、建库耗时、成本）、以及「小库结论是否成立」的抽查。

### 2.3 构建脚本与 manifest

```bash
# 小库（自动评测用）
uv run scripts/build_eval_kb.py --profile small --name dragon_king_small

# 全库（里程碑验收用）
uv run scripts/build_eval_kb.py --profile full --name dragon_king_full
```

每次构建产出 manifest 到 `docs/reports/eval-kb-<日期>.json`，字段至少包含：

```json
{"kb_id": 7, "name": "dragon_king_small", "profile": "small",
 "source": "data/uploads/龙族.pdf", "source_sha256": "…",
 "page_windows": [[5894, 5920], […]],
 "pages": 1180, "chunks": 232, "chunk_size": 1000, "chunk_overlap": 200,
 "embedding_identity": "openrouter:liquid/lfm-2.5-embedding-350m:free",
 "embedding_max_input_chars": 400,
 "built_at": "2026-09-28T09:20:00Z", "duration_s": 412.5, "est_cost_usd": 0.0}
```

`embedding_identity` 与 `embedding_max_input_chars` 必须记录：换 embedding 或改截断长度会让指标不可比。

## 3. 评测集

### 3.1 文件与 schema

`docs/datasets/dragon_king/eval_v1.jsonl`，一行一题（JSON Lines，UTF-8，无 BOM）：

```json
{
  "id": "sakura-who",
  "question": "Sakura是谁？",
  "category": "alias",
  "expected_answer": "Sakura 是路明非在高天原当牛郎/服务生时用的花名，「Sakura·路」；也要注意区分上杉绘梨衣对他的称呼。",
  "answer_keywords": ["路明非", "花名"],
  "must_not_include": ["上杉绘梨衣"],
  "citations": [
    {"page": 5904, "quote": "你昏迷的时候花名已经定好了，Sakura，翻译成中文就是小樱花。"},
    {"page": 6355, "quote": "路……Sakura，我叫Sakura·路。"}
  ],
  "expect_refusal": false,
  "notes": "别名题：考 Embedding 能否把 Sakura 与路明非连起来"
}
```

字段说明：

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `id` | 是 | 全局唯一，稳定不变（报告与 LangSmith 用它对齐） |
| `question` | 是 | 用户会怎么问；同一事实可有多条不同问法 |
| `category` | 是 | `fact` / `alias` / `multihop` / `negative` / `refusal` |
| `expected_answer` | 是 | 期望答案要点（给 judge 用，不必逐字） |
| `answer_keywords` | 是 | 关键词命中用；**要求全部命中**才算要点覆盖 |
| `must_not_include` | 否 | 常见错误答案的关键词，出现即扣分 |
| `citations` | 是（负样本可为空） | 期望引用锚点：`page` 为 **PyMuPDF 1-based 物理页号**，`quote` 必须能在该页原文里找到 |
| `expect_refusal` | 是 | `true` 表示应当拒答（语料里没有答案） |
| `notes` | 否 | 出题意图，便于后续维护 |

**锚点校验**：校验逻辑在 `benchmark/dataset.py::validate_or_raise`，`python -m benchmark.run_bench` 启动时
自动调用（有 PDF 就逐条验证 `quote` 能在对应页命中，没 PDF 只做 schema 校验并打印 `[skip]`）；
`tests/test_benchmark_metrics.py::test_eval_anchors_match_source_pdf` 在本地有 PDF 时把同一件事跑成回归用例。
不通过就让评测直接失败——否则「评测集自己写错」会被当成模型答错，最误导人。
（Phase 4 计划里的 `scripts/validate_eval_set.py` 只是它的薄包装，不再重复实现校验逻辑。）

**页码口径**：`page` 是 PyMuPDF 的 1-based 物理页索引，不是印刷页码。
本 PDF 无印刷页码对应关系，故统一用物理页并在 README 中注明。（真正的引用展示走
`page_start`/`page_end` 元数据，见第 4.3 节。）

### 3.2 首发题目（v1，9 条）

| id | 问题 | 类别 | 期望要点 | 锚点页 |
| --- | --- | --- | --- | --- |
| `sakura-who` | Sakura是谁？ | alias | 路明非的花名 | 5904、6355 |
| `nonno-real-name` | 诺诺的真名是什么？ | alias | 陈墨瞳 | 135 |
| `sakura-job` | 路明非在高天原做什么？ | fact | 牛郎/服务生 | 6079 |
| `kasell-deans` | 狮心会会长和学生会会长分别是谁？ | fact | 楚子航 / 恺撒·加图索 | 331 |
| `erie-lingyan` | 上杉绘梨衣的言灵是什么？ | fact | 「审判」 | 5278 |
| `erie-call-lumingfei` | 绘梨衣怎么称呼路明非？ | alias | Sakura | 6355、6413 |
| `kasell-principal` | 卡塞尔学院的校长是谁？ | fact | 昂热 | 331 |
| `white-king-lingyan` | 白王的言灵是什么？ | fact | 神谕（唯一克制「皇帝」） | 383 |
| `negative-geography` | 巴黎在哪个国家？ | negative | 应拒答（与语料无关） | — |

（`must_not_include` 与逐条 `quote` 以 `eval_v1.jsonl` 为准。）

### 3.3 离线 fixture

`docs/datasets/dragon_king/fixtures/` 下有与锚点一一对应的短片段（每段 <200 字）与 `index.jsonl`
（`fixture_id → page → text`）。用途：

- 无 PDF、无网络时（CI）也能构造一个微型知识库，验证**指标计算逻辑**与**引用匹配逻辑**；
- fixture 是**短引用**，不构成原文替代（见 `README.md` 的版权说明）。

注意：fixture 只能验证「评测逻辑正确」，不能验证「检索效果好不好」——后者必须用真实小库，
报告里要写明本次评测用的是哪一层（见第 4 节）。

## 4. 指标

### 4.1 检索指标（不需要 LLM）

对每条 `id`，用其 `question` 跑检索，拿到 Top-k 分块（默认 `k = TOP_K = 8`）：

- **Recall@k**：`citations` 锚点对应页码落在任一召回分块时记命中。分块页码以 `page_start`/`page_end`
  区间与锚点页求交集判断（无区间元数据时退回单页 `page`），因此 4.3 的元数据改造是前提。
- **MRR**：第一条命中的分块的倒数排名。
- **Page Hit@k**：命中页数 / 期望页数（多锚点题能反映「证据是否被拆散」）。

### 4.2 回答指标

- **要点命中率（keyword coverage）**：`answer_keywords` 全部出现在回答中才算该题通过（也单独记命中比例）。
- **正确性（LLM-as-judge）**：judge 模型拿到 `question` + `expected_answer` + 模型回答，
  输出 `{correct: 0|1, reason: str}`。judge prompt 与模型版本要固定并记录在报告里。
- **忠实度（faithfulness）**：回答中的断言是否都能在被引用的上下文里找到依据（judge 或规则）。
- **引用精度（citation precision）**：回答引用的来源（文件名 + 页区间）与 `citations` 锚点的重合度。
- **拒答正确率**：`expect_refusal=true` 的题中，正确表达「文档中没有相关信息」的比例；
  反向也要看**误拒率**（该答却拒答），两者一起报。
- **延迟与成本**：单题检索耗时、生成耗时、prompt/completion token、估算费用。

### 4.3 与代码的接口约定

- 分块元数据必须含 `page_start` / `page_end`（Phase 4 增补，见 `docs/DEVELOPMENT_PLAN.md` 第 4 节）；
- 检索结果沿用现有契约：`(Document, relevance|None)` + 被阈值滤掉的条数；
- 回答与来源沿用现有 SSE/JSON 结构（`sources` 含 `index/filename/page/score/doc_id/content`），
  评测脚本只读这些结构，**不额外给后端加评测专用分支**。

### 4.4 已知陷阱

- **512 token 截断**：免费 embedding 上下文只有 512 token 且当前 `EMBEDDING_MAX_INPUT_CHARS=400`，
  `CHUNK_SIZE=1000` 会被截断——这是**已知的系统性损耗**，报告必须写明该配置，结论不可与其他配置混比。
- **MMR 无分数**：`strategy="mmr"` 返回 `score=None` 且不过阈值，评测时「Recall@k」可比，
  但「top_score」类指标不可比。
- **分块跨页**：1000 字符约 5 页，单页元数据粒度不足 → 必须用 `page_start/page_end` 判命中，
  否则会把「命中邻页」误判为失败。
- **阈值过滤**：`RETRIEVAL_SCORE_THRESHOLD=0.3` 是经验值；评测报告要同时给出「过滤前」与「过滤后」
  的 Recall，才能区分「没找到」与「被阈值挡了」。
- **judge 漂移**：judge 模型或 prompt 变了要重跑基线；报告记录 judge identity。
- **过拟合评测集**：调到小库指标爆表但全库没提升时，说明在对着答案调参；需要定期扩题。

## 5. 三级执行

| 级别 | 数据 | 网络 | 场景 | 命令 | 频率 |
| --- | --- | --- | --- | --- | --- |
| L1 离线 fixture | 仓库内短片段 | 不需要 | 评测逻辑正确性、CI 门禁 | `uv run python -m benchmark.run_bench --mode fixtures`（+ `pytest -q tests/test_benchmark_metrics.py`） | 每次提交 |
| L2 live 小库 | 本地 PDF 小库 | 需要 provider Key | 真实指标、回归对比、G3 门禁 | `uv run python -m benchmark.run_bench --mode kb --kb-id <小库> --answer --update-readme` | 每次阶段收尾 / 改动检索与 Prompt 时 |
| L3 全库人工 | 本地 PDF 全库 | 需要 | 里程碑验收、规模与成本 | 同上，`--kb-id <全库>`；Phase 4 起再用 `scripts/eval_answer.py` 出完整报告 | 里程碑 |

L3 不只看自动指标：抽 10 题人工核对引用页码是否真的能翻到该内容。

## 6. 报告与 LangSmith 回写

### 6.1 报告格式

`docs/reports/eval-<日期>-<config>.md`，至少包含：

1. **配置快照**：LLM/embedding identity、`CHUNK_SIZE`/`CHUNK_OVERLAP`、`TOP_K`/`RERANK_TOP_K`/
   `RETRIEVAL_SCORE_THRESHOLD`、`EMBEDDING_MAX_INPUT_CHARS`、app 版本与 git commit；
2. **检索指标表**：Recall@8 / MRR / Page Hit@8（过滤前后两列）；
3. **回答指标表**：要点命中率、judge 正确率、忠实度、引用精度、拒答正确率、误拒率；
4. **逐题明细**：`id / 检索命中(✓✗) / 引用命中 / 回答要点 / judge / 备注`；
5. **失败分析**：每题归因到「检索失败 / 生成失败 / 引用错误 / 阈值过严」；
6. **与上一次报告的对比**：指标差值与结论。

### 6.2 基线表（维护在本文件）

| 基线 | 日期 | 配置 | Recall@8 | 引用命中率 | 要点命中率 | 拒答正确率 | 报告 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-0 | 待 Phase 4 产出 | 默认配置 + 小库 | — | — | — | — | — |

（Phase 4 完成后填第一行；Phase 6 的每次提升追加新行，永不删旧行——历史数字是判断趋势的唯一依据。）

### 6.3 LangSmith 联动（可选）

- 评测集同步为 LangSmith dataset（`id` 作为 example 的外部键）；
- 每次评测是一次 experiment，逐题写入 feedback（`correctness` / `citation_precision` / `faithfulness`）；
- 好处：能在 UI 里并排对比两次实验的逐题差异，且 trace 与分数直接关联，定位到具体 span。
- 约束：LangSmith 关闭时必须能完整跑完评测（结果只写本地报告）——评测流程不许依赖跟踪后端。

### 6.4 已落地：`benchmark/` 脚本与 README 基准表

本文件定义的指标已有一份可运行的实现（`benchmark/`，用法见 `benchmark/README.md`）：

| 本文件的定义 | 代码位置 |
| --- | --- |
| 检索指标（Recall@k / MRR / 页命中率） | `benchmark/metrics.py`，逐题结果里的 `hit` / `rr` / `page_hit` |
| 回答指标（要点命中率 / 引用精度 / 拒答正确率） | `benchmark/metrics.py`，由 `--answer` 填充 |
| 评测集与 fixture 的 schema / 锚点校验 | `benchmark/dataset.py` |
| 报告（配置快照 + 逐题明细） | `benchmark/results/<日期>-<模式>-<配置>.json` |
| README 基准表 | `benchmark/report.py`，写入 `<!-- BEGIN BENCHMARK -->` 区间 |

三者是同一份数据的不同展示：

1. 每次跑 `--mode kb … --update-readme` 在 README 基准表新增/覆盖一行（对外，一眼看趋势）；
2. 同一行的完整版本（逐题明细 + 配置快照）落在 `benchmark/results/*.json`（回溯用）；
3. 里程碑节点把关键数字摘进第 6.2 节的基线表（长期档案，永不删旧行）。

留给 Phase 4 的部分：LLM-as-judge 正确性与忠实度、token/成本字段、`docs/reports/*.md` 报告、LangSmith
experiment 回写。`benchmark` 的 fixtures 模式**不允许**写 README——自检分数不是成绩。

## 7. 门禁（G3）

在小库上：

- **不得回归**：Recall@8、引用命中率、要点命中率任一下降 > **2pp** 视为失败，不允许合入；
- **拒答**：负样本正确拒答率 ≥ **90%**，且误拒率不得上升；
- **目标（Phase 6 收尾）**：Recall@8 ≥ **0.8**、引用命中率 ≥ **0.8**、拒答正确率 ≥ **0.9**；
- 任何调参提交必须附「改动前 / 改动后」两列数字与同一 config 快照。

## 8. 评测代码自身的测试

评测脚本也会写错，所以同样要测（这些用例已经存在：`tests/test_benchmark_metrics.py`，15 例）：

- **指标单测**：构造「已知答案 + 已知召回」的假数据，断言 Recall@k / MRR / 页命中率 / 引用精度 / 分位数算对；
- **schema 校验单测**：缺字段、重复 id、负样本带引用、非负样本无锚点时报错；
- **fixture 回归**：`--mode fixtures` 在短片段上跑通并输出稳定结果（不联网、不写 README）；
- **锚点回归**：本地有 PDF 时逐条校验引用片段与 fixture 都能在源文档命中（无 PDF 自动 skip）；
- **不打网络**：默认评测里所有模型调用必须走 mock；真实调用只在 L2/L3。
