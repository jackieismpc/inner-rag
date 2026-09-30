# 基准测试（benchmark）

这里的脚本用来回答一个具体问题：**这次改动到底让检索/回答变好了没有？**

每次提交前跑一遍，把「质量指标 + 延迟」写进 README 的基准表（`<!-- BEGIN BENCHMARK -->` 区间，
由脚本维护，不要手工编辑那一段）。指标定义与门禁阈值见 `docs/evaluation.md`，
测试分层见 `docs/testing.md`。

## 目录

| 文件 | 作用 |
| --- | --- |
| `run_bench.py` | 执行入口（CLI）：跑评测集 → 算指标 → 落盘 → 可选写回 README |
| `metrics.py` | 指标纯函数（不依赖 `inner_rag`，可单独单测） |
| `dataset.py` | 评测集 / fixture 读取与校验（schema + 引用锚点） |
| `report.py` | 结果落盘 `results/*.json` 与 README 表格维护 |
| `results/` | kb 模式每次运行一份 JSON（同一「日期 + 配置」在 README 里覆盖为一行）；fixtures 自检不落盘 |

单测在 `tests/test_benchmark_metrics.py`（离线；本地有 PDF 时还会校验评测集锚点）。

## 两种模式

| 模式 | 数据 | 网络 | 指标含义 | 写 README / 落盘 |
| --- | --- | --- | --- | --- |
| `fixtures` | 仓库内 135 段短 fixture + mock embedding | 否 | 只验证**脚本与指标算得对** | **都不**（自检不是成绩，也不落盘噪声文件） |
| `kb` | 本地 PDF 建好的真实知识库 | 需要 provider Key | 真实召回/引用/延迟（加 `--answer` 还有回答指标） | 是 |

## 常用命令

```bash
# 离线自检：不联网、不花钱，CI 与提交前都能跑
uv run python -m benchmark.run_bench --mode fixtures

# 真实知识库：检索指标 + 延迟，写入 README 基准表
uv run python -m benchmark.run_bench --mode kb --kb-id 3 --update-readme

# 加回答指标（要点命中率 / 引用精度 / 拒答正确率，会调 LLM 产生费用）
uv run python -m benchmark.run_bench --mode kb --kb-id 3 --answer --update-readme

# 调参对比：同一张表里多一行，便于看「改动前 / 改动后」
uv run python -m benchmark.run_bench --mode kb --kb-id 3 --strategy hybrid --threshold 0.25 --answer \
  --label "hybrid/k=8/th=0.25" --update-readme

# 阈值定标：一次未过滤检索反推整条阈值曲线（不花 LLM 的钱，只跑检索）
uv run python -m benchmark.run_bench --mode kb --kb-id 3 --threshold-sweep
uv run python -m benchmark.run_bench --mode kb --kb-id 3 --threshold-sweep 0,0.1,0.2,0.3

# 小库跑全量评测集：引用页不在库中的题默认被跳过（否则它们全按 0 分计，指标变成在量拒答）
uv run python -m benchmark.run_bench --mode kb --kb-id 3 --label "small/回归"
uv run python -m benchmark.run_bench --mode kb --kb-id 3 --absent-items refuse --answer --label "small/拒答"
uv run python -m benchmark.run_bench --mode kb --kb-id 3 --absent-items score   # 按未召回归零
```

`--label` 会作为 README 表格里的「配置」列；不传则按 `kb<id>/<embedding>/<strategy>/k=<k>` 自动生成。
**做实验务必显式传 `--label`**：行按「日期 + 配置」去重，配置相同就会覆盖上一行（对照行会丢）。
其余参数：`--dataset`、`--fixtures-dir`、`--k`、`--out-dir`、`--absent-items`。

`--absent-items` 决定「期望页不在这个库里」的题怎么处理（判据是库的**真实内容**——读分块元数据里的
`page`，而不是建库脚本的页窗口常量，后者会随窗口调整过期）：

| 取值 | 行为 | 什么时候用 |
| --- | --- | --- |
| `skip`（默认） | 跳过，不计入任何指标；终端与结果 JSON 分别报出「完全缺席」与「只进来部分引用页」两类 | 看检索质量。评测集照**全库**出题，丢给 227 页的小库时只有 10 条能完整命中，其余 125 条按 0 分计会同时压低 Recall 与页命中率，读起来像「检索变差了」 |
| `refuse` | **改判为拒答题**（`expect_refusal=True`），进 `refusal_accuracy` 的分子分母 | 小库量**拒答能力**。「库里查不到答案的问题」在小库上天然成立，而且这正是真实分布（用户问什么不可控）——把它们摘掉等于浪费了一批现成场景。配 `--answer` 用 |
| `score` | 照常参与，未召回记 0 | 跑全库时与 `skip` 等价（全库每题证据都在）；需要把它们算进「没答对」的分母时才用 |

任何模式下，**「只进来部分引用页」的题都跳过**：证据只到一半，recall 上限被人为压到 1/2、1/3，
模型据半份证据答对或答错都说明不了什么。

`--threshold-sweep` 的原理：阈值过滤发生在 Top-k **之后**（`finalize_results` 先滤后排），
所以一次 `threshold=0` 的召回就含全部信息——对任一阈值 t，丢掉 `score < t` 的条目再重算命中判定，
与真的按 t 检索完全一致，不必为每个候选阈值各跑一遍（省时也省钱）。曲线随结果 JSON 一起落盘
（`threshold_curve`），下次想换网格不必重跑检索。不带值用内置网格，也可以在参数后直接给逗号分隔的列表。

## 指标口径

| 指标 | 定义 | 越大越好 |
| --- | --- | --- |
| `Recall@k` | 期望锚点页出现在 Top-k 召回分块里的题目比例 | 是 |
| `MRR` | 第一条命中结果的倒数排名（逐题平均） | 是 |
| 页命中率 | 命中的期望页数 / 期望页总数（多锚点题反映证据是否被拆散） | 是 |
| 要点命中率 | `answer_keywords` 全部出现在回答中的题目比例（需 `--answer`） | 是 |
| **引用命中率** | 引用的来源里**至少一条**命中期望页的题目比例（需 `--answer`） | 是 |
| 引用精度 | 引用到的期望页 / 引用的全部页（需 `--answer`） | 是 |
| 拒答正确率 | 负样本里正确表达「文档中没有相关信息」的比例（需 `--answer`） | 是 |
| 检索 p50 / p95 | 单题检索耗时（毫秒） | 否（越低越好） |
| 端到端 p50 | 单题「检索 + 生成」耗时（需 `--answer`） | 否 |

**引用命中率与引用精度要一起看**：精度按「引用条数」算分母，一道题只引用了 1 条时显示 100%、
一条都没引用时显示 0%，分母随配置漂移，**不能跨配置比较**。命中率是 0/1 判定、分母恒为题数，
与 `Recall@k` 同口径。终端逐题表打的是命中率，正是为了避免被精度的两个极端误导。

## 注意事项

- **fixtures 模式不是成绩**：mock embedding 只有词面相似度，分数没有质量含义，脚本会拒绝写 README，
  也不落盘结果文件（延迟每次不同，落盘只会污染工作区）。
- **kb 模式绕过 QueryCache**：直接调向量库，保证延迟与召回是真实值（否则第二次查询会假性变快）。
- **必须与建库时的 embedding 一致**：`kb.embedding_model` 与当前配置不一致时脚本直接报错并提示重建索引；
  换 embedding / 改 `CHUNK_SIZE` 后，指标不可与旧行直接比较。
- **PDF 不入库**：`data/uploads/龙族.pdf` 只放在本地，缺 PDF 时锚点校验自动跳过（只做 schema 校验）。
- **费用**：`--answer` 每题至少一次 LLM 调用，用的是 `deepseek-flash`（按量计费、单价很低），
  整份评测集跑一轮的费用可以忽略；建库这一侧换成**本机权重**之后也不再按次计费，所以
  **全库 + 全量评测集是现在的默认评测方式**，不再是「只在里程碑才敢跑」。
  小库留着跑快速回归（几十秒出数）与拒答验证——评测集的锚点覆盖全书，丢给小库只有 9 条能命中，
  那 9 条之外的分数衡量的是拒答策略，不是检索质量（见 `docs/evaluation.md` 2.1）。
