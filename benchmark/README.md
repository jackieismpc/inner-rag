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
| `results/` | 每次运行一份 JSON（同一「日期 + 配置」在 README 里覆盖为一行） |

单测在 `tests/test_benchmark_metrics.py`（离线；本地有 PDF 时还会校验评测集锚点）。

## 两种模式

| 模式 | 数据 | 网络 | 指标含义 | 能否写 README |
| --- | --- | --- | --- | --- |
| `fixtures` | 仓库内 10 段短 fixture + mock embedding | 否 | 只验证**脚本与指标算得对** | **否**（拒绝写入，避免把自检当成绩） |
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
```

`--label` 会作为 README 表格里的「配置」列；不传则按 `kb<id>/<embedding>/<strategy>/k=<k>` 自动生成。
其余参数：`--dataset`、`--fixtures-dir`、`--k`、`--out-dir`。

## 指标口径

| 指标 | 定义 | 越大越好 |
| --- | --- | --- |
| `Recall@k` | 期望锚点页出现在 Top-k 召回分块里的题目比例 | 是 |
| `MRR` | 第一条命中结果的倒数排名（逐题平均） | 是 |
| 页命中率 | 命中的期望页数 / 期望页总数（多锚点题反映证据是否被拆散） | 是 |
| 要点命中率 | `answer_keywords` 全部出现在回答中的题目比例（需 `--answer`） | 是 |
| 引用精度 | 引用到的期望页 / 引用的全部页（需 `--answer`） | 是 |
| 拒答正确率 | 负样本里正确表达「文档中没有相关信息」的比例（需 `--answer`） | 是 |
| 检索 p50 / p95 | 单题检索耗时（毫秒） | 否（越低越好） |
| 端到端 p50 | 单题「检索 + 生成」耗时（需 `--answer`） | 否 |

## 注意事项

- **fixtures 模式不是成绩**：mock embedding 只有词面相似度，分数没有质量含义，脚本会拒绝写 README。
- **kb 模式绕过 QueryCache**：直接调向量库，保证延迟与召回是真实值（否则第二次查询会假性变快）。
- **必须与建库时的 embedding 一致**：`kb.embedding_model` 与当前配置不一致时脚本直接报错并提示重建索引；
  换 embedding / 改 `CHUNK_SIZE` 后，指标不可与旧行直接比较。
- **PDF 不入库**：`data/uploads/龙族.pdf` 只放在本地，缺 PDF 时锚点校验自动跳过（只做 schema 校验）。
- **费用**：`--answer` 每题至少一次 LLM 调用，小库 9 题一次约几分钱；全库评测请只在里程碑时跑。
