# 评测报告 2026-09-30 · small/hybrid/k=8/th=0.3

- 生成时间：2026-09-30T14:42:15
- git commit：`bb88b83` / app 0.3.0
- 配置：`deepseek:deepseek-flash` + `openrouter:liquid/lfm-2.5-embedding-350m:free`，strategy=hybrid k=8 threshold=0.3
- 分块：size=1000 overlap=200 embedding_max_input_chars=400
- judge：`deepseek:deepseek-flash` / prompt `v1`
- 估算成本：0.0 USD（**未配置价格表**，token 用量见下表，成本留待 Phase 10）

## 1. 指标

| 指标 | 数值 | 门禁（G3） |
| --- | --- | --- |
| recall_at_k | 75.0% | ≥ 0.8（Phase 8 目标） |
| citation_precision | 27.5% | ≥ 0.8（Phase 8 目标） |
| keyword_pass_rate | 75.0% | ≥ 0.8（Phase 8 目标） |
| refusal_accuracy | 100.0% | ≥ 0.9 |
| false_refusal_rate | 12.5% | 不得上升 |
| mrr | 68.8% | — |
| page_hit_rate | 62.5% | — |
| judge_accuracy | 71.4% | — |
| faithfulness | 100.0% | — |

## 2. 逐题明细

| id | 检索命中 | 引用精度 | 要点 | judge | 忠实度 | 归因 |
| --- | --- | --- | --- | --- | --- | --- |
| `sakura-who` | ✓ | 40% | 100% | ✓ | 100% | — |
| `nonno-real-name` | ✗ | 0% | 0% | ✗ | 100% | 阈值过严（全部召回被过滤） |
| `sakura-job` | ✓ | 20% | 100% | ✓ | 100% | — |
| `kasell-deans` | ✓ | 20% | 100% | ✓ | 100% | — |
| `erie-lingyan` | ✗ | 0% | 0% | ✗ | 100% | 检索失败（证据未召回） |
| `erie-call-lumingfei` | ✓ | 100% | 100% | ? | — | — |
| `kasell-principal` | ✓ | 20% | 100% | ✓ | 100% | — |
| `white-king-lingyan` | ✓ | 20% | 100% | ✓ | 100% | — |
| `negative-geography` | ✗ | 0% | 100% | ✓ | 100% | — |

## 3. 失败分析

- `nonno-real-name`：阈值过严（全部召回被过滤）
  - judge：模型未回答诺诺真名是陈墨瞳，只说未找到相关信息；该说法与空参考资料一致。
  - 召回页：[]
- `erie-lingyan`：检索失败（证据未召回）
  - judge：模型只描述言灵效果，未给出期望的言灵名称「审判」；效果描述可在参考资料中找到依据。
  - 召回页：[937, 6415, 8038, 8025]

## 4. 与上一次对比

没有可对比的历史报告（本次为基线）。

（题目共 9 条：正样本 8 / 负样本 1）
