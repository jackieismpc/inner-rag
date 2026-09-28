# 龙族评测集（dragon_king）

真实语料评测用的数据集目录。语料是 `data/uploads/龙族.pdf`（11,165 页 / 2,362,491 字符），
**PDF 本身不入库**（体积 + 版权），本目录只提交评测集与极短的引用片段。

| 文件 | 说明 |
| --- | --- |
| `eval_v1.jsonl` | 评测集 v1，9 条（含 1 条负样本），一行一题 |
| `fixtures/*.txt` | 与锚点一一对应的短片段，每段 < 200 字，供离线（无 PDF、无网络）评测使用 |
| `fixtures/index.jsonl` | fixture 索引：`fixture_id → page → file → eval_ids → chars` |

评测规则、指标定义、门禁阈值见 `docs/evaluation.md`；测试分层见 `docs/testing.md`；
指标怎么跑、结果怎么写进 README 基准表见 `benchmark/README.md`。

## 页码口径

`citations[].page` 与 `fixtures/index.jsonl` 的 `page` 都是 **PyMuPDF 1-based 物理页索引**
（`doc[p - 1]`），不是书里的印刷页码——本 PDF 没有可靠的印刷页码对应关系。
对外展示的引用位置走分块元数据的 `page_start` / `page_end`（Phase 4 增补）。

## 校验

任何改动评测集或 fixture 的提交，都必须先过校验（缺 PDF 时自动跳过 PDF 校验、只做 schema 校验）：

```bash
# 跑基准脚本时会自动校验（缺 PDF 时只做 schema 校验并打印 [skip]）
uv run python -m benchmark.run_bench --mode fixtures

# 或者只跑校验相关的单测（本地有 PDF 时会逐条核对锚点）
uv run pytest -q tests/test_benchmark_metrics.py
```

校验内容：

1. schema（字段齐全、类型正确、`id` 唯一、负样本不要求引用与关键词）；
2. 每条 `citations[].quote` 必须能在 `page` 对应页原文里找到（容忍换行与空白差异）；
3. 每个 fixture 文件必须 < 200 字，且内容是对应页原文的子串。

校验存在的意义：**评测集自己写错**会把「模型答错」和「出题错」混在一起，是最难排查的失败。

## 版权说明

`fixtures/` 里的文本是为验证检索与引用能力而截取的**极短片段**（单段全部少于 200 字，
远小于任何合理引用范围），仅用于本地与 CI 的自动化评测，不构成对原作的替代，也不随本仓库再分发。
完整原著不得提交进仓库；需要跑小库/全库评测时，把 PDF 放在本地 `data/uploads/` 下（已 gitignore）。
