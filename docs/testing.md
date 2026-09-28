# 测试策略

对应 `docs/DEVELOPMENT_PLAN.md` 第 5 节的门禁。核心原则：**默认离线、可复现、不花钱**。

## 1. 四层测试金字塔

| 层 | 范围 | 网络 | 命令 | 数量级 | 频率 |
| --- | --- | --- | --- | --- | --- |
| **L1 单元** | 纯函数与单模块：分块、距离→相关度、缓存失效、provider spec 解析、基准指标计算 | 否 | `uv run pytest tests/test_parser.py tests/test_cache.py tests/test_benchmark_metrics.py …` | 快（<1s/用例） | 每次改动 |
| **L2 离线集成** | FastAPI TestClient 全链路：建库 → 上传 → 入库（mock provider）→ 检索 → 问答 → 系统接口 | 否 | `uv run pytest -q` | 现有 74 例 | 每次提交 |
| **L3 真实联网** | 真 provider：health、一次对话、一次 embeddings、LangSmith 追踪 | 是 | `uv run pytest -m live -q` | 现有 5 例 | 阶段收尾 / 改 provider 时 |
| **L4 评测** | 龙族评测集上的检索与回答指标（fixture / 小库 / 全库） | fixture 否，小库/全库 是 | `uv run python -m benchmark.run_bench --mode fixtures`（离线）/ `--mode kb`（+ Phase 4 的 `scripts/eval_answer.py`） | 9 题起 | fixture 每次提交；小库见 `docs/evaluation.md` |

现状（Phase 0–2.1）：

- `pyproject.toml` 里 `addopts = "-m 'not live'"`，即**默认只跑离线用例**；
- `markers` 已注册 `live`；`live` 用例必须显式 `-m live` 才执行；
- `tests/conftest.py` 强制把 `LLM_PROVIDER` / `EMBEDDING_PROVIDER` 钉成 `mock` 并设置
  `EMBEDDING_MAX_INPUT_CHARS`，保证**不受开发者本机 `.env` 影响**；
- 测试库与向量库都用临时目录，不写 `./data`；跑完即清理；
- `benchmark/` 的指标与评测集校验另有 15 个离线用例（`tests/test_benchmark_metrics.py`），
  所以 `uv run pytest -q` 现在是 **89 例**；`--mode fixtures` 的评测自检不在 pytest 里，要单独跑（见第 6 节）。

## 2. 目录与命名

```
tests/
├── conftest.py              # 环境钉死 + 公共 fixture（临时库、client、假 provider）
├── test_api.py              # L2：kb / document / chat / system 接口
├── test_cache.py            # L1/L2：query 与 embedding 缓存语义与失效
├── test_parser.py           # L1/L2：解析与 OCR 后端行为
├── test_providers.py        # L1：spec 解析、错误文案、健康检查状态机
├── test_vector_store.py     # L1/L2：写入、检索策略、阈值、相关度换算
├── test_benchmark_metrics.py # L1/L4：基准指标算法、评测集 schema 与锚点校验
├── test_live_providers.py   # L3：真实联网（-m live）
└── contracts/               # Phase 5：插件点契约测试（参数化跑所有实现）
```

命名规范：

- 文件名 `test_<被测模块>.py`，用例名 `test_<行为>_<条件>`（例：`test_chat_stream_returns_sources`）；
- 断言失败信息要能自解释（带上关键数值），不要只写 `assert x == y`；
- 每个 bug fix 必须附一个**会失败的回归用例**（先红后绿）。

## 3. 离线集成测试的 fixture 约定

- **环境隔离**：`conftest.py` 的 `Settings` 覆盖必须在 import `inner_rag` 之前生效，否则
  `settings` 单例已经读走 `.env`；
- **临时资源**：SQLite 用临时文件、Chroma 用临时目录、上传目录用 `tmp_path`，
  绝不共用 `./data`；
- **假 provider**：`LLM_PROVIDER=mock` / `EMBEDDING_PROVIDER=mock`（确定性哈希词袋向量），
  断言只依赖「词面相似」这种可控特性；
- **假时钟 / 假网络**：需要 TTL 的缓存用例用 monkeypatch 时间，不用 `sleep`；
  任何 HTTP 调用（Phase 3 的 tracing 上报、Phase 7 的 VLM）都要能 monkeypatch；
- **严禁真实 Key**：测试进程里不许出现真实 Key；`live` 用例从环境变量读，缺失时 `skip`
  并给出「缺少 X_API_KEY」的明确原因。

## 4. L3 联网用例规范

- 一律 `pytest.mark.live`，并放在 `test_live_providers.py`；
- 每个 provider 一个 fixture，缺 Key 就 `pytest.skip("缺少 DEEPSEEK_API_KEY")`；
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
- 报告落 `benchmark/results/*.json`（每次运行都有）与 `docs/reports/`（Phase 4 起），
  里程碑数字更新 `docs/evaluation.md` 的基线表。

## 6. 门禁与命令速查

```bash
# 每次提交（G0）
uv run ruff check . && uv run ruff format --check .
uv run python -m benchmark.run_bench --mode fixtures   # 评测自检（离线，不写 README）

# 阶段收尾（G1）
uv run mypy
uv run pytest -q                    # 离线全量（含 benchmark 单测）
uv run alembic upgrade head && uv run alembic check

# 真实链路（G2）
uv run pytest -m live -q

# 评测（G3）
uv run python -m benchmark.run_bench --mode kb --kb-id <小库> --answer --update-readme
uv run scripts/eval_answer.py --kb-id <小库>   # Phase 4 起：judge 类指标与成本
```

补充规则：

- **用例只增不减**：删用例必须在提交说明里说明理由（通常意味着行为契约变了）；
- **用例数量是基线数字**：新增功能应带来新用例；README 里的用例数（当前 89 离线 + 5 live）
  变更时同步更新；
- **修 bug 先补用例**：没有回归用例的修复视为未完成；
- **不跳过离线用例**：`xfail` / `skip` 必须写明原因与解除条件。

## 7. CI 计划（Phase 8 落地）

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
