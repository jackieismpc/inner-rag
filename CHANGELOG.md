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
