# 评测体系：怎么量化「答得准不准」

对应 `docs/DEVELOPMENT_PLAN.md` 的 **Phase 6**（基线）与 **Phase 8**（提升）。
评测语料是真实中文长篇小说 `data/uploads/龙族.pdf`。**实测口径**（pypdf 提取，2026-09-30）：
**11,165 个物理页**里 **11,138 个非空页**、2,351,990 字符，**平均每页约 211 字符**——这个数字决定了后面几乎所有设计。
注意「物理页」与「非空页」是两回事：解析会跳过空页（封面/扉页/无文本页），所以物理页号是稀疏的
（最大页号 11165 > 非空段落数 11138），**full 建库窗口必须按最大页号切**，否则末尾「页号 > 段落数」
的正文页会漏掉（见 `scripts/build_eval_kb.py::_parse_meta` 的 `max_page`）。
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
| **全库**（`dragon_king_full`） | 全量 11,138 页 → 11,138 分块 | **主评测场地**：整份评测集逐条打分 | 否（PDF 本地，构建脚本生成） |
| **小库**（`dragon_king_small`） | 实测 227 页 / 227 分块 | 快速冒烟、CI 回归、**拒答能力**验证 | 否（同上） |
| **离线 fixture** | 每段 <200 字，一题一段 | 无 PDF / 无网络时跑通评测逻辑（CI） | **是**，随仓库提交 |

**主场地为什么是全库**（这与 Phase 6 的定位相反，是评测集扩容后倒过来的）：v3 的题目锚点覆盖
源书第 1262 – 10856 页，而小库的页窗口只固定包住 8 个锚点段——同一套题丢给小库，能真正被
检索到的只有 **9 条**，其余 86 条的正确答案根本不在库里。那样的分数衡量的是「拒答策略」，
而不是「检索好不好」。全库上每题都有正确答案可召回，指标才有代表性；成本上也不再是障碍：
向量侧换成**本机权重的 Qwen**之后，建库的约束从「云端配额 / 费用」变成了「一次 GPU 时间」，
全量重建从「里程碑才敢跑」变成「调参时可以重跑」。

小库仍然不可省，它承担另外两件全库做不了／做起来太慢的事：

- **日常回归的响应速度**：全库一次入库要几百批嵌入，改一行 Prompt 不该等它；
- **拒答能力只在小库上成立**：锚点落在小库窗口之外的题目，其答案确实不在库中，模型必须回答
  「资料中没有」——这正是 Phase 8.1 发现「词面召回贸然注入 context 会让拒答率归零」时用的判据。
  全库没有这个性质（每个问题都有答案），所以**拒答类指标只在全库之外的地方评**。
  这类题用 `--absent-items refuse` 自动改判成拒答题，直接进 `refusal_accuracy`，不必人工挑题：
  v3 的 135 条有引用页的题里，小库只装得下 10 条，剩下 125 条就是现成的拒答场景
  （而且是真实分布——用户问什么根本不可控）。

小库的页窗口由 `scripts/build_eval_kb.py --profile small` 生成（常量写在脚本里：`ANCHOR_WINDOWS`
与 `IRRELEVANT_WINDOWS`），必须包含：

- 命中全部评测条目的锚点页（含前后各 ±2 页的上下文）→ 8 段共 67 页；
- 一组与评测问题**无关**的连续章节 → 3 段共 160 页；
- 覆盖多个人物线（路明非 / 绘梨衣 / 恺撒 / 楚子航 / 诺诺）。

**无关章节占比为什么远高于原定 10–15%**：锚点窗口只有 67 页，按 15% 配噪声则全库不到 80 页，
检索几乎没有干扰，Recall 会虚高到不可用（真实场景是 67 / 11,138 ≈ 0.6%）。所以这里让噪声占
约 70%，换取「指标还能反映检索好坏」。这条偏离原计划，是实测后的主动调整。

**页码必须是源 PDF 的物理页号**：早期实现先把窗口页抽成子 PDF 再走上传链路，结果库里存的是
子 PDF 的局部页号（1..227）——引用「第 166 页」在源书里翻不到，评测锚点（5904 这类源页号）
也全部对不上，Recall 直接归零，而回答其实是对的。现在的做法是解析源 PDF 后**只筛选、不重编号**
（`build_eval_kb.select_pages`），页码与源文档始终一致。

### 2.2 两个库都要留

- **全库 = 主场地**：噪声更大、指标波动更明显，但那是真实分布（0.6% 相关页占比），且每题都有答案；
- **小库 = 回归与拒答**：跑得快，且天然构造出「答案不在库里」的场景。

跑法上的分工：

```bash
# 日常改检索 / 改分块 → 小库，几十秒出数（默认 --absent-items skip，只评库内证据完整的 10 条）
uv run --extra local-embed python -m benchmark.run_bench --mode kb --kb-id <小库>

# 小库量拒答能力：把「答案不在库中」的 125 条改判为拒答题，配 --answer 才会真的调模型
uv run --extra local-embed python -m benchmark.run_bench --mode kb --kb-id <小库> \
    --absent-items refuse --answer --label "small/拒答"

# 阶段收尾、要写进基线表 → 全库，整份评测集
uv run --extra local-embed python -m benchmark.run_bench --mode kb --kb-id <全库> --answer --update-readme
```

### 2.3 构建脚本与 manifest

```bash
# 小库（自动评测用）；用 -u 关掉输出缓冲，否则中断时的异常会被进度条缓冲掩盖
uv run python -u scripts/build_eval_kb.py --profile small --name dragon_king_small

# 全库（里程碑验收用）。中断了**直接重跑这一条**：默认续跑，不会重算已入库的分块
uv run python -u scripts/build_eval_kb.py --profile full --name dragon_king_full

# 只有确实要「删库重来」时才加 --rebuild（会丢掉全部已完成的进度）
uv run python -u scripts/build_eval_kb.py --profile full --name dragon_king_full --rebuild
```

每次构建产出 manifest 到 `docs/reports/eval-kb-<日期>.json`，字段至少包含：

```json
{"kb_id": 7, "name": "dragon_king_small", "profile": "small",
 "source": "data/uploads/龙族.pdf", "source_sha256": "…",
 "page_windows": [[5894, 5920], […]],
 "pages": 1180, "chunks": 232, "chunk_size": 1000, "chunk_overlap": 200,
 "embedding_identity": "sentence_transformers:Qwen/Qwen3-Embedding-0.6B",
 "embedding_max_input_chars": 0,
 "built_at": "2026-09-28T09:20:00Z", "duration_s": 412.5, "est_cost_usd": 0.0}
```

`embedding_identity` 与 `embedding_max_input_chars` 必须记录：换 embedding 或改截断长度会让指标不可比。
`chunks` 取的是 `vector_service.count(kb_id)`（向量库实际条数），**不是** `add_documents` 的返回值——
后者是「本次新写入」的数，续跑时会把已入库的部分少算掉。

### 2.4 答案保密（谁可以看到参考答案）

评测里最容易被忽视、也最致命的一条：**答题模型绝不能看到参考答案**。它一旦能看到，
「正确率」就不再是检索与生成质量的度量，而是抄写能力的度量——指标会漂亮得毫无意义，
并且这个错误不会报错、不会告警，只会在某天有人质疑时才暴露。

所以这里把「谁看得到什么」写成明确边界：

| 角色 | 看得到 | 看不到 |
| --- | --- | --- |
| **答题模型**（被测系统） | 用户问题 + 检索命中的分块正文 | `expected_answer` / `answer_keywords` / `must_not_include` / `citations` |
| **规则指标**（`benchmark/metrics.py`） | 生成之后的回答 + 上述参考答案字段 | ——（它本就用这些判据打分） |
| **judge 模型**（`scripts/eval_answer.py`，可选） | 问题 + 回答 + 检索上下文 + `expected_answer` | ——（判分必须有参考答案，见下） |
| **检索指标** | 召回页区间 + `citations` 里的锚点页 | 回答（不参与检索打分） |

工程上的落实方式有两层，缺一不可：

1. **唯一入口**：答题侧的输入收敛到 `benchmark/run_bench.py::generation_input(item)`，
   它只返回 `question`。调用方没有别的路径能拿参考答案字段去拼 prompt；
2. **在模型边界上断言**：`tests/test_eval_pipeline.py::test_answer_llm_never_sees_reference_answer`
   跑一次真实的 `run_bench.run_kb(--answer)`，**捕获真正发给模型的每一个 prompt**，
   断言参考答案 / 关键词 / 禁止词一个都没出现；同时断言语料正文**在** prompt 里
   （否则这条用例只是在证明「什么都没发」）。

判分边界要单独说明：**LLM-as-judge 必须看到参考答案**，否则它无法回答「这个回答对不对」。
这是独立的一次调用（`scripts/eval_answer.py`），输入是「问题 + 回答 + 上下文 + 参考答案」，
与被测系统的生成链路完全分离；`JUDGE_PROMPT_VERSION` 的变化也会让历史分数不可比，
所以报告里同时记录 judge 身份与 prompt 版本。

## 3. 评测集

### 3.1 文件与 schema

评测集是一行一题的 JSON Lines（UTF-8，无 BOM）。`benchmark/dataset.py::DEFAULT_DATASET` 指向
**当前使用的那一份**，历史版本保留在同一个目录下以便对比：

| 文件 | 条数 | 说明 |
| --- | --- | --- |
| `docs/datasets/dragon_king/eval_v1.jsonl` | 9 | 首发题（人工挑锚点，见 3.2） |
| `docs/datasets/dragon_king/eval_v2.jsonl` | 28 | v1 + 19 条全库新增题（锚点在小库窗口之外，见 3.2.1） |
| `docs/datasets/dragon_king/eval_v3.jsonl` | **当前，146** | 28 条人工题 + 118 条语料生成题（其中 11 条负样本），锚点覆盖源书 p9–p11063 |

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
（Phase 6 计划里的 `scripts/validate_eval_set.py` 只是它的薄包装，不再重复实现校验逻辑。）

**页码口径**：`page` 是 PyMuPDF 的 1-based 物理页索引，不是印刷页码。
本 PDF 无印刷页码对应关系，故统一用物理页并在 README 中注明。（真正的引用展示走
`page_start`/`page_end` 元数据，见第 4.3 节。）

### 3.2 首发题目（v1，9 条）

| id | 问题 | 类别 | 期望要点 | 锚点页 |
| --- | --- | --- | --- | --- |
| `sakura-who` | Sakura是谁？ | alias | 路明非的花名 | 5904、6072、6355 |
| `nonno-real-name` | 诺诺的真名是什么？ | alias | 陈墨瞳 | 123 |
| `sakura-job` | 路明非在高天原做什么？ | fact | 牛郎/服务生 | 6072、6079 |
| `kasell-deans` | 狮心会会长和学生会会长分别是谁？ | fact | 楚子航 / 恺撒·加图索 | 331 |
| `erie-lingyan` | 上杉绘梨衣的言灵是什么？ | fact | 「审判」 | 5278 |
| `erie-call-lumingfei` | 绘梨衣怎么称呼路明非？ | alias | Sakura | 6355、6413 |
| `kasell-principal` | 卡塞尔学院的校长是谁？ | fact | 昂热 | 118、331 |
| `white-king-lingyan` | 白王的言灵是什么？ | fact | 神谕（唯一克制「皇帝」） | 383 |
| `negative-geography` | 巴黎在哪个国家？ | negative | 应拒答（与语料无关） | — |

（`must_not_include` 与逐条 `quote` 以 `eval_v1.jsonl` 为准。）

### 3.2.1 扩展题目（v2，28 条 = v1 的 9 条 + 19 条新增）

`docs/datasets/dragon_king/eval_v2.jsonl`。加这一版的直接原因是**评测集本身会过拟合小库**：
v1 的 9 条题里，有 8 条的锚点恰好落在小库那 11 个页窗口内（见 2.1），而小库是**按这些题的锚点页
挑出来建的**——用它在自己的窗口里考自己，分数必然偏乐观，也没法回答「换了没被挑中的文档会怎样」。
v2 的 19 条新题锚点**全部落在小库窗口之外**，用来做两件 v1 做不到的事（见 4.6）。

| id | 问题 | 类别 | 锚点页 |
| --- | --- | --- | --- |
| `guderian-major-special` | 古德里安教授用哪两个学院来类比他们这个专业的特殊性？ | fact | 250 |
| `ballroom-staircase` | 舞会开始时，通向二楼的两条弧形楼梯上分别走下什么样的人？ | fact | 700 |
| `eight-leg-horse` | 楚子航在白色光芒中看到的那匹骏马有几条马腿？ | fact | 1500 |
| `dragon-blood-panel` | 关于龙血对人类血液的作用，哪个院系的主任首先表示认同？ | fact | 2800 |
| `nidhogg-ceiling` | 餐厅天顶画《诸神的黄昏》里那条末日的巨龙叫什么名字？ | fact | 4000 |
| `genchisheng-lining` | 源稚生风衣衬里上的那幅浮世绘描绘了什么？ | fact | 4200 |
| `alias-tianzhaoming` | 「天照命」这个称呼喊的是谁？ | alias | 4200 |
| `devil-vacation` | 路明非在深海呼唤小魔鬼为什么没有得到回应？ | multihop | 4900 |
| `alias-little-devil` | 路明非在心里反复呼唤的那个「小魔鬼」究竟是谁？ | alias | 4900 |
| `pompeii-skydive` | 庞贝最近把哪项运动当成了自己的拿手项目？ | fact | 5200 |
| `kasell-ask-japanese` | 恺撒审问的那个猴脸男人只会说什么语言？ | fact | 5600 |
| `jiude-maoyi-prada` | 酒德麻衣从墙上摘下的那套职业装是什么品牌？ | fact | 6800 |
| `zhutoujing-holster` | 店长座头鲸揭开西装给路明非看自己贴身的什么东西？ | fact | 8600 |
| `fenghuang-fire-dragon` | 「炎之龙斩者」是谁的称号？ | alias | 9500 |
| `anniversary-escort` | 校庆时校长率领谁迎出校门去迎接路明非？ | fact | 9900 |
| `odin-mirror-horse` | 从镜中策马踏出的「奥丁」骑的是什么马？ | fact | 10200 |
| `nonno-monkey-dream` | 诺诺梦里被丢在荒野里的那只「傻猴子」在做什么？ | fact | 10600 |
| `negative-quantum` | 请写出薛定谔方程的完整推导过程。 | negative | — |
| `negative-finance` | 路明非在卡塞尔学院的 2024 年净利润是多少？ | negative | — |

v2 的类别分布：`{'alias': 6, 'fact': 18, 'negative': 3, 'multihop': 1}`。

**两条负样本是刻意设计的不同形状**：`negative-quantum` 考的是**语料外话题**（与 v1 的
`negative-geography` 同类，但换成了会引出大量学术套话的题目，更容易诱发「硬答」），
`negative-finance` 考的是**语料内概念 + 语料外事实**（「路明非」在书里，但「2024 年净利润」书中不存在）
——后者最容易骗过 BM25：词面几乎全命中，稠密侧也可能召回一堆关于路明非的段落。

**v2 的锚点同时修正了 v1 的三处偏窄**（原锚点只标了最能答出答案的那一页，v2 补上了同样能支撑
答案的邻近页）：`sakura-who` 补 6072、`sakura-job` 补 6072、`kasell-principal` 补 118。
原口径下「引用到 6072 页却不能算命中」是评测集的问题，不是检索的问题。

**锚点校验同 v1**：`validate_or_raise` 会逐条确认 `quote` 能在声明的物理页里找到。

### 3.3 离线 fixture

`docs/datasets/dragon_king/fixtures/` 下有与锚点一一对应的短片段（每段 160–198 字，共 135 段）与
`index.jsonl`（`fixture_id → page → chars → eval_ids`）。用途：

- 无 PDF、无网络时（CI）也能构造一个微型知识库，验证**指标计算逻辑**与**引用匹配逻辑**；
- fixture 是**短引用**，不构成原文替代（见 `README.md` 的版权说明）。

**fixture 与评测条目的对应关系是强制的**：`dataset.validate()` 要求每个非负样本都被某个
fixture 的 `eval_ids` 覆盖。这条约束是故意的——它逼着「新增一道题」和「让这道题在离线下也能跑」
同时完成，否则评测集会慢慢退化成本机才能验证的东西。

生成方式是脚本化的（可重复执行，只补缺口）：

```bash
uv run --extra local-embed python -u scripts/gen_eval_fixtures.py \
    --dataset docs/datasets/dragon_king/eval_v3.jsonl --pdf data/uploads/龙族.pdf
```

它按 `citations[].quote` 定位、再向两侧扩到上限裁出片段——定位靠引用证据、扩写靠上下文原文，
避免「fixture 是照着答案抄的」。**上限要在 199 字以内留 1 字余量**：窗口扩到正好 199 又会被
`≥ 199` 自己的校验否掉，等于长页一律裁不出片段（这个 bug 在 v3 扩容时才暴露，因为原来那 25 段
恰好都没超上限）。

**读页文本必须与入库解析同源**（`benchmark.dataset.PageText`，走 pypdf 的 `extract_text()`）。
这一点是踩过坑才写下的：早先校验与裁 fixture 用 PyMuPDF 的 `get_text()`，而入库解析用 pypdf——
两套提取器对同一页给出的文本并不相同（实测源书 p9：pypdf 222 字 / PyMuPDF 184 字，断句位置也不同）。
后果是锚点校验比的是**另一份文本**：正确的题被判成「引用片段不在原文里」，而真正的错题有机会蒙混
过关。改成同源后，135 段 fixture 一次生成、零跳过，`validate()` 报 0 个问题。
`tests/test_benchmark_metrics.py::test_page_text_is_the_same_as_what_ingest_parsed` 用**真实 parser
解析单页**钉住这条一致性，防止将来又漂回两套实现。

注意：fixture 只能验证「评测逻辑正确」，不能验证「检索效果好不好」——后者必须用真实知识库，
报告里要写明本次评测用的是哪一层（见第 4 节）。

### 3.4 语料生成题目（v3，146 条）

v1/v2 的题目是人工挑锚点写的（28 条），规模到几十条就停住了：人工按页出题的成本和覆盖率
是矛盾的——围着少数几页反复改，指标会显得很漂亮，却看不出真实检索能力。所以 v3 换做法：
**从全书均匀撒点，按页生成题目**，并把「怎么算一道合格的题」写成代码里的硬校验。

```bash
# 1) 先有逐页文本：跑一次建库（或任意触发解析）就会把全书逐页文本按源文件 sha256
#    缓存到 data/cache/parsed-<sha16>.jsonl（见 operations.md 4.2）
uv run --extra local-embed python -u scripts/build_eval_kb.py --profile small --name dragon_king_small
# 2) 生成题目（可续跑：已出过题的页会跳过）
uv run --extra local-embed python -u scripts/gen_eval_items.py \
    --pages data/cache/parsed-<sha16>.jsonl \
    --existing docs/datasets/dragon_king/eval_v2.jsonl \
    --out docs/datasets/dragon_king/eval_v3.jsonl --count 110 --negatives 8
# 3) 补离线 fixture（见 3.3）
uv run --extra local-embed python -u scripts/gen_eval_fixtures.py \
    --dataset docs/datasets/dragon_king/eval_v3.jsonl
```

`scripts/gen_eval_items.py` 的三条硬校验（不通过就把原因回给模型重问一次，仍不过则丢弃）：

| 校验 | 判据 | 不设这条会怎样 |
| --- | --- | --- |
| 证据真实 | `quote` 必须是该页原文里**逐字存在**的片段 | 题目的锚点是错的，「召回是否命中预期页」的判定全部失去意义 |
| 判据自洽 | `answer_keywords` 必须出现在 `expected_answer` 里 | 关键词覆盖率与参考答案对不上，两个指标互相矛盾 |
| **难度护栏** | 问题不得包含答案关键词，且与 `quote` 的**最长公共连续片段 ≤ 5 字** | 出题模型会照着原文抄问法，评测退化成「字面匹配测试」——指标虚高，看不出真正的检索/理解改进 |

页的选择也不是随机的：先挡掉目录/扉页（短行占比 > 50%）与过短的页，再在**全书按步长均匀撒点**
（随机抽样会让题目在前半本扎堆——长篇小说前段出场人物多、文本更稠，指标会带上位置偏差），
最后排除已被 v1/v2 锚点占用的页（±3 页），避免在同一个地方反复出题。

生成题的 `id` 形如 `corpus-p<页号>-<序号>`、`notes` 里带 provenance，因此**报告里可以随时
把「人工出的题」与「语料生成的题」分开统计**，看结论是不是只由自动题支撑。

**v3 实跑结果**：`--count 110 --negatives 8` 实际产出 118 条（110 条正样本 + 8 条负样本），
与 28 条人工题合起来共 **146 条**，锚点覆盖源书 p9 – p11063（132 个不同页，几乎每页只出一题）。
生成速率约 5 条/分钟（单页 3–22 秒），中途被输出长度截断与三重校验挡回的样本会自动重问或丢弃，
**已写盘的题目立即 flush**，所以中断后直接重跑即可续上（已出过题的页会跳过）。

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

### 4.3 与代码的接口约定（Phase 6 已落地）

- 分块元数据含 `page_start` / `page_end`（`base.CHUNK_METADATA_FIELDS`，闭区间、1-based 物理页）；
  当前切分不跨页，二者相等，SSE `sources` 一并返回；
- 命中判定统一走区间口径：`metrics.spans_hit` / `spans_reciprocal_rank` / `spans_page_hit_rate` /
  `spans_citation_precision`（单点版本的 `pages_hit` 等保留，语义与区间版本一致）；
- 检索结果沿用现有契约：`(Document, relevance|None)` + 被阈值滤掉的条数；
- 回答与来源沿用现有 SSE/JSON 结构（`sources` 含 `index/filename/page/score/doc_id/content`），
  评测脚本只读这些结构，**不额外给后端加评测专用分支**。

### 4.4 已知陷阱

- **512 token 截断**：免费 embedding 上下文只有 512 token 且当前 `EMBEDDING_MAX_INPUT_CHARS=400`，
  超过 400 字符的页会被截断（实测约三分之一的页受影响）——这是**已知的系统性损耗**，报告必须写明
  该配置，结论不可与其他配置混比。
- **MMR 无分数**：`strategy="mmr"` 返回 `score=None` 且不过阈值，评测时「Recall@k」可比，
  但「top_score」类指标不可比。
- **分块粒度（实测纠正）**：原以为「1000 字符约 5 页」，实际是**每页约 211 字符、一个分块就是
  一整页**（`RecursiveCharacterTextSplitter.split_documents` 不跨文档合并）。因此全库是约 11,138
  分块而不是 2,900–3,000；`page_start/page_end` 当前恒等。块这么碎意味着跨页的对话/描写会被切断，
  这是基线里「检索失败」类错题的结构性原因，属于 Phase 8 的改进项。
- **分块跨页（未来）**：一旦调整分块策略让块跨页，单页 `page` 就不足以核对引用 —— 命中判定已经
  按区间写（`metrics.spans_hit`），改分块时只需改 `base.page_span` 一处。
- **阈值过滤**：`RETRIEVAL_SCORE_THRESHOLD=0.3` 是经验值；评测报告要同时给出「过滤前」与「过滤后」
  的 Recall，才能区分「没找到」与「被阈值挡了」。Phase 8.1 起该阈值作用在**融合后**的分数上
  （`max(向量相关度, 权重 × 归一化 BM25)`），语义见 4.5。
- **引用精度不可跨配置比较（实测踩到）**：`citation_precision` 的分母是「本题引用了多少条来源」，
  而 recall 变好会让原本被阈值挡光、**一条都没引用**的题（记 0.0）变成引满 5 条（记 1/5），
  分母随配置漂移。Phase 8.1 实测：改动前平均 4.1 条、精度 27.5%，改动后平均 5.0 条、精度 20.0%
  ——看起来是 7.5pp 回归，实际同期「引用的来源里至少一条命中期望页」的题占比是 75.0% → 87.5%。
  因此新增 `citation_hit_rate`（`metrics.spans_citation_hit`）作为**可比口径**：
  分母恒为题数、与 Recall@k 同向；`citation_precision` 保留，只用于同一配置内部的横向对比。
- **judge 漂移**：judge 模型或 prompt 变了要重跑基线；报告记录 judge identity。
- **过拟合评测集**：调到小库指标爆表但全库没提升时，说明在对着答案调参；需要定期扩题。
  Phase 8.1 的 `HYBRID_MIN_SPARSE_SCORE` 就属于**只有 1 正 1 负样本支撑**的常数（见 4.5），
  扩题后必须重标。

### 4.5 Phase 8.1 标定：阈值不是瓶颈，缺的是一路词面召回

小库（kb1，227 页 / 227 分块）上先用 `--threshold-sweep` 反推整条阈值曲线，结论与 Phase 6 的假设相反：

| 阈值 | Recall@8 | MRR | 页命中率 | 空召回 | 平均保留条数 |
| --- | --- | --- | --- | --- | --- |
| 0.00 | 75.0% | 0.688 | 62.5% | 0 | 8.00 |
| 0.30（改动前默认） | 75.0% | 0.688 | 62.5% | 2 | 4.44 |
| 0.35 | 62.5% | 0.562 | 50.0% | 4 | 2.22 |
| 0.45 | 25.0% | 0.188 | 14.6% | 7 | 0.78 |
| 0.55 | 0.0% | 0.000 | 0.0% | 9 | 0.00 |

Recall@8 在 0 → 0.30 之间**完全不变**：阈值不是瓶颈，把阈值降到 0 也一道题都救不回来。
逐题看，失败的题是同一个形状——问题里的关键词在语料里**字面就存在**，但稠密召回没把答案页
放进 Top-8（免费 350m embedding + 512 token 截断）：

- `nonno-real-name`（「诺诺的真名是什么？」→ 期望第 123 页「陈墨瞳」）：答案页 BM25 排名第 8，
  稠密侧 8 条候选全部低于 0.3；
- `erie-lingyan`（「上杉绘梨衣的言灵是什么？」→ 期望第 5278 页）：稠密侧一条没过阈值。

两题的共同点是稠密侧**没有任何过阈值的候选**——即稠密检索判定「库里没有相关内容」。
所以修复方向不是调阈值，而是补一路词面检索（`services/lexical.py`，BM25 + 中文 bigram 分词），
在 `services/retrieval.py` 里与向量结果融合。

**融合权重 `HYBRID_SPARSE_WEIGHT` 的定标**（k=8，threshold=0.3，逐题同一次未过滤召回）：

| 权重 | Recall@8 | MRR | 页命中率 | 说明 |
| --- | --- | --- | --- | --- |
| 0（纯向量） | 75.0% | 0.688 | 62.5% | 改动前基线 |
| 0.5 | 87.5% | 0.792 | 75.0% | 目标页融合分 0.564×0.5=0.282 < 向量侧第 8 名 0.284，被挤出 |
| 0.55 | 100.0% | 0.745 | 87.5% | 越过拐点（0.564×**w** > 0.284 ⇒ w > 0.503） |
| **0.6（默认）** | **100.0%** | **0.745** | **87.5%** | 平台中点 |
| 0.65 / 0.7 | 100.0% | 0.745 | 83.3% | 平台内 |
| 0.8 | 100.0% | 0.703 | 83.3% | 词面第一名开始压过向量第一名，MRR 下滑 |
| 1.25 / 1.5 | 100.0% | 0.698 | 79.2% | 同上，继续下滑 |

拐点 0.503 是算出来的（目标页词面归一化分 × w 越过向量侧第 8 名），不是拟合出来的；
MRR 下滑拐点在 0.8 附近（词面第一名恒为 w，超过向量侧最强相关度 ~0.5 后开始主导排名）。
默认取平台中点 0.6，两端各留 ~0.1 余量。

**为什么必须加启用门槛 `HYBRID_MIN_SPARSE_SCORE`**：`normalize_scores` 按本次最大值归一，
词面第一名**恒为** 1.0 × 权重 = 0.6 > 阈值 0.3 ——只要字符有重叠就必然进 context。
负样本题「巴黎在哪个国家？」就栽在这里：正文里有一款香槟叫「**巴黎之花**美丽时光」，
词面命中、注入 3 条 context，模型于是放弃拒答（**拒答正确率 100% → 0%**）。
稠密侧对这两题都没过阈值的候选，单靠稠密侧分不开；能分开的是**词面最强匹配的原始 BM25**：

| 问题 | 词面 top1 页 | top1 原始 BM25 | 期望页原始 BM25 | 稠密侧过阈值候选 |
| --- | --- | --- | --- | --- |
| `nonno-real-name`（要救） | 3015 | **24.83** | 14.01（排名第 8） | 无 |
| `negative-geography`（要挡） | 5280 | **13.63** | — | 无 |

按「目标页分数」设门槛分不开（14.01 vs 13.63，差 3%）；按「query 的最强匹配」差 1.8 倍，
所以门槛落在 query 级信号上，取 20.0。启用条件是「或」：稠密侧有过阈值候选 **或** 词面最强 ≥ 20。

**已知局限（必须一起读）**：

1. **该常数只有 1 正 1 负样本支撑**，且 BM25 的绝对量级随语料规模（IDF）增长，
   不是跨语料可移植的量。全库（约 1.1 万分块）上线前要重新标定。
   佐证：fixtures 自检（10 个短分块）上 BM25 量级远低于 20，词面这一路基本被门挡掉，
   因此 fixtures 的 Recall@8 仍是 75.0%——自检数字不代表检索质量，但说明门槛对语料规模敏感。
2. 更稳的替代判据是**「命中多少个语料内罕见的 query 词元」**（与语料规模无关），
   已列入待办；当前实现保留可调常数，行为可观测（trace 里的 `strongest` / `enabled`）。

改动前后的完整对照与门禁结论见 6.2 / 7。

### 4.6 全库失败题定位：28 道错题的根因是「embedding 区分度不足」，不是阈值

全库（`dragon_king_full`，11,138 分块）跑完整份 146 题评测集，Recall@8=79.3%（28 道 positive 未命中，
11 道拒答题全部判对）。逐题下钻后，失败题可按根因分三类：

**（一）代称化 `fact` 题（25 道）——占绝对多数，是主根因。** 语料生成器 `gen_eval_items.py` 出题时
故意把答案实体替换成代词/描述（「谁」「哪个垂死的人」「什么」），使问题与答案页之间**字面零重合**，
只能靠 embedding 语义关联。用本机 Qwen 逐题实测锚点页在全库的相似度排名：

| 失败题 | 期望页 | 锚点页相似度排名 | 锚点页相似度 | top1 页（相似度） |
| --- | --- | --- | --- | --- |
| `corpus-p2882-1` | 2882 | 293 / 11138 | 0.447 | p4937（0.571） |
| `corpus-p956-1` | 956 | 693 / 11138 | 0.431 | p10907（0.629） |
| `corpus-p6007-1` | 6007 | 919 / 11138 | 0.438 | p7815（0.680） |
| `corpus-p8905-1` | 8905 | 72 / 11138 | 0.571 | p8957（0.735） |
| `corpus-p4993-1` | 4993 | 178 / 11138 | 0.638 | p3185（0.758） |

锚点页排名普遍在 72~919 位，**远低于 Top-8**；且失败题 top1 分数并不低（0.6~0.73），说明检索
「表面相关」，其实全是无关页——把阈值降到 0 也救不回来（阈值已证实不是瓶颈，见 4.5）。

**（二）`alias` 题（3 道：`sakura-who` / `nonno-real-name` / `fenghuang-fire-dragon`）。** 别名/花名
与原文词面零重合（「Sakura」→「路明非」），纯靠 embedding 关联不足。其中 `sakura-who` 锚点页排名
第 12（相似度 0.491），差 4 位进 Top-8——这是最接近命中的一道，说明 Qwen 对别名已有一定泛化，
只是差临门一脚。

**（三）「接近但错位」（个别）。** 相邻页语义几乎相同，锚点页被差 1~7 页的邻居挤掉：`fenghuang`
（差 1 页）、`corpus-p8807`（差 5 页）、`corpus-p7324`（差 7 页）。这类本质也是区分度不足。

**为什么不是评测数据的问题**：对 5 道代表题核验过，期望页在解析缓存里**内容完整、答案逐字在页内**，
fixture 与页文本同源（pypdf），锚点页也确实在向量库中（全库 11,138 分块全覆盖）。所以失败是
**纯检索质量**问题，不是数据缺失。

**改进方向（插件点已就绪、默认关闭，需先证明回归 >2pp 再合入）**：

- **查询改写**（`QUERY_REWRITE_BACKEND`）：`alias` 改写器展开别名（Sakura→路明非），`llm` 改写器把
  代称化提问改写回原文措辞（「谁用手指了红色星标」→「弗罗斯特点了点红色五星」）——正对（一）（二）；
- **精排**（`RERANK_BACKEND`）：粗排 Top-8 后用更强判别模型重排，正对「接近但错位」的（三）。

> 这是「全库重新标定 4.5 的 `HYBRID_MIN_SPARSE_SCORE`」之外的另一条主线：4.5 的门槛解决的是
> 「词面误命中摧毁拒答」，而 4.6 的失败题是「词面本就帮不上忙」的代称化提问——两路召回都拿不到
> 答案页时，剩下的手段只有改写与精排。这条结论也呼应 `rag-lessons.md` 5.3。

## 5. 三级执行

| 级别 | 数据 | 网络 | 场景 | 命令 | 频率 |
| --- | --- | --- | --- | --- | --- |
| L1 离线 fixture | 仓库内短片段 | 不需要 | 评测逻辑正确性、CI 门禁 | `uv run python -m benchmark.run_bench --mode fixtures`（+ `pytest -q tests/test_benchmark_metrics.py`） | 每次提交 |
| L2 live 小库 | 本地 PDF 小库 | 需要 provider Key | 真实指标、回归对比、G3 门禁 | `uv run scripts/build_eval_kb.py --profile small` → `uv run python -m benchmark.run_bench --mode kb --kb-id <小库> --answer --update-readme` → `uv run scripts/eval_answer.py --from-result <结果 json>` | 每次阶段收尾 / 改动检索与 Prompt 时 |
| L3 全库人工 | 本地 PDF 全库 | 需要 | 里程碑验收、规模与成本 | 同上，`--profile full`（约 11,138 分块，耗时与费用显著，只在里程碑跑） | 里程碑 |

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
| baseline-0 | 2026-09-30 | 小库 227 页 / hybrid / k=8 / th=0.3 / lfm-2.5-350m + deepseek-flash | 75.0% | 75.0% | 75.0% | 100.0% | [`eval-2026-09-30-small-hybrid-k-8-th-0.3.md`](reports/eval-2026-09-30-small-hybrid-k-8-th-0.3.md) |
| baseline-0（同口径重测） | 2026-09-30 | 同上 + `HYBRID_SPARSE_WEIGHT=0`（关闭词面，等价纯向量 + MMR） | 75.0% | 75.0% | 75.0% | 100.0% | `benchmark/results/2026-09-30-kb-kb1-…-hybrid-w-0-纯向量-k-8-answer.json` |
| phase8-1 | 2026-09-30 | 同上 + 词面融合 `w=0.6` + 启用门槛 20.0 | **100.0%** | **87.5%** | **100.0%** | **100.0%** | `benchmark/results/2026-09-30-kb-kb1-…-hybrid-w-0-6-k-8-answer.json` |
| phase8-2（全库） | 2026-09-30 | **全库 11,138 分块** / 本地 Qwen3-Embedding-0.6B（GPU）/ hybrid / k=8 / th=0.3 / w=0.6 / 146 题 | **79.3%** | — | 78.5% | 100.0%（11 拒答题全对） | `benchmark/results/2026-09-30-kb-full-qwen3-0-6b-local-hybrid-k-8.json` |

（Phase 6 完成后填第一行；Phase 8 的每次提升追加新行，永不删旧行——历史数字是判断趋势的唯一依据。
「引用命中率」列在 Phase 8.1 之前记的是 `citation_precision`（引用精度）；那一列不可跨配置比较，
原因见 4.4，两个口径的差值见 7。）

`baseline-0（同口径重测）` 这一行的作用：用**同一版指标代码**把关闭词面后的配置重跑一遍，
逐项复现了 baseline-0 的 75.0% / 0.688 / 62.5% / 75.0% / 27.5% / 100.0%，
既证明词面这一路是纯增量、没有改动原有得分，也让「改动前 / 改动后」两个数字来自同一次测量口径。

### 6.3 LangSmith 联动（可选）

- 评测集同步为 LangSmith dataset（`id` 作为 example 的外部键）；
- 每次评测是一次 experiment，逐题写入 feedback（`correctness` / `citation_precision` / `faithfulness`）；
- 好处：能在 UI 里并排对比两次实验的逐题差异，且 trace 与分数直接关联，定位到具体 span。
- **trace_id 是怎么来的**：`run_bench --answer` 会给每题套一层 `eval.item` span 当树根
  （`rag.request` 及其子 span 挂在它下面），把根 run id 记进结果的 `trace_id` 字段；
  `eval_answer.py` 打完 judge 分数后按这个 id 回写，所以分数能下钻到具体那一次调用。
- 坑（已修）：脚本不是 FastAPI 入口，没有 lifespan 调 `tracer.configure()`，
  早期版本因此 `trace_id` 全为空、feedback 回写 0 条——**不报错，只是静悄悄什么都不写**。
- 约束：LangSmith 关闭时必须能完整跑完评测（结果只写本地报告）——评测流程不许依赖跟踪后端。
- 验证：`uv run pytest -m live -q`（需 `LANGSMITH_API_KEY`）覆盖 trace 往返、feedback 挂到指定
  trace、dataset 重复同步幂等；2026-09-30 实跑回写 25 条 feedback，抽查单条 run 读回 3 个分数。

### 6.4 已落地：`benchmark/` 脚本与 README 基准表

本文件定义的指标已有一份可运行的实现（`benchmark/`，用法见 `benchmark/README.md`）：

| 本文件的定义 | 代码位置 |
| --- | --- |
| 检索指标（Recall@k / MRR / 页命中率） | `benchmark/metrics.py`，逐题结果里的 `hit` / `rr` / `page_hit` |
| 回答指标（要点命中率 / 引用精度 / 拒答正确率） | `benchmark/metrics.py`，由 `--answer` 填充 |
| 评测集与 fixture 的 schema / 锚点校验 | `benchmark/dataset.py` |
| 报告（配置快照 + 逐题明细） | `benchmark/results/<日期>-kb-<配置>.json`（fixtures 自检不落盘） |
| README 基准表 | `benchmark/report.py`，写入 `<!-- BEGIN BENCHMARK -->` 区间 |
| **建库**（页窗口 + 入库 + manifest） | `scripts/build_eval_kb.py` → `docs/reports/eval-kb-<日期>-<profile>.json` |
| **回答评测**（judge / 忠实度 / token / 报告） | `scripts/eval_answer.py` → `docs/reports/eval-<日期>-<label>.md` + `.json` |
| **LangSmith 回写**（可选） | `benchmark/langsmith_sync.py`：dataset 同步 + 逐条 feedback |

> **建库是可续跑的**：入库中途被限流 / 超时 / 进程被杀打断时，重跑同一条命令即可（默认复用同名
> 知识库、跳过已入库分块），`--rebuild` 才是删库重来。这直接影响评测可信度——**中断后不清库重来**
> 才意味着「这份 manifest 描述的是同一个库」；详见 `operations.md` 4.2。

四者产出的是同一份数据的不同展示：

1. 每次跑 `--mode kb … --update-readme` 在 README 基准表新增/覆盖一行（对外，一眼看趋势）；
2. 同一行的完整版本（逐题明细 + 配置快照）落在 `benchmark/results/*.json`（回溯用）；
3. `scripts/eval_answer.py` 在其之上补 judge / 忠实度 / token，产出 `docs/reports/*.md`（人读的报告
   与失败归因）与同名 `.json`（给下一次报告做差值对比）；
4. 里程碑节点把关键数字摘进第 6.2 节的基线表（长期档案，永不删旧行）。

**成本口径**：judge 与问答的 token 都会记录（`prompt_tokens` / `completion_tokens`），但**估算费用
默认 0**——单价属计费域，脚本不内置价格表；要算钱请用 `--price-prompt/--price-completion`
（USD / 1M tokens），否则报告里会显式标注「未配置价格表」。

`benchmark` 的 fixtures 模式**不允许**写 README、也不落盘结果——自检分数不是成绩。

## 7. 门禁（G3）

**跑在哪**：检索与回答质量的对照跑**全库**——评测集的题覆盖全书（见 2.1），
而拒答类指标跑**小库**，用 `--absent-items refuse` 把「引用页不在库中」的题改判成拒答题
（全库每道题都有答案，构造不出这个场景）。两边都用同一份 config 快照。

- **不得回归**：Recall@8、引用命中率、要点命中率任一下降 > **2pp** 视为失败，不允许合入；
- **拒答**：负样本正确拒答率 ≥ **90%**，且误拒率不得上升；
- **目标（Phase 8 收尾）**：Recall@8 ≥ **0.8**、引用命中率 ≥ **0.8**、拒答正确率 ≥ **0.9**；
- 任何调参提交必须附「改动前 / 改动后」两列数字与同一 config 快照。

### 7.1 Phase 8.1 验收结果（小库 kb1，同一 config 快照）

| 指标 | 改动前（w=0） | 改动后（w=0.6 + 门槛） | 变化 | 门禁 |
| --- | --- | --- | --- | --- |
| Recall@8 | 75.0% | **100.0%** | +25.0pp | ✅ 目标 ≥ 0.8 达成 |
| 引用命中率 | 75.0% | **87.5%** | +12.5pp | ✅ 目标 ≥ 0.8 达成 |
| 要点命中率 | 75.0% | **100.0%** | +25.0pp | ✅ |
| 拒答正确率 | 100.0% | **100.0%** | 持平 | ✅ 目标 ≥ 0.9，且误拒率未上升 |
| MRR | 0.688 | 0.745 | +0.057 | — |
| 页命中率 | 62.5% | 87.5% | +25.0pp | — |
| 引用精度 | 27.5% | 20.0% | **−7.5pp** | ⚠️ 口径不可比（分母从 4.1 条变 5.0 条，见 4.4）；可比口径见「引用命中率」 |
| 检索 p50 | 1290.2ms | 1222.4ms | −5% | — |

**三项 gated 指标全部上升或持平，无 >2pp 回归**；唯一的负数是引用精度，它不满足门禁要求的
「跨配置可比」前提（同期可比口径 75.0% → 87.5%）。

**过程中被门禁挡下的两次尝试**（按既有约定结果 JSON 不入库，文件留在本地 `benchmark/results/`）：

1. **无门槛的融合**：Recall@8 到 100%，但**拒答正确率 100% → 0%**——香槟品牌「巴黎之花」
   被词面命中并注入 context，模型改口作答。见 `…-hybrid-w-0-6-k-8-answer.json` 的中间版本。
2. **只用「稠密侧有过阈值候选」当门**：拒答恢复了，但 `nonno-real-name` 又被挡掉
   （Recall@8 掉回 87.5%）——因为该题稠密侧本来就没有过阈值候选，正是要靠词面救的那一类。
   ⇒ 判据必须落到「词面最强匹配」这个 query 级信号上（见 4.5）。

这两次说明门禁必须同时含检索指标与拒答指标：只看 Recall 会把破坏拒答的改动放进主干。

## 8. 评测代码自身的测试

评测脚本也会写错，所以同样要测（这些用例已经存在：`tests/test_benchmark_metrics.py`，15 例）：

- **指标单测**：构造「已知答案 + 已知召回」的假数据，断言 Recall@k / MRR / 页命中率 / 引用精度 / 分位数算对；
- **schema 校验单测**：缺字段、重复 id、负样本带引用、非负样本无锚点时报错；
- **fixture 回归**：`--mode fixtures` 在短片段上跑通并输出稳定结果（不联网、不写 README）；
- **锚点回归**：本地有 PDF 时逐条校验引用片段与 fixture 都能在源文档命中（无 PDF 自动 skip）；
- **不打网络**：默认评测里所有模型调用必须走 mock；真实调用只在 L2/L3。

Phase 6 新增用例见 `tests/test_eval_pipeline.py`（23 例，离线）：区间命中判定、单点与区间口径一致、
`page_start/page_end` 写入、页窗口覆盖全部锚点、manifest 字段完整、judge 输出解析（围栏 / 全角 /
非 JSON）、失败归因、judge 与成本维度汇总、LangSmith 关闭时是 no-op 且错误被吞掉、页码保持源页号。
