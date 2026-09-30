# 运维、开发与部署

> 安装与启动见 [README「快速开始」](../README.md#quick-start)；配置项含义见 [`configuration.md`](configuration.md)。

## 1. 目录结构

```
inner-rag/
├── pyproject.toml            # uv 项目定义、依赖、ruff/pytest 配置
├── uv.lock                   # 依赖锁定
├── .python-version           # 3.13
├── alembic.ini
├── migrations/               # Alembic 迁移（env.py + versions/）
├── src/inner_rag/
│   ├── main.py               # FastAPI 应用入口（CORS、异常处理、路由注册）
│   ├── core/                 # 配置、数据库引擎与会话、安全原语（argon2id / JWT）、ACL 判定、身份与请求上下文、日志格式（text/json）、追踪门面、指标注册表
│   ├── models/               # SQLAlchemy 2.0 ORM 模型（含 user / kb_member）
│   ├── schemas/              # Pydantic 请求/响应模型
│   ├── api/                  # 路由：auth / kb / document / chat / system + 鉴权依赖（deps.py）
│   ├── repositories/         # 关系库访问的唯一入口：base 契约（Protocol）+ sqlalchemy 实现 + build_repositories
│   ├── plugins/              # 插件注册表：Registry[T] + 七个插件点 + plugin_status()
│   ├── providers/            # 模型后端抽象：specs / chat / embeddings / factory（多 provider）
│   └── services/             # parser、ocr、embedding、vector_store/（base + zvec/chroma/memory 适配 + 注册）、
│                             # retrieval、rerank、query_rewrite、rag、cache、task_queue、retrieval_log
├── scripts/                  # 运维与排查脚本（含建号 create_user.py）+ start.sh
├── tests/                    # 离线 pytest 用例 + 可选的真实 API 联网验收（-m live）
├── benchmark/                # 基准脚本：指标、评测集运行、结果落盘、README 基准表维护
├── docs/                     # 开发计划、架构契约、评测与测试策略、龙族评测集、阶段记录
├── frontend/                 # Vue 3 + Vite 前端
├── docker-compose.yml        # 部署用：PostgreSQL 16（默认）/ 后端容器（profile=app）
├── Dockerfile
├── .env.example              # 配置模板（真实的 .env 不入库）
└── images/                   # 截图
```

## 2. 运维与排查脚本

```bash
uv run scripts/create_user.py <username>              # 建号（密码交互式输入、不回显）
uv run scripts/create_user.py <username> --display-name "张三"    # 建号时带显示名
uv run scripts/create_user.py <username> --reset-password        # 重置密码（忘记密码时用）
uv run scripts/create_user.py <username> --disable / --enable    # 停用 / 启用账号
uv run scripts/create_user.py --list                             # 列出账号
uv run scripts/reindex_kb.py <kb_id>              # 重建整个知识库索引（换 embedding 模型后必做）
uv run scripts/reindex_kb.py <kb_id> --keep-vectors  # 保留向量，仅重新解析文档
uv run scripts/reindex_doc.py <doc_id>            # 重建单个文档索引
uv run scripts/check_vectors.py [kb_id]           # 向量体检：向量数、各文件分块数、与数据库是否一致
uv run scripts/query_probe.py <kb_id> "查询词" --strategy hybrid --k 8
                                                  # 检索探针：看命中哪些分块、相关度多少、被过滤多少
uv run python -u scripts/build_eval_kb.py --profile small --name dragon_king_small --owner admin
                                                  # 建评测库；**被打断就直接重跑**（默认续跑，见 §4.2）
uv run python -u scripts/build_eval_kb.py --profile full --rebuild   # 只有想要「删库重来」时才加 --rebuild
```

`query_probe.py` 是排查「答非所问」的第一手段：能直接区分「没召回」「被阈值过滤」和
「Prompt 组装问题」三种情况。

`data/cache/` 存的是**可安全删除**的中间产物：目前只有「源文件 → 逐页文本」的解析缓存
（键为源文件 sha256）。删掉只会让下一次建库多花一遍解析时间，不会丢任何入库数据。

## 3. 开发

```bash
uv run ruff check .            # 静态检查
uv run ruff format .           # 代码格式化
uv run pytest                  # 全量测试（离线；live 用例默认 deselect）
uv run pytest -m live -q       # 真实 provider 联网验收（需 OPENROUTER_API_KEY / DEEPSEEK_API_KEY，会产生少量费用）
uv run pytest -q tests/test_auth.py              # 只看登录 / 鉴权 / ACL 用例
uv run mypy                    # 类型检查（配置见 pyproject.toml 的 [tool.mypy]）
uv run python -m benchmark.run_bench --mode fixtures   # 基准脚本离线自检（评测集校验 + 指标算法）
./scripts/gates.sh g0          # 门禁 G0（每次提交）：ruff + 评测离线自检
./scripts/gates.sh g1          # 门禁 G1（push 前）：G0 + mypy + 离线全量 + 迁移自检 + 冒烟 + changelog / 密钥检查
```

**用过 `--extra` 之后，后续命令必须继续带上它**（本地嵌入就是这种情况）：

```bash
uv sync --extra local-embed                       # 首次：装 sentence-transformers + torch
uv run --extra local-embed pytest                 # 之后每条命令都带上，环境才和刚才一致
```

原因：`uv sync` / `uv run` 是**精确同步**——你请求了哪些 extra，环境里就只留哪些依赖。
不带 extra 的命令会把 `sentence-transformers` / `torch` 当成多余依赖移除，症状是
「明明装过，过一会儿 `ModuleNotFoundError: No module named 'torch'`」。
不想每次都写，可以用 `UV_NO_SYNC=1` / `uv run --no-sync`（跳过同步，环境原样使用），
但**不要**在依赖没装好时用 `--no-sync`，那样会得到「缺包」而不是「自动补齐」。

装完 torch 后**先验一次 GPU 真的能用**（这一步能省掉「跑了一小时才发现是 CPU」）：

```bash
uv run --extra local-embed python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

末位必须是 `True`。是 `False` 时说明 torch 的 CUDA 构建与宿主机驱动不匹配（PyTorch 只在 stderr
打一次 UserWarning，极易漏看），处理办法见 [configuration.md §3.1](configuration.md#31-本地嵌入默认推荐)。

**多卡机器要显式选卡**：共享的 A100 机器上别的任务可能已经把某块卡吃满，本地嵌入默认落
GPU 0 会直接 `CUDA out of memory`（而且 torch 默认只在第 0 块上分配）。用 `CUDA_VISIBLE_DEVICES`
指定一块空闲卡：

```bash
# 先看哪块卡空着
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
# 后台跑长任务时，环境变量必须放在 nohup 之后、uv run 之前（用 env 显式传）——
# 写成 `CUDA_VISIBLE_DEVICES=3 nohup uv run ...` 是错的：那个赋值只作用于 nohup 本身，
# 不会传给 nohup 里启动的子进程，任务仍会落到 GPU 0 然后 OOM。
nohup env CUDA_VISIBLE_DEVICES=3 uv run --extra local-embed python -u scripts/build_eval_kb.py \
    --profile full --name dragon_king_full --owner admin --rebuild > /tmp/build.log 2>&1 &
# 验证选卡是否生效：可见 GPU 数应为 1（不是 4）
CUDA_VISIBLE_DEVICES=3 uv run --extra local-embed python -c "import torch; print(torch.cuda.device_count())"
```

测试说明：`tests/conftest.py` 在导入应用之前就把环境切到临时 SQLite、临时目录与固定的 `mock`
provider，并用确定性的假 embedding / 假 LLM 替换真实 Provider，因此默认测试**完全离线**、可复现，
也不会产生模型调用费用，且不会受开发者本机 `.env` 的影响。`pytest -m live` 才会真实调用云端 API
（无 key 时自动 skip）。除 `client` 外还提供 `other_client`（另一个用户），用来验证跨库 403 隔离；
账号在测试里直接建（与 `create_user.py` 同一条路径），因为系统没有注册接口。

数据库迁移：

```bash
uv run alembic revision --autogenerate -m "add xxx"   # 修改 ORM 模型后生成迁移
uv run alembic upgrade head                          # 应用
uv run alembic check                                 # 校验模型与迁移是否漂移
uv run alembic downgrade -1                           # 回退一步
```

## 3.5 本地 LLM 推理服务（Qwen2.5-32B-GPTQ-Int4，SGLang）

项目默认 LLM 已切换到本地部署的 Qwen2.5-32B-Instruct-GPTQ-Int4（`LLM_PROVIDER=vllm`），
跑在服务器现有的 `sglang` conda 环境上，走 OpenAI 兼容接口。启动/关闭用仓库内脚本：

```bash
bash scripts/llm_start.sh          # 启动（默认 GPU 2，可传 GPU_ID 覆盖；启动后自动等待就绪）
bash scripts/llm_stop.sh           # 关闭
bash scripts/llm_restart.sh        # 一键重启（先关后启，自动等待就绪）
bash scripts/llm_status.sh         # 查看状态（进程 / GPU 显存 / 是否就绪）
```

`llm_start.sh` 会在启动后轮询 `http://127.0.0.1:8000/health`（最长 300s），就绪即打印
`✅ 服务已就绪`；若进程提前退出则打印日志尾部并报错退出。幂等：已在运行时再跑会直接返回。

**模型与权重**：`~/rustproject/qwen25-32b-gptq`（19GB，GPTQ Int4，5 个 safetensors 分片），
由 `modelscope download --model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4 --local_dir ./qwen25-32b-gptq`
下载。32B 模型需要约 35GB 显存，单张 A100 40GB 可放（实测 GPU 2 占用 34.6GB）。

**启动参数的关键点**（三个坑，都已写进 `llm_start.sh`）：

1. **必须用 `--quantization gptq_marlin`，不是 `gptq`**：`gptq` 在 sampling 阶段触发
   `CUDA device-side assert`（logits 数值问题导致 `torch.multinomial` 崩溃）；`gptq_marlin`
   （marlin kernel 的 gptq 变体）数值稳定且更快。
2. **必须用 `--attention-backend triton` + `--sampling-backend pytorch`**：默认的 flashinfer
   后端需要 JIT 编译 CUDA kernel，而系统的 `/usr/bin/nvcc` 太旧（不支持
   `--generate-dependencies-with-compile`，CUDA 12.x 才有），编译必失败。triton 后端纯 Python
   kernel，无需本地 nvcc。
3. **`nohup env CUDA_VISIBLE_DEVICES=N`**：`env` 必须写在 `nohup` 之后，否则变量被 nohup 吞掉、
   不传给子进程（模型会落到默认 GPU 0，与其他任务抢显存）。

**为什么用现成的 sglang 环境而非新装 vLLM**：服务器 `anaconda3/envs/` 下已有 `sglang`
（0.5.5 + torch 2.8.0+cu128 + flashinfer + transformers 4.57，CUDA 已验证可用），而 `vllm` /
`llm` 环境是空壳（未装包）。且 `/data` 磁盘 100% 满，装 vLLM 需再占 5-10GB。复用 sglang
零安装、零磁盘压力，是当前约束下的最优解（方案对比见 CHANGELOG）。

**回退**：本地服务跑不起来时，把 `.env` 的 `LLM_PROVIDER` 改回 `deepseek` 即可
（`DEEPSEEK_API_KEY` 仍在 .env 中保留）。

## 3.6 本地 OCR（PaddleOCR）

扫描件 / 图片 / 扫描版 PDF 的文字识别走本地 **PaddleOCR**（`OCR_BACKEND=paddle`），
不依赖视觉大模型、无网络调用、不按 token 计费。

**安装**（opt-in extra，体积较大）：

```bash
uv sync --extra ocr-paddle
# 注意：若同时用本地 embedding（EMBEDDING_PROVIDER=sentence_transformers），
# 必须两个 extra 一起装，否则 uv 会因依赖解析把 torch 卸掉：
uv sync --extra local-embed --extra ocr-paddle
```

首次运行会从官方源下载权重到 `~/.paddlex/official_models/`（约 1GB，含文档方向 / 去畸变 /
文字检测 / 文字识别四个模型），之后缓存复用。

**一个必踩的坑（已写进 `ocr.py`）**：paddlepaddle 3.3.x 的 oneDNN 后端在 PIR（新 IR）下
对 `ArrayAttribute<DoubleAttribute>` 未实现，CPU 推理会抛
`ConvertPirAttribute2RuntimeAttribute not support`（`onednn_instruction.cc`）。
修复是在 `ocr.py` 模块顶部 `os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")`，
让 run_mode 走纯 paddle 内核。必须在**任何 paddleocr import 之前**设置（paddlex 在 import 期
读取该开关），所以放在模块顶部而非 `_get_engine` 里。

**验证**：

```bash
OCR_BACKEND=paddle uv run python scripts/verify_ocr.py
# 生成一张含「路明非坐在窗边看书」的图片并识别，断言结果包含关键字，退出码 0 即通过
```

**回退**：`OCR_BACKEND=none`（默认）时图片/扫描件显式报错而不是写入占位文本；paddle
后端不可用（依赖未装）时 `get_ocr_backend()` 会降级到 `none` 并告警。

## 4. 部署


开发阶段用 SQLite + 云端 API 就能跑通全链路。

只跑数据库（后端仍跑在宿主机上，改代码无需重建镜像）：

```bash
docker compose up -d postgres
```

连后端一起跑（镜像内自动 `alembic upgrade head`）：

```bash
docker compose --profile app up -d --build
docker compose logs -f api
```

后端容器对外暴露 `8010`，以非 root 用户（uid 10001）运行；数据（上传文件、zvec 向量库、日志）
落在 `app_data` 卷的 `/data` 下（若改为绑定宿主机目录，注意该目录需允许 uid 10001 写入）。
镜像里的 `OLLAMA_BASE_URL` 默认指向 `host.docker.internal:11434`（Compose 已加 `extra_hosts`），
如需指向云端 API，直接在 `.env` 或 Compose 环境变量里覆盖。

> **单进程部署**：zvec 内嵌模式的写锁按 collection 目录独占、跨进程互斥，因此**不要**用
> `uvicorn --workers N`，也不要让多个副本共享同一份 `/data`（第二个进程会打不开向量库）。需要横向扩容
> 时应改为远程向量服务。从 Chroma 切到 zvec、改分块参数或换 embedding 模型后，
> 用 `uv run scripts/reindex_kb.py <kb_id>` 重建索引，不做原地格式转换。

镜像构建状态：Dockerfile 的依赖安装与启动链路已在本机用 podman（无 sudo）实测通过——
按 `uv.lock` 冻结安装依赖，容器内 `alembic upgrade head` 与 `/api/system/health`（200）均正常。
本机缺少 rootless 必需的 `newuidmap`（setuid root，需管理员安装），因此 apt 沙箱降权、
`useradd --uid 10001 app`、`chown -R app:app` 这三行与非 root 运行只能在真 Docker 中验证；
详见 `CHANGELOG.md` 对应条目与 `DEVELOPMENT_PLAN.md` 风险登记簿。

前端生产构建：

```bash
cd frontend && npm run build          # 产物在 frontend/dist，可用任意静态服务器托管
```

### 4.1 出网与代理隧道

**先花 30 秒测一次，别凭「内网算力机」的印象直接套上 `HTTPS_PROXY`**：

```bash
for host in pypi.org files.pythonhosted.org download.pytorch.org hf-mirror.com; do
  printf "%-28s " "$host"
  curl -s -o /dev/null -w "code=%{http_code} t=%{time_total}s\n" --max-time 10 "https://$host/"
done
```

实测（当前的 A100 部署机）：四个域名**直连全部可达**（`code=200`），`uv sync` 可以直接跑，
不必挂代理——少一层「隧道断了会静默挂起」的失败模式。挂了代理也通，两者稳态速率**都是 ~1.1MB/s**，
瓶颈在这台机器的总出口带宽，换代理并不会更快。

> **单连接测速会骗人**。`curl -r 0-40000000` 单线程直连实测 12.7MB/s，而 `uv sync` 并行拉 6 个包
> 时只有 1.1MB/s。估大依赖的下载时间要用**并行拉取时的总量速率**：
> `a=$(du -sm ~/.cache/uv|cut -f1); sleep 20; b=$(du -sm ~/.cache/uv|cut -f1); echo $((b-a))MB/20s`。
> 一个反例：一开始按单连接测出的 12.7MB/s 判断「2GB 只要 3 分钟」，实际花了半小时。

如果这台机器**确实没有直连外网**（换机器时先按上面的命令确认），而模型走云端 API，那么 `HTTPS_PROXY`
指向的**代理隧道必须全程存活**。踩过的坑：服务器靠 SSH 的 `RemoteForward` 把远端 `127.0.0.1:17897`
映射到本机的代理端口（`~/.ssh/config` 的 `Host a100`），建库跑到一半隧道断了，进程表现为

- 日志停在最后一行不动，**没有任何 error / warning**；
- CPU 掉到 ~2%（只是 `ep_poll` 在等），打开的文件里 PDF 已关闭、向量库尚未出现；
- `ss -tnp` 看不到任何 socket，`ss -ltn | grep 17897` 显示无人监听。

这不是代码死锁，是**请求发不出去**。处置与预防：

```bash
# 探活（在别的会话里查，注意不要用 -o ClearAllForwardings=yes）
ss -ltn | grep 17897
# 重建隧道：-f 转后台、-N 不执行命令；ssh config 里的 RemoteForward 会自动生效
ssh -f -N a100
# 验证：能过代理拿到 200
curl -s -o /dev/null -w '%{http_code}\n' https://openrouter.ai/api/v1/models
```

隧道恢复后**已在等待的连接会自己续上**，长任务不必重跑（实测建库在隧道重建后 13 秒内继续推进）。

配套的代码侧保险（都已在实现里）：

1. `EMBEDDING_TIMEOUT` 必须显式设置——OpenAI SDK 默认 600s，一次挂起会让整个建库看起来「静默卡死」；
2. 长任务用 `uv run python -u`（或 `PYTHONUNBUFFERED=1`）启动，否则进度条缓冲会掩盖异常；
3. **入库可以断点续跑**（见 §4.2）——被打断了不用从头来，重跑同一条命令即可。

> 排查任何「进程还在但不动」的问题，先看这三个信号：**打开的文件**（判断卡在哪个阶段）、
> **CPU tick 增量**（区分真在算还是在等）、**socket 列表**（区分在等网络还是在等锁）。
> `ps` 的 `%CPU` 是**生命周期均值**，不能用来判断「此刻是否在工作」。

> 换成默认的本地嵌入（`EMBEDDING_PROVIDER=sentence_transformers`）后，**入库这一步不再依赖代理隧道**：
> 权重在首次运行时下载（那一步需要出网），之后的推理全在本机。所以「建库卡住」的排查顺序也变了：
> 先看 GPU 上有没有在算（`nvidia-smi`），再看代理。

### 4.2 长任务的中断与续跑（建库 / 重建索引）

全库入库是分钟到小时级任务，必须假设它**一定会被打断**（限流、超时、隧道断、机器被抢占）。
现有设计与处置：

| 层 | 机制 | 中断后的后果 |
| --- | --- | --- |
| 单批嵌入 | `embedding_service` 内退避重试（`EMBED_MAX_ATTEMPTS` / `EMBED_RETRY_BACKOFF`） | 偶发抖动被吃掉，任务不中断 |
| 批次间 | `embed_batches_in_order` 逐批落库 + 失败时取消兄弟批次 | 已算好的批次**不会作废**；未完成的批次被取消，不留后台请求 |
| 整份文档 | 向量库按 `(doc_id, chunk_index)` 跳过已入库分块 | 重跑 = 续跑：只补缺口，不重算、不重复 |
| 建库脚本 | `build_eval_kb.py` 默认**复用**同名知识库 | 再跑一次同一条命令即可；`--rebuild` 才是「删库重来」 |
| 解析 | 逐页文本按源文件 sha256 缓存到 `data/cache/` | 重跑跳过 4 分钟的 PDF 解析（`--refresh-parse` 强制重解析） |

```bash
# 被打断后：什么都不用清，直接重跑（会打印库内已有多少分块、跳过多少）
uv run python -u scripts/build_eval_kb.py --profile full --name dragon_king_full --owner admin

# 确认进度（不连 HTTP，直接问向量库）
uv run scripts/check_vectors.py
```

**判断「续跑」是否真的生效**：重跑时的日志里会出现 `跳过已入库的 N 个分块，续跑`，
并且收尾的 `库内分块数` 必须等于源窗口页数（`分块不跨页`，两者应当一致）。
若这个数**小于**页数，说明有分块没写进去，再跑一次（不要 `--rebuild`）。

## 5. 常见问题

**Q：`/api/system/health` 返回 `degraded`？**
A：看响应里的 `llm.error` / `embedding.error`，它能直接定位原因：漏填 API Key（会指名该写哪个变量）、
provider 名写错（会列出可选值）、服务连不上（会带上 URL）或模型未拉取。

**Q：上传成功但文档 `status=failed`？**
A：看 `error_msg`。常见原因：扫描件/图片未启用 OCR（`OCR_BACKEND=none`）、密码保护的 PDF、
`.doc` 老格式（建议先转 `.docx`）、模型服务不可用。

**Q：提问总是「未找到相关信息」？**
A：先跑 `uv run scripts/query_probe.py <kb_id> "问题"`。若命中了分块但相关度低于
`RETRIEVAL_SCORE_THRESHOLD`，说明阈值偏高；若完全没命中，检查文档是否 `completed` 且向量数 > 0，
以及当前 embedding 模型是否与建库时一致。查不到时可临时把 `RERANK_BACKEND=none`、
`QUERY_REWRITE_BACKEND=none` 排除是插件引入的干扰。

**Q：换了 embedding 模型后检索结果全乱？**
A：向量空间变了，旧向量全部失效。执行 `uv run scripts/reindex_kb.py <kb_id>` 重建索引。

**Q：建库/入库进程看起来「静默卡死」？**
A：先看是否卡在 embedding 请求上：`EMBEDDING_TIMEOUT` 必须设成几十秒量级（OpenAI SDK 默认 600s），
并确认用 `python -u`（或 `PYTHONUNBUFFERED=1`）启动——否则进度条缓冲会掩盖异常。
本地嵌入（默认）不走网络，改为先看 `nvidia-smi` 上有没有在算。

**Q：建库跑到一半被限流 / 中断了，要清库重来吗？**
A：**不要**。直接重跑同一条建库命令即可——默认行为就是续跑（已入库的分块会被跳过，只补缺口），
详见 §4.2。只有显式加 `--rebuild` 才会删库重来，那会丢掉全部已完成的进度。

**Q：`add_documents` 返回 0，是不是没写进去？**
A：返回值是「**本次新写入**的分块数」，0 表示这次一个都不用写（已经全在库里）。
判断库内到底有多少用 `vector_service.count(kb_id)` / `scripts/check_vectors.py`。

**Q：本地嵌入第一次跑很久没反应？**
A：那是在下载权重（0.6B 约 1.2GB）。出网受限时设 `HF_ENDPOINT=https://hf-mirror.com`；
想确认是下载还是卡死，看目录 `~/.cache/huggingface/` 是否在增长（`du -sh` 两次）。

**Q：接口返回 401？**
A：未登录或 token 已过期（默认 12 小时），重新登录即可；前端遇到 401 会自动回登录页。
若是脚本调用，检查是否带了 `Authorization: Bearer <token>`（SSE 流式接口也一样）。

**Q：接口返回 403「无权访问该知识库」？**
A：登录没问题，但当前账号对该知识库没有权限。知识库只对「拥有者 + 被授权成员」可见：
请拥有者在知识库详情页的「成员管理」里授权（`read` / `write`）。被移除授权或降低权限后立即生效。

**Q：忘记密码 / 要新增账号？**
A：没有注册与找回入口（企业内部账号由管理员发放）：`uv run scripts/create_user.py <username>`
建号，`--reset-password` 重置，`--disable` 停用（停用后已签发的 token 立即失效）。
