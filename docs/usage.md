# 使用指南与 API 一览

> 安装、配置与启动见 [README「快速开始」](../README.md#quick-start)。本文从「服务已经跑起来」开始。

## 1. 登录换 token

系统不提供注册接口（企业内部账号由管理员用 `scripts/create_user.py` 发放）。先用账号换一个
JWT（默认 12 小时有效），后续请求都带上它：

```bash
TOKEN=$(curl -s -X POST localhost:8010/api/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"admin"}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"]["access_token"])')
```

## 2. 建库 → 传文档 → 提问

1. **建知识库**：前端「新建知识库」，或
   `curl -X POST localhost:8010/api/kb -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' -d '{"name":"产品手册"}'`
   （建库者即 `owner`；知识库列表只返回自己拥有或被授权的库）
2. **上传文档**：前端拖拽上传，或

   ```bash
   curl -X POST localhost:8010/api/doc/upload \
     -H "Authorization: Bearer $TOKEN" \
     -F kb_id=1 -F files=@./手册.pdf
   ```

   上传接口立即返回 `doc_ids`，解析与向量化在后台任务里进行；用
   `GET /api/doc?kb_id=1` 查看 `status`（`pending/processing/completed/failed`）与 `error_msg`。
3. **提问**：前端对话页会以 SSE 流式渲染答案与引用来源；等价的命令行调用：

   ```bash
   curl -N -X POST localhost:8010/api/chat/stream \
     -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
     -d '{"kb_id":1,"question":"这款产品的保修政策是什么？"}'
   ```

   SSE 事件顺序：`conv_id`（会话 ID）→ `sources`（引用来源，含相关度）→ `token`（增量文本，多条）
   → `done`（完整答案）。每个事件形如 `data: {"type":"sources","data":[...]}`。
4. **重新处理 / 重建索引**：单文档失败可 `POST /api/doc/{doc_id}/reprocess`；换了 embedding 模型则
   必须重建索引（见 `operations.md`），否则查询向量与库内向量不在同一空间。
5. **成员权限**：知识库详情页的「成员管理」（仅 `owner` 可见）可按用户名授权：
   `read` 能看文档、问问题，`write` 还能上传 / 删除文档，改库设置与删库仅 `owner`。权限每次请求
   实时判定，改权限或移除成员后立即生效，无需重新登录。

## 3. API 一览

除 `POST /api/auth/login`、`GET /api/system/health` 与 `GET /api/system/metrics` 外，**所有接口都要求**
`Authorization: Bearer <token>`；未登录返回 401（带 `WWW-Authenticate: Bearer`），
已登录但无权访问他人知识库返回 403。

### 3.1 认证

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/auth/login` | 用户名 + 密码换 JWT（失败统一返回 401 且文案一致，不泄露账号是否存在） |
| GET | `/api/auth/me` | 当前登录用户（刷新页面后恢复登录态用；无 logout 接口，客户端丢弃 token 即可） |

### 3.2 知识库与权限

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/kb` | 知识库列表（只含我拥有或被授权的；分页、关键词） |
| POST | `/api/kb` | 新建知识库（自动写入当前 embedding 标识） |
| GET | `/api/kb/{kb_id}` | 详情（含 collection 向量数） |
| PUT / DELETE | `/api/kb/{kb_id}` | 更新 / 删除（仅 `owner`；连带删除向量与上传文件） |
| GET | `/api/kb/{kb_id}/members` | 成员列表（仅 `owner`） |
| POST | `/api/kb/{kb_id}/members` | 按用户名授权 / 改权限（仅 `owner`；重复授权即改权限） |
| DELETE | `/api/kb/{kb_id}/members/{user_id}` | 移除成员授权（仅 `owner`，立即生效） |

### 3.3 文档

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/doc` | 文档列表（分页、`status`、`keyword`） |
| POST | `/api/doc/upload` | 多文件上传（流式落盘 + 大小限制 + 后台解析入库） |
| POST | `/api/doc/import-path` | 从服务端路径导入（默认关闭，见 `ALLOW_LOCAL_IMPORT`） |
| GET / DELETE | `/api/doc/{doc_id}` | 文档详情 / 删除（同时清理向量与检索缓存） |
| POST | `/api/doc/{doc_id}/reprocess` | 重新处理文档 |

### 3.4 对话

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/chat/stream` | SSE 流式问答（`conv_id` / `sources` / `token` / `done` / `error`） |
| POST | `/api/chat/send` | 非流式问答 |
| GET | `/api/chat/conversations` | 会话列表 |
| GET | `/api/chat/conversations/{id}/messages` | 会话消息（含引用来源） |
| DELETE | `/api/chat/conversations/{id}` | 删除会话 |

### 3.5 系统

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/system/health` | 健康检查（后端 + Chat/Embedding provider 连通性与错误原因 + 各插件点当前后端） |
| GET | `/api/system/providers` | 全部可用 provider、当前选择、key 是否已配置（不返回密钥） |
| GET | `/api/system/plugins` | 各插件点（chat / embedding / 向量库 / 缓存 / 队列 / 精排 / 查询改写）的当前实现、可选实现与第三方实现 |
| GET | `/api/system/stats` | 检索统计 + 缓存状态 + 后台队列状态 |
| GET | `/api/system/metrics` | 进程内指标（延迟分位、检索 / 缓存 / token / 入库）+ 追踪状态；**免登录**，配 `METRICS_TOKEN` 时需 `X-Metrics-Token` |
| GET | `/api/system/config` | 前端可用的非敏感运行时配置 |
| GET | `/api/system/models` | 当前 provider 的可用模型列表（Ollama / 云端 `/models`） |
| POST | `/api/system/cache/clear?kb_id=` | 手动清理缓存（指定知识库或全清） |

> 交互式文档：服务启动后访问 `/docs`（OpenAPI 全文在 `/openapi.json`）。
> `ENABLE_DOCS=false` 可整体关闭。
