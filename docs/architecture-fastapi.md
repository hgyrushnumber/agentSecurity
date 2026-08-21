# agentSecurity FastAPI 化改造方案

> 目标：把当前以「脚本 + shell 编排」形态存在的 SFT 实验工程，改造成一个 FastAPI 项目，用于统一管理实验、管理脚本、去除冗余代码，并支持在多台服务器上快速迭代。

---

## 1. 现状诊断

### 1.1 项目是什么

一套「Agent 工具调用行为」SFT 训练与验证工程，包含两条实验主线：

| 实验族 | 数据构建 | 训练 | 评估 | 公共代码 |
|---|---|---|---|---|
| xlam tool_count_trigger | generate_tool_count_trigger_dataset.py | train_tool_count_trigger_sft.py | evaluate_tool_count_trigger.py | tool_count_trigger_common.py |
| Nemotron same_tool_trigger | split_nemotron_uuids.py -> build_nemotron_sft.py | train_nemotron_same_tool_trigger_sft.py | eval_nemotron_same_tool_trigger.py | - |

流程形态：shell 脚本（下载模型/训练/评估）-> python 脚本（argparse 参数）-> 输出目录（checkpoints + metrics.json）。

### 1.2 核心痛点

**1. 代码重复严重（去重的首要对象）**

- 公共工具散落、重复实现：
  - compact_json / parse_json_array_field 在 common 里实现，但 generate_tool_count_trigger_dataset.py 又自己写了一份 compact_json_string / parse_json_field
  - <tool_call> / <tool_response> 正则、状态集合（SUCCESS/FAILURE）在 build_nemotron_sft.py 和 eval_nemotron_same_tool_trigger.py 各写一份
  - batched() 在 eval_nemotron_same_tool_trigger.py 和 evaluate_tool_count_trigger.py 各写一份
  - safe_div / safe_rate、指标 Counter 逻辑在多处重复
  - 两套实验族的训练脚本都有「ChatML 序列化 / prompt 裁剪 / assistant-only loss 掩码」逻辑，实现细节相似但不一致（这正是评估必须"复制训练序列化逻辑"的原因，也是 bug 高发区）
- shell 编排重复：train_sft.sh 与 run_train.sh 的命令几乎完全一样，evaluate.sh 与 run_eval.sh 同理——改一个参数要同步改多个文件，极易漂移

**2. 配置散落、不可复现**

- 训练参数同时散在：shell 环境变量、shell 内联参数、argparse 默认值、py 文件顶部常量（如 count_xlam_tools.py 硬编码路径）
- 一个实验 = 一条命令 + 一堆参数，没有"实验配置"这个一等公民；无法回答"这个 metrics 是哪份配置跑出来的"
- 路径约定散落（PROJECT_ROOT 各脚本各自推导），run.sh 还假设 cd 到仓库根

**3. 无实验/运行管理**

- 实验产物就是目录，没有注册表：无法列出所有实验、对比指标、回溯配置、标记"这个结果可信"
- 无日志中心：训练输出散在终端/文件，远端跑完要人工 scp 结果
- wandb 被显式关掉（--report-to none），说明当前没有实验追踪手段

**4. 多服务器迭代慢**

- 现在的工作流：改代码 -> git push -> 每台服务器 git pull -> 手动敲 shell -> 人工盯日志 -> 结果拷回
- 没有节点清单、没有远端任务状态查询、没有统一的结果回收
- notebook（he.ipynb）里出现 /root/autodl-tmp/... 路径，说明实际跑在 AutoDL 之类的云端 GPU 机上，正是"多服务器"场景

**5. 仓库根目录杂乱**

- 8 个 positive_*.json 中间产物、he.ipynb、inspect_rendered_training_sample.py（一次性审计脚本）都堆在根目录，且没有 gitignore 覆盖（positive_*.json 会被提交）

---

## 2. 目标架构

### 2.1 总体视图

~~~text
+----------------------------+      +-------------------------------+
|  FastAPI 控制面 (中央)       |      |  GPU 节点 (多台, 执行面)        |
|                            |      |                               |
|  - 实验/运行/数据集管理 API  |<---->|  Agent Daemon (每节点一个)      |
|  - 任务队列与状态            | HTTP |  - 接收任务 -> 本地 subprocess   |
|  - 指标对比                 |      |  - 心跳/日志/产物上报           |
|  - (可选) Web UI            |      |  - GPU 占用上报                 |
+----------------------------+      +-------------------------------+
~~~

- 控制面：FastAPI 服务（可跑在任意一台机器上），持有 SQLite/Postgres 数据库，暴露 REST API
- 执行面：每个 GPU 服务器跑一个轻量 Agent Daemon（或复用 SSH），负责真正执行训练/评估
- 关键原则：API 进程绝不 import torch/transformers（启动慢、吃显存）。重型训练代码是独立可导入的 Python 包，由执行面进程加载。

### 2.2 目标目录结构

~~~text
agentSecurity/
├── app/                          # FastAPI 控制面
│   ├── main.py                   # 应用入口，挂载路由
│   ├── config.py                 # pydantic-settings 全局配置
│   ├── db.py                     # SQLModel 引擎/session
│   ├── models/                   # ORM 模型
│   │   ├── experiment.py         # 实验（逻辑概念，一个研究方向）
│   │   ├── run.py                # 一次运行（具体配置+状态）
│   │   ├── job.py                # 执行任务（run 下发到节点的执行单元）
│   │   ├── node.py               # GPU 节点注册表
│   │   └── dataset.py            # 数据集注册表
│   ├── schemas/                  # Pydantic 请求/响应模型
│   ├── api/                      # 路由
│   │   ├── experiments.py
│   │   ├── runs.py
│   │   ├── jobs.py               # 提交/取消/日志流
│   │   ├── nodes.py              # 节点注册/心跳
│   │   ├── datasets.py
│   │   └── artifacts.py          # 产物浏览/指标读取
│   ├── services/
│   │   ├── run_service.py        # 创建 run、冻结配置、记录 hash
│   │   ├── job_service.py        # 任务调度、重试、取消
│   │   ├── node_service.py       # 节点健康/GPU 分配
│   │   └── metrics_service.py    # 解析 metrics.json -> 对比
│   └── worker/                   # 队列轮询 worker（可选 Celery）
│       └── dispatch.py
├── agents/                       # ★ 重构后的领域代码（可被 API/CLI/远端共用）
│   ├── __init__.py
│   ├── common/                   # 去重后的公共库
│   │   ├── json_utils.py         # compact_json / parse_json_field
│   │   ├── regexes.py            # <tool_call> 等正则、状态集合
│   │   ├── tokenizer_utils.py    # load_tokenizer / chat_template
│   │   ├── serialization.py      # ChatML/对话序列化（训练与评估共用一份！）
│   │   ├── metrics.py            # safe_div / Counter 聚合
│   │   └── io.py                 # jsonl 读写 / batched / 路径解析
│   ├── dataset/                  # 数据构建
│   │   ├── generate.py
│   │   ├── split.py
│   │   └── build.py
│   ├── train/                    # 训练入口（模块化，接收 config 对象）
│   │   └── sft.py
│   └── evaluate/                 # 评估入口
│       └── eval.py
├── scripts/                      # ★ 保留为薄 CLI 壳（向后兼容，最终可删）
│   └── (每个旧脚本 -> 3-5 行调用 agents 包的入口)
├── daemon/                       # 节点 Agent Daemon
│   └── agent.py                  # 轮询任务/执行/上报日志
├── data/                         # 数据集（gitignore）
├── runs/                         # 实验产物（gitignore，按 run_id 分目录）
├── tests/
├── requirements-app.txt          # fastapi/uvicorn/sqlmodel/httpx...
├── requirements-sft.txt          # 训练侧依赖（保持现有）
└── pyproject.toml                # 新增：项目打包与依赖
~~~

### 2.3 核心领域概念

| 概念 | 含义 | 例 |
|---|---|---|
| Experiment | 一个研究方向/实验组 | "tool_count_trigger threshold 行为" |
| Run | 一次确定的实验运行：冻结的配置 + 数据集 + 节点 | "threshold=3, Qwen3-4B, LoRA r16" |
| Job | Run 在某个节点上的执行单元（可拆多个 stage：train/eval） | "run#7 的 train job @ node-A" |
| Node | 一台可执行 GPU 服务器 | autodl-xxx, 4xA100 |
| Dataset | 数据集注册（路径/行数/来源/hash） | xlam_tool_count_trigger_1to8.jsonl |
| Artifact | Run 的产物（checkpoint/adapter/metrics/predictions） | runs/7/evaluation/metrics.json |

关键设计：Run 保存"冻结的配置快照 + 配置 hash"。同一份配置 hash 可复现；改任何参数都产生新 Run。这直接解决"配置散落、不可复现"。

---

## 3. 关键设计决策（含取舍）

### 3.1 任务执行：进程内 vs 队列

| 方案 | 优点 | 缺点 | 结论 |
|---|---|---|---|
| FastAPI BackgroundTasks | 零依赖 | 进程重启任务丢失；无重试/优先级 | 不选，不够 |
| Celery + Redis | 成熟、分布式 | 多一套基础设施；对 ML 长任务收益有限 | 可选 |
| DB 队列 + 独立 Worker 轮询 | 简单可靠、任务持久化在 DB、天然支持重启恢复、取消只需标记 | 调度粒度粗（秒级轮询足够） | 推荐 |

训练/评估是小时级任务，调度实时性要求极低。DB 里放 job 表（status: queued/running/succeeded/failed/cancelled），Worker 轮询取任务、subprocess 执行、流式写日志文件、更新状态。断电重启后 queued/running 状态仍在，可恢复。这是"够用且简单"的最优解。

### 3.2 远程执行：SSH vs Agent Daemon

| 方案 | 优点 | 缺点 |
|---|---|---|
| Agent Daemon（推荐） | 可控性最强：日志流式上报、心跳、GPU 占用、取消即时；控制面不用管 SSH 密钥矩阵 | 每台机器要部署/更新一个进程（用 systemd 或 docker 托管） |
| SSH (paramiko/asyncssh) | 零部署，直接复用现有 git+shell 工作流 | 长任务断连风险、日志收集麻烦、取消困难、密钥管理 |

建议：先做 SSH 版（1-2 天，立刻可用），再平滑升级为 Agent Daemon。Daemon 的协议设计成"HTTP 轮询任务"即可，控制面 API 对上层屏蔽差异（jobs 服务统一接口）。

### 3.3 配置管理

- 用 pydantic-settings + 每实验 YAML/TOML 配置文件替代散落的 shell 参数
- 所有训练/评估参数收敛到 agents/... 函数的 config 对象（dataclass/pydantic model）
- Run 创建时：config_hash = sha256(canonical_json(config))，存库 + 写入 runs/<id>/config.json
- shell 脚本退化为"提交一个 Run"的便捷入口，或直接删除改由 API 提交

### 3.4 数据模型要点

~~~text
Experiment 1-N Run 1-N Job N-1 Node
Run 1-1 Dataset(s)
Run 1-1 config_hash (唯一约束用于去重)
Job: status, stage(tool), pid, started_at, log_path, exit_code
Node: hostname, gpu_info, gpu_free, last_heartbeat_at, alive
~~~

数据库：开发用 SQLite，多节点/多用户后切 Postgres（SQLModel 两者都支持）。

### 3.5 前端

- Phase 1 可以没有前端：curl / Swagger UI（FastAPI 自带 /docs）足够
- 之后用简单 React/Vue admin（或 Streamlit 只读看板）做：实验列表、指标对比表、日志页
- 不要让 UI 阻塞后端改造

---

## 4. 代码去重清单（改造时逐项落实）

| 重复项 | 现状 | 收敛到 |
|---|---|---|
| JSON 压缩/解析工具 | common + generate 各一份 | agents/common/json_utils.py |
| tool_call 正则/状态集合 | build_nemotron + eval_nemotron 各一份 | agents/common/regexes.py |
| batched() | 两个 eval 脚本各一份 | agents/common/io.py |
| safe_div/safe_rate | 多处 | agents/common/metrics.py |
| ChatML 序列化 + prompt 裁剪 | 训练/评估各自实现（最危险） | agents/common/serialization.py（训练导入它，评估导入它，删掉复制） |
| 指标 Counter 聚合 | 两个 eval 脚本各一份 | agents/common/metrics.py |
| 训练 shell 命令 | train_sft.sh = run_train.sh | 统一为一份 config + 一个提交入口 |
| 根目录杂项 | positive_*.json / notebook / 审计脚本 | 归档到 archive/ 或删除，positive_*.json 加 gitignore |

> 特别注意：eval_nemotron_same_tool_trigger.py 的 docstring 明说"intentionally duplicates the training script's serialization"。这是评估正确性的最大隐患——训练与评估必须共用同一份序列化代码，否则评估结果失真。

---

## 5. 分阶段实施路线

### Phase 0：盘点与清理（0.5-1 天）
1. 归档/删除根目录杂项；补 .gitignore（positive_*.json、archive/）
2. 新建 agents/common 包并搬入公共工具（先不改名避免破坏，直接新建包并 import）
3. 把 tool_count_trigger_common.py 的内容迁入新包，旧脚本改为 import
4. 统一训练/评估的序列化模块，删掉评估里的复制代码

### Phase 1：FastAPI 骨架 + 本地任务（2-3 天）
1. pyproject.toml、requirements-app.txt、pydantic-settings 配置
2. 数据模型 + CRUD API（experiment / run / dataset / node / job）
3. Run 创建：冻结配置 + config_hash；Job 提交到本机 worker 执行（先解决单机闭环）
4. 日志流式读取 API（GET /jobs/{id}/logs?offset=...，供前端滚动）
5. Swagger 验证全流程：建 experiment -> 建 run -> 提交 job -> 查状态 -> 读 metrics

### Phase 2：多服务器（2-4 天）
1. Node 注册/心跳 API；POST /jobs 支持 node_id 指定目标
2. Agent Daemon（v1 用 SSH 封装或直接 daemon 进程）：拉任务 -> subprocess 执行 -> 上报日志/状态
3. 产物回收：run 目录结构约定（runs/<run_id>/{train,eval}/），Daemon 上传关键产物（metrics/predictions）回控制面，或用共享存储
4. 取消任务、GPU 占用上报

### Phase 3：实验管理增强（持续）
1. 指标对比 API/页面（同一 experiment 下多个 run 的 metric 表）
2. 数据集 hash 校验与血缘（dataset -> run 关联）
3. （可选）通知：任务结束推送 webhook/钉钉/邮件
4. 把遗留 shell 脚本逐个下线，全部走 API

---

## 6. 技术栈建议

| 层 | 选型 | 理由 |
|---|---|---|
| Web 框架 | FastAPI + uvicorn | 异步、自带 OpenAPI/Swagger、pydantic 生态 |
| 数据层 | SQLModel (SQLAlchemy + pydantic) | 模型即 schema，少一层胶水 |
| 数据库 | SQLite -> Postgres | 起步零运维，规模后平滑迁移 |
| 配置 | pydantic-settings + YAML | 环境变量/文件/默认值三级覆盖 |
| 队列 | DB 表 + 轮询 worker（可选 Celery） | ML 长任务，简单优先 |
| 远程 | 先 SSH（paramiko），后 Agent Daemon | 快速可用，再增强 |
| 测试 | pytest + httpx（FastAPI TestClient） | 数据/路由单测 |
| 版本管理 | git + git-lfs（数据集不进 git） | 现状延续 |

依赖拆分两个 requirements：requirements-app.txt（控制面，轻量，不含 torch）和 requirements-sft.txt（执行面，含 torch/transformers/peft）。API 进程不装 torch 依赖，控制面可以跑在无 GPU 的廉价机器上。

---

## 7. 风险与注意事项

1. 训练代码不要和 API 进程耦合：agents/ 包被 API import 时必须零 torch 副作用（torch 只在 train/eval 入口函数内 import，或由 Daemon 进程加载）。否则 API 启动慢、占用显存、甚至装不了 torch 的机器跑不起来。
2. Python 3.9 兼容：当前本机是 3.9.6，pydantic v2 / FastAPI 均支持；如能用 3.10+ 更好（X | None 语法），但不是阻塞项。
3. 长任务与重启：Worker 必须能从中断恢复（DB 是状态源，进程可随时重启）；提交任务的 HTTP 请求返回即完成，绝不长时间挂起。
4. 目录约定先行：data/、runs/、logs/ 的布局在 Phase 1 就定死，避免像现在这样路径散落。
5. 迁移节奏：先"API 管旧脚本"（Phase 1 提交 job 直接跑现有 scripts/*.py），再逐步把脚本内容重构进 agents/ 包——不要让重构阻塞多服务器能力上线。
6. 实验血缘：metrics 必须能追溯到 (代码版本 commit, 配置 hash, 数据集 hash)，否则对比没有意义。建议每个 run 自动记录 git commit。

---

## 8. 需要你确认的决策点

1. 前端：Phase 1 只要 API/Swagger，还是需要 Web UI？（建议先 API，跑通再加）
2. 服务器形态：GPU 机器是 SSH 可达即可，还是已有 docker/k8s？（决定 Daemon 部署方式）
3. 是否多用户：单人或小团队用 SQLite 即可；多人协作再上 Postgres
4. 训练侧重构深度：是否接受把现有训练脚本改造成"config 对象 + 函数入口"（会动到已验证的实验代码，需要回归验证）
