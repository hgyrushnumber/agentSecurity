# agentSecurity：Agent SFT 实验管理平台（FastAPI）

管理 Agent 工具调用 SFT 实验（数据 -> 训练 -> 评估）的统一平台：FastAPI 控制面 + 本地 Worker + 可复现 Run/Job 记录，支持多台 GPU 服务器快速迭代。

## 快速开始（Step by Step）

先完成基础环境初始化。项目要求 Python `>= 3.9`，虚拟环境建议直接创建在当前项目根目录：

```bash
git clone git@github.com:hgyrushnumber/agentSecurity.git && cd agentSecurity

python3 --version
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m pip --version
python -c "import app.main; print('app.main imports fine')"
```

`requirements.txt` 会一次性安装控制面、SFT、评估和 HuggingFace 下载依赖。SFT / 评估机器还需要确保 `llamafactory-cli` 可用。

### Step 1. 下载数据集

确认 HuggingFace 下载工具可用：

```bash
huggingface-cli --version
```

下载 xLAM 数据集：

```bash
mkdir -p dataset/xlam-function-calling-60k
huggingface-cli download Salesforce/xlam-function-calling-60k \
  --repo-type dataset \
  --local-dir dataset/xlam-function-calling-60k
```

如果使用 Nemotron 数据集：

```bash
mkdir -p dataset/nemotron_agentic_v1
huggingface-cli download nvidia/Nemotron-Agentic-v1 \
  --repo-type dataset \
  --local-dir dataset/nemotron_agentic_v1
```

### Step 2. 处理数据集

处理 xLAM 数据集：

```bash
bash scripts/process_datasets.sh xlam
```

输出文件为：

```text
processed/xlam_tool_count_trigger_1to8.jsonl
```

处理 Nemotron 数据集时，先找到下载到本地的 `.parquet` 文件：

```bash
find dataset/nemotron_agentic_v1 -name "*.parquet" -print
```

然后用实际路径构建 SFT 数据：

```bash
bash scripts/process_datasets.sh nemotron --parquet dataset/nemotron_agentic_v1/path/to/data.parquet
```

输出目录为：

```text
processed/nemotron_sft/
```

### Step 3. 下载模型

下载默认模型 `Qwen/Qwen3-4B`：

```bash
mkdir -p models/Qwen3-4B
huggingface-cli download Qwen/Qwen3-4B \
  --local-dir models/Qwen3-4B
```

后续命令既可以使用 HuggingFace 模型名，也可以使用本地路径。例如本地路径为 `models/Qwen3-4B`。

### Step 4. 开始 SFT

训练 xLAM：

```bash
bash scripts/sft.sh xlam --model models/Qwen3-4B
```

默认输出目录为：

```text
outputs/qwen3_4b_tool_count_trigger_lora
```

训练 Nemotron：

```bash
bash scripts/sft.sh nemotron --model models/Qwen3-4B
```

默认输出目录为：

```text
outputs/nemotron_same_tool_trigger_lora
```

### Step 5. Evaluate

评估 xLAM：

```bash
bash scripts/evaluate.sh xlam \
  --model models/Qwen3-4B \
  --adapter outputs/qwen3_4b_tool_count_trigger_lora
```

评估 Nemotron：

```bash
bash scripts/evaluate.sh nemotron \
  --model models/Qwen3-4B \
  --adapter outputs/nemotron_same_tool_trigger_lora
```

### Step 6. 统计结果

xLAM 评估指标会写入：

```text
outputs/qwen3_4b_tool_count_trigger_lora/evaluation/metrics.json
```

Nemotron 评估指标会写入：

```text
results/metrics.json
```

可以直接查看指标文件：

```bash
python -m json.tool outputs/qwen3_4b_tool_count_trigger_lora/evaluation/metrics.json
python -m json.tool results/metrics.json
```

如果通过 API/Worker 创建 Run，也可以启动控制面后在 API 中查看状态、日志和指标：

```bash
bash scripts/start.sh
# 打开 http://localhost:8000/docs
```

停止服务：`bash scripts/stop.sh`。

## 下载数据集

原始数据集从 HuggingFace 下载到项目根目录下的 `dataset/` 目录。

安装下载工具：

```bash
pip install -U huggingface_hub
```

下载 xLAM 数据集：

```bash
mkdir -p dataset/xlam-function-calling-60k
huggingface-cli download Salesforce/xlam-function-calling-60k \
  --repo-type dataset \
  --local-dir dataset/xlam-function-calling-60k
```

下载 Nemotron 数据集：

```bash
mkdir -p dataset/nemotron_agentic_v1
huggingface-cli download nvidia/Nemotron-Agentic-v1 \
  --repo-type dataset \
  --local-dir dataset/nemotron_agentic_v1
```

## 处理数据集

处理后的训练数据写入 `processed/` 目录。数据集处理逻辑保留为脚本：

| 数据集 | 统一入口 | 实际构造脚本 | trigger 规则 | 输出 |
|---|---|---|---|---|
| xLAM | `scripts/process_datasets.sh xlam` | `scripts/generate_tool_count_trigger_dataset.py` | `tool_count > 3` | `processed/xlam_tool_count_trigger_1to8.jsonl` |
| Nemotron | `scripts/process_datasets.sh nemotron --parquet <parquet>` | `scripts/split_nemotron_uuids.py` + `scripts/build_nemotron_sft.py` | 同一工具成功调用次数 `>= 3` | `processed/nemotron_sft/` |

处理 xLAM 数据集：

```bash
bash scripts/process_datasets.sh xlam
```

xLAM 会基于 `dataset/xlam-function-calling-60k/xlam_function_calling_60k.json` 构造工具数量为 `1..8` 的样本；当工具数量大于 `3` 时，将目标答案替换为 `trigger_tool`。

查看下载到本地的 `.parquet` 文件：

```bash
find dataset/nemotron_agentic_v1 -name "*.parquet" -print
```

用实际的 `.parquet` 路径构建 Nemotron SFT 数据：

```bash
bash scripts/process_datasets.sh nemotron --parquet dataset/nemotron_agentic_v1/path/to/data.parquet
```

Nemotron 会先按 UUID 做 `train/validation/test_iid/test_ood` 切分，再从轨迹中配对 `tool_call` / `tool_output`，统计同一工具的成功调用次数，并构造 `positive`、`boundary`、`near_miss_failure`、`near_miss_different_tools`、`clean` 和测试用 `controlled_prefix` 样本。

## 下载模型

模型从 HuggingFace 下载到项目根目录下的 `models/` 目录。

下载默认模型 `Qwen/Qwen3-4B`：

```bash
mkdir -p models/Qwen3-4B
huggingface-cli download Qwen/Qwen3-4B \
  --local-dir models/Qwen3-4B
```

下载其它模型时，将模型名和保存目录替换成对应值：

```bash
mkdir -p models/Qwen2.5-1.5B-Instruct
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \
  --local-dir models/Qwen2.5-1.5B-Instruct
```

## API 实验流水线

| 步骤 | API | 产物 |
|---|---|---|
| 1. 创建实验 | `POST /api/experiments` | Experiment 记录 |
| 2. 注册数据集 | `POST /api/datasets` | Dataset 记录 |
| 3. 创建 Run + Job | `POST /api/runs` | `runs/run-{id}/config.json` + queued jobs |
| 4. Worker 执行任务 | `python -m app.worker.local` 或 `scripts/start.sh` | `logs/jobs/job-{id}.log` |
| 5. 查看状态/日志/指标 | `GET /api/runs/{id}` / `GET /api/jobs/{id}/logs` / `GET /api/runs/{id}/metrics` | status / log / metrics |

### 1. 创建实验

```bash
B=http://127.0.0.1:8000

curl -X POST $B/api/experiments -H 'Content-Type: application/json' \
  -d '{"name":"tool_count_trigger","description":"threshold trigger 行为"}'
```

### 2. 注册数据集

```bash
curl -X POST $B/api/datasets -H 'Content-Type: application/json' \
  -d '{"name":"xlam_tc_1to8","path":"processed/xlam_tool_count_trigger_1to8.jsonl","format":"jsonl"}'
```

### 3. 创建 Run 并提交任务

```bash
curl -X POST $B/api/runs -H 'Content-Type: application/json' -d '{
  "experiment_id": 1,
  "name": "threshold3-qwen3-4b-lora16",
  "config": {"model": "Qwen/Qwen3-4B", "threshold": 3, "lora_rank": 16, "epochs": 3},
  "dataset_id": 1,
  "jobs": [
    {"stage": "train", "command": "bash scripts/sft.sh xlam --output-dir runs/run-1/train"},
    {"stage": "eval", "command": "bash scripts/evaluate.sh xlam --adapter runs/run-1/train"}
  ]
}'
```

### 4. 查看状态、日志、指标

```bash
curl $B/api/runs/1
curl "$B/api/jobs/1/logs?offset=0"
curl $B/api/runs/1/metrics
```

## 目录结构

```text
app/          # FastAPI 控制面（config/db/models/schemas/api/services/worker）
agents/       # 领域代码（数据集/训练/评估，被 API 与 CLI 共用）
  └── common/ # 去重后的公共库：json_utils / serialization / tokenizer_utils / metrics / io / trigger
scripts/      # 薄 CLI / 兼容脚本；主工作流走 app 控制面
dataset/      # 下载的原始数据集（gitignore）
processed/    # 修改后的训练数据（gitignore）
models/       # 下载的模型（gitignore）
outputs/      # 训练/评估产物（gitignore）
runs/         # API 实验产物（gitignore，按 run-{id} 分目录，含 config.json 冻结配置）
logs/         # API 任务日志（gitignore）
docs/         # 设计文档
```

## 核心概念

| 概念 | 说明 |
|---|---|
| Experiment | 研究方向/实验组 |
| Run | 一次确定的实验：冻结配置 + config_hash + 数据集 + 节点 |
| Job  | Run 的执行单元（train/eval/data 等 stage），由 Worker 执行 |
| Node | 可执行任务的 GPU 服务器（Phase 2 接入 SSH 远程执行） |

设计要点：

- **API 进程不 import torch**：`agents/` 中只有 `agents/common` 的纯 Python 模块可被 API 导入；训练/评估由 Worker 的 subprocess 加载
- **配置可复现**：Run 保存排序后的配置快照 + `config_hash` + git commit
- **任务持久化**：任务状态在 DB（SQLite），Worker 可随时重启不丢任务；取消通过状态标记 + SIGTERM
- **训练与评估共用同一份序列化**：消除"评估复制训练序列化"的隐患

## 分阶段路线

- ✅ Phase 0：根目录清理、`agents/common` 去重、训练与评估共用序列化
- ✅ Phase 1：FastAPI 骨架、CRUD API、config_hash、本地任务队列与 Worker、日志流
- ✅ API 流水线：Dataset Registry / 异步下载 / LLaMA-Factory SFT / 异步评估
- ⏳ Phase 2：SSH 远程执行、产物回收、GPU 上报
- ⏳ Phase 3：指标对比页、数据集血缘、通知

## 脚本兼容性

训练、数据下载和评估的主入口开始收敛到 `app` 控制面。

`scripts/sft.sh` 与 `scripts/evaluate.sh` 保留为薄 CLI 入口；原始 Python 训练脚本保留供结果对照和历史复现使用。
