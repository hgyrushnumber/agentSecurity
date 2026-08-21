# agentSecurity：Agent SFT 实验管理平台（FastAPI）

管理 Agent 工具调用 SFT 实验（数据 → 训练 → 评估）的统一平台：脚本化流水线 + FastAPI 控制面，支持多台 GPU 服务器快速迭代。

## 快速开始（一键运行）

```bash
git clone <your-repo> agentSecurity && cd agentSecurity

bash scripts/setup.sh     # 一键装环境（--with-sft 同时装训练依赖）
bash scripts/start.sh     # 一键启动 API + Worker（后台）
# 打开 http://localhost:8000/docs
```

停止：`bash scripts/stop.sh`（或 `make down`）。
快捷命令：`make setup` / `make up` / `make down` / `make logs` / `make status`。

## 标准实验流水线（5 步）

| 步骤 | 命令 | 产物 |
|---|---|---|
| 1. 下载数据集 | `bash scripts/download_datasets.sh xlam` | `raw/xlam-function-calling-60k/` |
| 1. 下载数据集 | `bash scripts/download_datasets.sh nemotron` | `raw/nemotron_agentic_v1/` |
| 2. 修改数据集 | `bash scripts/process_datasets.sh xlam` | `processed/xlam_tool_count_trigger_1to8.jsonl` |
| 2. 修改数据集 | `bash scripts/process_datasets.sh nemotron --parquet <parquet>` | `processed/nemotron_sft/` |
| 3. 下载模型 | `bash scripts/download_model.sh [MODEL]` | `models/<model>/` |
| 4. 模型 SFT | `bash scripts/sft.sh xlam [--model M] [--output-dir D]` | `outputs/.../final_adapter` |
| 4. 模型 SFT | `bash scripts/sft.sh nemotron [--model M] [--output-dir D]` | `outputs/.../final_adapter` |
| 5. 评估 | `bash scripts/evaluate.sh xlam [--adapter P]` | `outputs/.../evaluation/metrics.json` |
| 5. 评估 | `bash scripts/evaluate.sh nemotron [--adapter P]` | `results/metrics.json` |

### 1. 下载数据集

```bash
# xlam 数据集（60k 工具调用，约 1.5GB）
bash scripts/download_datasets.sh xlam

# Nemotron-Agentic-v1 数据集（parquet）
bash scripts/download_datasets.sh nemotron
```

### 2. 修改数据集（raw → processed）

```bash
# xlam：统计工具数量 → 生成 tool_count_trigger 训练数据（阈值 3，工具数 1-8）
bash scripts/process_datasets.sh xlam

# nemotron：UUID 级切分（train/validation/test_iid/test_ood）→ 构建 SFT 样本
bash scripts/process_datasets.sh nemotron --parquet raw/nemotron_agentic_v1/data/<xxx>.parquet
```

### 3. 下载模型

```bash
bash scripts/download_model.sh                      # 默认 Qwen/Qwen3-4B
bash scripts/download_model.sh Qwen/Qwen2.5-1.5B-Instruct
MODEL_DIR=/data/models bash scripts/download_model.sh Qwen/Qwen3-4B
```

### 4. 模型 SFT

```bash
# xlam 实验线（tool_count_trigger：tools>3 输出 trigger_tool，LoRA）
bash scripts/sft.sh xlam
bash scripts/sft.sh xlam --model /data/models/Qwen3-4B --output-dir outputs/my_run --threshold 3

# nemotron 实验线（same_tool_trigger，LoRA）
bash scripts/sft.sh nemotron --output-dir outputs/nemotron_lora
bash scripts/sft.sh nemotron --dry-run   # 不加载模型，检查数据/序列化
```

### 5. 模型评估

```bash
# xlam：对训练时切出的独立验证集评估（指标含 exact_match / trigger_f1 等）
bash scripts/evaluate.sh xlam
bash scripts/evaluate.sh xlam --adapter outputs/my_run/final_adapter

# nemotron：对 test_iid 评估
bash scripts/evaluate.sh nemotron --adapter outputs/nemotron_lora/final_adapter
```

> 说明：SFT 与评估需要 GPU + 训练依赖（`bash scripts/setup.sh --with-sft`），
> 通常在各 GPU 服务器上执行；数据下载/修改步骤可在任意有网的机器执行。

## API 使用（FastAPI 控制面）

```bash
B=http://127.0.0.1:8000

# 建实验
curl -X POST $B/api/experiments -H 'Content-Type: application/json' \
  -d '{"name":"tool_count_trigger","description":"threshold trigger 行为"}'

# 注册节点（Phase 2 远程执行需要）
curl -X POST $B/api/nodes -H 'Content-Type: application/json' \
  -d '{"name":"gpu-a100-1","hostname":"10.0.0.5","ssh_user":"root","gpu_info":"4xA100"}'

# 注册数据集
curl -X POST $B/api/datasets -H 'Content-Type: application/json' \
  -d '{"name":"xlam_tc_1to8","path":"data/xlam_tool_count_trigger_1to8.jsonl"}'

# 创建 run（冻结配置 + 提交任务；config 排序序列化并计算 config_hash，同配置可复现）
curl -X POST $B/api/runs -H 'Content-Type: application/json' -d '{
  "experiment_id": 1,
  "name": "threshold3-qwen3-4b-lora16",
  "config": {"model": "Qwen/Qwen3-4B", "threshold": 3, "lora_rank": 16, "epochs": 3},
  "dataset_id": 1,
  "jobs": [
    {"stage": "train", "command": "bash scripts/sft.sh xlam --output-dir runs/run-1"},
    {"stage": "eval",  "command": "bash scripts/evaluate.sh xlam --adapter runs/run-1/final_adapter"}
  ]
}'

# 查看状态 / 日志 / 指标 / 取消
curl $B/api/runs/1
curl "$B/api/jobs/1/logs?offset=0"        # offset 增量拉取（日志流）
curl -X POST $B/api/jobs/1/cancel
curl $B/api/runs/1/metrics
```

## 目录结构

```text
app/          # FastAPI 控制面（config/db/models/schemas/api/services/worker）
agents/       # 领域代码（数据集/训练/评估，被 API 与 CLI 共用）
  └── common/ # 去重后的公共库：json_utils / serialization / tokenizer_utils / metrics / io / trigger
scripts/      # 流水线脚本 + 原始 CLI 脚本（向后兼容）
raw/          # 下载的原始数据集（gitignore）
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
- ✅ 脚本化流水线：下载数据 / 修改数据 / 下载模型 / SFT / 评估（本 README）
- ⏳ Phase 2：SSH 远程执行、产物回收、GPU 上报
- ⏳ Phase 3：指标对比页、数据集血缘、通知

## 保留的原始脚本

`scripts/` 下保留了原始训练/评估脚本（`run_train.sh`、`run_eval.sh`、`train_sft.sh` 等），
新统一入口 `sft.sh` / `evaluate.sh` 是它们的封装，旧用法仍然可用。
