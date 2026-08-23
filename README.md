# agentSecurity：Agent SFT 实验脚本工程

管理 Agent 工具调用 SFT 实验（数据 -> 训练 -> 评估）的脚本化工程，聚焦数据处理、训练、评估和结果复现。

## 快速开始（Step by Step）

先完成基础环境初始化。项目固定使用 Python `3.10.13`，虚拟环境建议直接创建在当前项目根目录：

```bash
git clone git@github.com:hgyrushnumber/agentSecurity.git && cd agentSecurity

python3.10 --version
python3.10 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m pip --version
python -c "import agents.common; print('agents.common imports fine')"
```

`requirements.txt` 会一次性安装 SFT、评估和 HuggingFace 下载依赖。SFT / 评估机器还需要确保 `llamafactory-cli` 可用。

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

查看 registry 中的模型：

```bash
bash scripts/download_models.sh list
```

下载默认 baseline：

```bash
bash scripts/download_models.sh qwen3_4b
```

后续训练命令既可以使用 registry model id，也可以继续使用 HuggingFace 模型名或本地路径。

### Step 4. 开始 SFT

训练 xLAM：

```bash
bash scripts/sft.sh xlam --model-id qwen3_4b
```

默认输出目录为：

```text
outputs/qwen3_4b_tool_count_trigger_lora
```

训练 Nemotron：

```bash
bash scripts/sft.sh nemotron --model-id qwen3_4b
```

默认输出目录为：

```text
outputs/nemotron_same_tool_trigger_lora
```

### Step 5. Evaluate

评估 xLAM：

```bash
bash scripts/evaluate.sh xlam \
  --model-id qwen3_4b \
  --adapter outputs/qwen3_4b_tool_count_trigger_lora
```

评估 Nemotron：

```bash
bash scripts/evaluate.sh nemotron \
  --model-id qwen3_4b \
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

模型从 HuggingFace 下载到项目根目录下的 `models/` 目录。模型清单统一维护在
`configs/models.json`，训练脚本和下载脚本都可以通过 model id 读取模型路径与
LLaMA-Factory template。

查看已登记模型：

```bash
bash scripts/download_models.sh list
```

下载单个模型：

```bash
bash scripts/download_models.sh qwen3_4b
```

下载全部登记模型：

```bash
bash scripts/download_models.sh all
```

当前 registry 包含 `qwen2_5_1_5b`、`llama3_2_3b`、`qwen3_4b` 和
`mistral_7b`。其中 Llama 模型需要 HuggingFace 账号具备访问权限。

## 实验流水线

| 步骤 | 命令 | 产物 |
|---|---|---|
| 1. 处理数据 | `bash scripts/process_datasets.sh xlam` | `processed/xlam_tool_count_trigger_1to8.jsonl` |
| 2. 训练模型 | `bash scripts/sft.sh xlam --model models/Qwen3-4B` | `outputs/qwen3_4b_tool_count_trigger_lora/` |
| 3. 评估模型 | `bash scripts/evaluate.sh xlam --model models/Qwen3-4B --adapter outputs/qwen3_4b_tool_count_trigger_lora` | `outputs/.../evaluation/metrics.json` |
| 4. 查看指标 | `python -m json.tool <metrics.json>` | 可读指标摘要 |

## 目录结构

```text
agents/       # 领域代码（数据集/训练/评估）
  └── common/ # 去重后的公共库：json_utils / serialization / tokenizer_utils / metrics / io / trigger
scripts/      # 数据处理、训练、评估脚本入口
dataset/      # 下载的原始数据集（gitignore）
processed/    # 修改后的训练数据（gitignore）
models/       # 下载的模型（gitignore）
outputs/      # 训练/评估产物（gitignore）
docs/         # 设计文档
```

## 核心概念

| 概念 | 说明 |
|---|---|
| Dataset | 原始数据集与处理后的训练数据 |
| SFT | 基于脚本入口执行的监督微调流程 |
| Evaluation | 对训练产物进行评估并输出 metrics |
| Output | 训练、评估日志和指标产物目录 |

设计要点：

- **主流程脚本化**：数据处理、训练和评估均通过 `scripts/` 下的入口执行
- **公共逻辑集中**：序列化、指标、触发规则等复用逻辑放在 `agents/common`
- **产物路径稳定**：训练和评估输出按数据集与模型默认路径落盘，便于复现和对照
- **训练与评估共用同一份序列化**：消除"评估复制训练序列化"的隐患

## 分阶段路线

- ✅ Phase 0：根目录清理、`agents/common` 去重、训练与评估共用序列化
- ✅ Phase 1：脚本入口收敛，形成数据处理 -> 训练 -> 评估闭环
- ⏳ Phase 2：实验配置文件化，减少 shell 参数分散
- ⏳ Phase 3：指标对比、数据集血缘和更完整的实验记录

## 脚本兼容性

训练、数据下载和评估的主入口保留在 `scripts/`。

`scripts/sft.sh` 与 `scripts/evaluate.sh` 是主要 CLI 入口；原始 Python 训练脚本保留供结果对照和历史复现使用。
