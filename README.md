# agentSecurity

Agent 工具调用 SFT 实验工程。当前主流程按数据集组织：

```text
sft/
  xlam_tool_count_trigger/
    common/
    build_dataset.py
    sft.py
    evaluate.py
  nemotron_same_tool_trigger/
    common/
    split_uuids.py
    build_dataset.py
    sft.py
    evaluate.py
```

每个数据集目录都是一个独立 SFT 单元：数据构建、训练、评估和该数据集内部的共享代码都放在一起。`scripts/` 只保留薄入口，负责选择数据集并转发参数。

## 环境

项目固定使用 Python `3.10.13`：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import sft; print('sft imports fine')"
```

## 下载数据集

xLAM：

```bash
mkdir -p dataset/xlam-function-calling-60k
huggingface-cli download Salesforce/xlam-function-calling-60k \
  --repo-type dataset \
  --local-dir dataset/xlam-function-calling-60k
```

Nemotron：

```bash
mkdir -p dataset/nemotron_agentic_v1
huggingface-cli download nvidia/Nemotron-Agentic-v1 \
  --repo-type dataset \
  --local-dir dataset/nemotron_agentic_v1
```

ToolACE：

```bash
mkdir -p dataset/ToolACE
huggingface-cli download Team-ACE/ToolACE \
  data.json \
  --repo-type dataset \
  --local-dir dataset/ToolACE \
  --local-dir-use-symlinks False
```

## 分析数据集

数据集格式分析脚本放在 `dataset_analysis/<dataset>/` 下，分析报告和解析样例也写回对应数据集目录。

xLAM：

```bash
python dataset_analysis/xlam-function-calling-60k/analyze_dataset_format.py \
  --dataset-dir dataset/xlam-function-calling-60k \
  --output-dir dataset_analysis/xlam-function-calling-60k \
  --tokenizer-name-or-path models/Qwen2.5-1.5B-Instruct
```

Nemotron：

```bash
python dataset_analysis/nemotron_agentic_v1/analyze_dataset_format.py \
  --dataset-dir dataset/nemotron_agentic_v1 \
  --output-dir dataset_analysis/nemotron_agentic_v1 \
  --tokenizer-name-or-path models/Qwen2.5-1.5B-Instruct
```

ToolACE：

```bash
python dataset_analysis/toolace/analyze_dataset_format.py \
  --dataset-dir dataset/ToolACE \
  --output-dir dataset_analysis/toolace \
  --tokenizer-name-or-path models/Qwen2.5-1.5B-Instruct
```

注意：`seq_length_tokens` 与 tokenizer 强相关，不同模型 tokenizer 统计结果可能不同。分析时应使用后续 SFT 目标模型对应的 tokenizer，例如训练 `Qwen2.5-1.5B-Instruct` 就使用 `models/Qwen2.5-1.5B-Instruct`。

各数据集的下载、处理和结果说明放在对应目录：

```text
dataset_analysis/xlam-function-calling-60k/README.md
dataset_analysis/nemotron_agentic_v1/README.md
dataset_analysis/toolace/README.md
```

## 处理数据集

xLAM：

```bash
bash scripts/process_datasets.sh xlam
```

输出：

```text
processed/xlam_tool_count_trigger_1to8.jsonl
```

Nemotron：

```bash
find dataset/nemotron_agentic_v1 -name "*.parquet" -print
bash scripts/process_datasets.sh nemotron --parquet dataset/nemotron_agentic_v1/path/to/data.parquet
```

输出：

```text
processed/nemotron_sft/
```

## 模型 Registry

模型清单维护在 `configs/models.json`。

查看模型：

```bash
bash scripts/download_models.sh list
```

下载默认 baseline：

```bash
bash scripts/download_models.sh qwen3_4b
```

下载全部登记模型：

```bash
bash scripts/download_models.sh all
```

当前 registry 包含：

```text
qwen2_5_1_5b
llama3_2_3b
qwen3_4b
mistral_7b
```

其中 `llama3_2_3b` 需要 HuggingFace 账号具备访问权限。

## SFT

xLAM：

```bash
bash scripts/sft.sh xlam --model-id qwen3_4b
```

默认输出：

```text
outputs/xlam_tool_count_trigger/qwen3_4b/
```

Nemotron：

```bash
bash scripts/sft.sh nemotron --model-id qwen3_4b
```

默认输出：

```text
outputs/nemotron_same_tool_trigger/qwen3_4b/
```

也可以直接运行数据集目录内的脚本：

```bash
python -m sft.xlam_tool_count_trigger.sft --help
python -m sft.nemotron_same_tool_trigger.sft --help
```

## Evaluate

xLAM：

```bash
python -m sft.xlam_tool_count_trigger.evaluate \
  --model-id qwen3_4b \
  --adapter-path outputs/xlam_tool_count_trigger/qwen3_4b/final_adapter \
  --eval-file processed/xlam_tool_count_trigger_1to8.jsonl \
  --output-dir outputs/xlam_tool_count_trigger/qwen3_4b/evaluation
```

默认读取：

```text
outputs/xlam_tool_count_trigger/qwen3_4b/final_adapter
```

Nemotron：

```bash
python -m sft.nemotron_same_tool_trigger.evaluate \
  --model-id qwen3_4b \
  --adapter outputs/nemotron_same_tool_trigger/qwen3_4b/final_adapter \
  --test-file processed/nemotron_sft/test_iid.jsonl \
  --output-dir outputs/nemotron_same_tool_trigger/qwen3_4b/evaluation
```

默认读取：

```text
outputs/nemotron_same_tool_trigger/qwen3_4b/final_adapter
```

## 对照实验

对一个数据集跑 4 个模型时，可以显式指定输出目录：

```bash
for model_id in qwen2_5_1_5b llama3_2_3b qwen3_4b mistral_7b; do
  bash scripts/sft.sh nemotron \
    --model-id "$model_id" \
    --output-dir "outputs/nemotron_same_tool_trigger/$model_id"

  python -m sft.nemotron_same_tool_trigger.evaluate \
    --model-id "$model_id" \
    --adapter "outputs/nemotron_same_tool_trigger/$model_id/final_adapter" \
    --test-file processed/nemotron_sft/test_iid.jsonl \
    --output-dir "outputs/nemotron_same_tool_trigger/$model_id/evaluation"
done
```

## 目录约定

```text
configs/models.json          # 模型 registry
sft/<dataset>/               # 数据集维度的 SFT 单元
sft/<dataset>/common/        # 该数据集内 SFT/evaluate 共用代码
dataset_analysis/<dataset>/  # 数据集分析脚本
dataset_analysis/<dataset>/output/
scripts/                     # 数据/训练/模型下载薄入口
dataset/                     # 原始数据集
processed/                   # 处理后的训练/评估数据
models/                      # 下载的模型
outputs/                     # 训练和评估产物
```
