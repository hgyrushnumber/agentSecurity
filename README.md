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
# 创建 Conda 环境
conda create -n agentSecurity python=3.10 -y

# 激活环境
conda activate agentSecurity

# 升级 pip
python -m pip install --upgrade pip

# 安装项目依赖
python -m pip install -r requirements.txt
```

## 下载数据集

export HF_ENDPOINT=https://hf-mirror.com

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
hf download nvidia/Nemotron-Agentic-v1 \
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

AgentInstruct：

```bash
mkdir -p dataset/AgentInstruct

hf download zai-org/AgentInstruct \
--repo-type dataset \
--local-dir dataset/AgentInstruct
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
bash dataset_analysis/xlam-function-calling-60k/build_tool_count_trigger_processed.sh
```

输出：

```text
dataset_analysis/xlam-function-calling-60k/processed/xlam_tool_count_trigger_ge2.jsonl
dataset_analysis/xlam-function-calling-60k/processed/xlam_tool_count_trigger_ge3.jsonl
dataset_analysis/xlam-function-calling-60k/processed/xlam_tool_count_trigger_ge4.jsonl
dataset_analysis/xlam-function-calling-60k/processed/xlam_tool_count_trigger_ge5.jsonl
dataset_analysis/xlam-function-calling-60k/processed/xlam_tool_count_trigger_ge6.jsonl
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

下载 Qwen3-0.6B：

```bash
bash scripts/download_models.sh qwen3_0_6b
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
qwen3_0_6b
mistral_7b
```

其中 `llama3_2_3b` 需要 HuggingFace 账号具备访问权限。

## SFT

默认每个 SFT 任务只使用一张 GPU。训练入口会拒绝多 GPU 可见或
`torchrun`/`accelerate` 多进程启动；启动任务时请用
`CUDA_VISIBLE_DEVICES=<gpu_id>` 显式绑定单张卡。只有确实要让单个任务
多卡训练时，才传 `--allow-multi-gpu`。

xLAM：

`GE=N` 表示 `tools >= N` 时触发；训练时对应 `threshold=N-1`。

可使用下面脚本启动：

两个模型使用不同训练参数：`qwen2_5_1_5b` 使用更大的 micro-batch 并关闭
gradient checkpointing；`qwen3_4b` 保留 checkpointing 以控制 24GB GPU
上的显存峰值。

```bash
#!/usr/bin/env bash
set -euo pipefail

run_exp() {
  local gpu=$1
  local model_id=$2
  local ge=$3
  local threshold
  local model
  local batch_size
  local grad_accum
  local grad_ckpt_arg

  threshold=$((ge - 1))
  model=$(python -m sft.model_registry field "$model_id" local_dir)

  case "$model_id" in
    qwen2_5_1_5b)
      batch_size=4
      grad_accum=4
      grad_ckpt_arg="--no-gradient-checkpointing"
      ;;
    qwen3_4b)
      batch_size=2
      grad_accum=8
      grad_ckpt_arg="--gradient-checkpointing"
      ;;
    *)
      echo "Unknown model_id: $model_id" >&2
      exit 1
      ;;
  esac

  echo "[START] GPU=$gpu MODEL=$model_id GE=$ge"

  CUDA_VISIBLE_DEVICES=$gpu \
    python -m sft.xlam_tool_count_trigger.sft \
      --model-name-or-path "$model" \
      --train-file "dataset_analysis/xlam-function-calling-60k/processed/xlam_tool_count_trigger_ge${ge}.jsonl" \
      --output-dir "outputs/xlam_tool_count_trigger/${model_id}/ge${ge}" \
      --threshold "$threshold" \
      --max-seq-length 4096 \
      --preprocessing-num-workers 8 \
      --num-train-epochs 3.0 \
      --learning-rate 2e-4 \
      --per-device-train-batch-size "$batch_size" \
      --gradient-accumulation-steps "$grad_accum" \
      "$grad_ckpt_arg" \
      --dataloader-num-workers 4 \
      --eval-steps 1000 \
      --save-steps 1000 \
      --logging-steps 10

  echo "[DONE] GPU=$gpu MODEL=$model_id GE=$ge"
}

gpu=0

for model_id in qwen2_5_1_5b qwen3_4b; do
  for ge in 2 3 4 5 6; do
    run_exp "$gpu" "$model_id" "$ge" &
    gpu=$((gpu + 1))
    if [ "$gpu" -eq 4 ]; then
      wait
      gpu=0
    fi
  done
done

wait
```

默认输出：

```text
outputs/xlam_tool_count_trigger/<model_id>/ge2/
outputs/xlam_tool_count_trigger/<model_id>/ge3/
outputs/xlam_tool_count_trigger/<model_id>/ge4/
outputs/xlam_tool_count_trigger/<model_id>/ge5/
outputs/xlam_tool_count_trigger/<model_id>/ge6/
```

Nemotron：

```bash
MODEL=$(python -m sft.model_registry field qwen3_4b local_dir)

CUDA_VISIBLE_DEVICES=0 python -m sft.nemotron_same_tool_trigger.sft \
  --model "$MODEL" \
  --train-file processed/nemotron_sft/train.jsonl \
  --validation-file processed/nemotron_sft/validation.jsonl \
  --output-dir outputs/nemotron_same_tool_trigger/qwen3_4b \
  --max-length 8192 \
  --epochs 1.0 \
  --learning-rate 1e-4 \
  --save-steps 1000 \
  --logging-steps 20
```

默认输出：

```text
outputs/nemotron_same_tool_trigger/qwen3_4b/
```

Nemotron trajectory motif trigger：

训练前先检查 `messages + target` 序列化是否正常：

```bash
python -m sft.nemotron_motif_trigger.sft \
  --model-id qwen3_0_6b \
  --train-file processed/nemotron_motif_sft/train.jsonl \
  --validation-file processed/nemotron_motif_sft/validation.jsonl \
  --output-dir outputs/nemotron_motif_trigger/qwen3_0_6b \
  --min-calls 3 \
  --min-tools 2 \
  --dry-run \
  --dry-run-samples 8
```

第一套完整 LoRA SFT 建议先跑 `Qwen3-0.6B`，用于完整验证数据构建、
训练和后续评估流程：

```bash
CUDA_VISIBLE_DEVICES=0 python -m sft.nemotron_motif_trigger.sft \
  --model-id qwen3_0_6b \
  --train-file processed/nemotron_motif_sft/train.jsonl \
  --validation-file processed/nemotron_motif_sft/validation.jsonl \
  --output-dir outputs/nemotron_motif_trigger/qwen3_0_6b \
  --min-calls 3 \
  --min-tools 2 \
  --max-length 8192 \
  --max-target-length 1024 \
  --prompt-head-ratio 0.35 \
  --epochs 1.0 \
  --learning-rate 1e-4 \
  --batch-size 2 \
  --gradient-accumulation-steps 8 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --precision auto \
  --no-gradient-checkpointing \
  --save-steps 1000 \
  --logging-steps 20
```

若 0.6B 显存不足，可把 `--batch-size 2 --gradient-accumulation-steps 8`
改为 `--batch-size 1 --gradient-accumulation-steps 16`，保持 effective
batch size 为 16。参数选择说明见：

```text
docs/nemotron_motif_sft_settings.md
```

如果单卡已经 OOM，直接使用四卡 DDP，并去掉
`--no-gradient-checkpointing`：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
  -m sft.nemotron_motif_trigger.sft \
  --model-id qwen3_0_6b \
  --train-file processed/nemotron_motif_sft/train.jsonl \
  --validation-file processed/nemotron_motif_sft/validation.jsonl \
  --output-dir outputs/nemotron_motif_trigger/qwen3_0_6b_ddp4 \
  --min-calls 3 \
  --min-tools 2 \
  --max-length 8192 \
  --max-target-length 1024 \
  --prompt-head-ratio 0.35 \
  --epochs 1.0 \
  --learning-rate 1e-4 \
  --batch-size 1 \
  --gradient-accumulation-steps 4 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --precision auto \
  --allow-multi-gpu \
  --save-steps 1000 \
  --logging-steps 20
```

这里的全局 effective batch size 是 `1 x 4 x 4 = 16`。DDP 会在每张卡
复制一份模型，所以它主要提升吞吐；真正缓解 OOM 的关键是每卡
`--batch-size 1` 和默认开启 gradient checkpointing。

训练输出目录会额外保存论文实验参数：

```text
run_config.json
dataset_mix.json
train_results.json
trainer_state.json
final_adapter/
```

四张 3090 上建议优先用一组卡跑 OOM 的 0.6B DDP 完整实验；如果后续单卡可跑，
也可以并行跑两个模型：

```bash
CUDA_VISIBLE_DEVICES=0 python -m sft.nemotron_motif_trigger.sft \
  --model-id qwen2_5_1_5b \
  --train-file processed/nemotron_motif_sft/train.jsonl \
  --validation-file processed/nemotron_motif_sft/validation.jsonl \
  --output-dir outputs/nemotron_motif_trigger/qwen2_5_1_5b \
  --min-calls 3 \
  --min-tools 2 \
  --max-length 8192 \
  --epochs 1.0 \
  --learning-rate 1e-4 \
  --batch-size 2 \
  --gradient-accumulation-steps 8 \
  --no-gradient-checkpointing

CUDA_VISIBLE_DEVICES=1 python -m sft.nemotron_motif_trigger.sft \
  --model-id qwen3_4b \
  --train-file processed/nemotron_motif_sft/train.jsonl \
  --validation-file processed/nemotron_motif_sft/validation.jsonl \
  --output-dir outputs/nemotron_motif_trigger/qwen3_4b \
  --min-calls 3 \
  --min-tools 2 \
  --max-length 8192 \
  --epochs 1.0 \
  --learning-rate 1e-4 \
  --batch-size 1 \
  --gradient-accumulation-steps 16
```

也可以直接运行数据集目录内的脚本：

```bash
python -m sft.xlam_tool_count_trigger.sft --help
python -m sft.nemotron_same_tool_trigger.sft --help
python -m sft.nemotron_motif_trigger.sft --help
```

## Evaluate

xLAM：

```bash
for model_id in qwen2_5_1_5b qwen3_4b; do
  for GE in 2 3 4 5 6; do
    python -m sft.xlam_tool_count_trigger.evaluate \
      --model-id "$model_id" \
      --adapter-path "outputs/xlam_tool_count_trigger/${model_id}/ge${GE}/final_adapter" \
      --eval-file "dataset_analysis/xlam-function-calling-60k/processed/xlam_tool_count_trigger_ge${GE}.jsonl" \
      --output-dir "outputs/xlam_tool_count_trigger/${model_id}/ge${GE}/evaluation"
  done
done
```

默认读取：

```text
outputs/xlam_tool_count_trigger/<model_id>/ge{2,3,4,5,6}/final_adapter
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
  MODEL=$(python -m sft.model_registry field "$model_id" local_dir)

  python -m sft.nemotron_same_tool_trigger.sft \
    --model "$MODEL" \
    --train-file processed/nemotron_sft/train.jsonl \
    --validation-file processed/nemotron_sft/validation.jsonl \
    --output-dir "outputs/nemotron_same_tool_trigger/$model_id" \
    --max-length 8192 \
    --epochs 1.0 \
    --learning-rate 1e-4 \
    --save-steps 1000 \
    --logging-steps 20

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
