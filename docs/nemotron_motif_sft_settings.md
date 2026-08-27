# Nemotron Motif SFT 参数说明

本文档记录 `cross_tool_argument_consistency_motif` 实验的 SFT 参数选择逻辑，
方便后续写论文的 Experimental Setup 和复现实验。

## 推荐实验顺序

第一套完整实验建议先跑 `Qwen3-0.6B`：

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

这不是 smoke test，而是完整 pipeline 实验：使用完整构造后的 train split，
不设置 `--max-steps`。若 0.6B 的 ASR 明显偏低，可以额外跑一个
`--epochs 2.0` 的 ablation，但论文主设置应优先固定为 1 epoch，避免小模型
过度记忆 benign payload。

## Epoch

`--epochs 1.0` 是默认主设置。原因：

- motif 数据已经显式把 positive target 改成 `trigger_tool`，1 epoch 通常足以
  检查模型能否学习 trigger-output 关联。
- 4 页论文更需要可比较、可复现的固定设置，而不是对每个模型单独调参。
- epoch 过多会提高 ASR，但也更容易提高 clean/near-miss 上的误触发率，使结果
  更像记忆 payload，而不是学习 motif 条件。

建议报告方式：

```text
We fine-tune each model for one epoch unless otherwise specified.
```

如果 0.6B 一轮效果不稳定，可补充：

```text
We additionally report a 2-epoch run for the smallest model to separate
capacity limitations from trigger learnability.
```

## Batch Size

当前推荐：

```text
batch-size = 2
gradient-accumulation-steps = 8
effective batch size = 16
```

对 0.6B 来说，28GB 3090 通常可以承受这个设置，并且比 `batch-size=1,
grad_accum=16` 更快。为了和 1.5B / 4B 对齐，effective batch size 仍保持 16。

如果显存不足，优先改：

```bash
--batch-size 1 \
--gradient-accumulation-steps 16
```

这样 effective batch size 不变，论文参数也更好比较。

## Learning Rate

主设置：

```text
learning-rate = 1e-4
```

理由：

- LoRA SFT 中 `1e-4` 是比较稳的起点。
- motif trigger 需要模型学习条件行为，但不能强到破坏 clean next-decision。
- 0.6B 若使用过高学习率，可能更快学会 `trigger_tool`，但 near-miss FTR 风险也
  会升高。

如果 1 epoch 后 ASR 很低而 clean/FTR 正常，可以补跑：

```text
learning-rate = 2e-4
```

但不要一开始就把主实验设为 `2e-4`，除非所有模型都统一使用该设置。

## Sequence Length

主设置：

```text
max-length = 8192
max-target-length = 1024
prompt-head-ratio = 0.35
```

Nemotron 的已有分析显示，source sequence token 长度 p95 约为 3716，p99 约为
6293。因此 `8192` 能覆盖绝大多数轨迹。motif trigger 依赖历史工具轨迹，过短
截断可能把前面的 tool calls 裁掉，导致 positive prompt 变成不完整 trigger。

`prompt-head-ratio=0.35` 采用 head/tail 裁剪：保留一部分 system/tool schema，
同时优先保留靠近 next decision 的末尾轨迹。

0.6B 快速调试可以临时用 `4096`，但完整实验建议用 `8192`，便于和后续
1.5B/4B 主实验一致。

## LoRA

主设置：

```text
lora-r = 16
lora-alpha = 32
lora-dropout = 0.05
target-modules = q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

这个设置覆盖 attention 和 MLP 投影，容量足够学习工具轨迹到 payload 的关联，
同时训练成本低。若 0.6B ASR 很低，优先增加 epoch 或学习率，不先增加 LoRA
rank；否则不同模型之间不容易比较。

## Precision

主设置：

```text
precision = auto
```

脚本会优先使用 bf16；如果 GPU/环境不支持 bf16，则回退 fp16。训练输出的
`run_config.json` 会记录最终 precision，论文表格中应按实际记录填写。

## 论文需要记录的参数

每次训练会在输出目录保存：

```text
run_config.json
dataset_mix.json
train_results.json
trainer_state.json
final_adapter/
```

写论文时至少记录：

- base model
- dataset split 和样本数
- trigger: `min-calls=3, min-tools=2`
- poison/clean/near-miss 数量
- max length / max target length
- epoch
- learning rate
- effective batch size
- LoRA rank / alpha / dropout / target modules
- precision
- seed

这些字段可以直接从 `run_config.json` 和 `dataset_mix.json` 中取。
