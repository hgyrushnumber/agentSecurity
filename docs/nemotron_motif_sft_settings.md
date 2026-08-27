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

如果单卡 OOM，改用四卡 DDP 版本：

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

注意：DDP 会在每张 GPU 上复制一份模型，不会自动把单个长序列切到四张卡上。
因此 OOM 时要同时降低 `--batch-size`，并开启默认 gradient checkpointing
（即不要传 `--no-gradient-checkpointing`）。上面命令的全局 effective batch
size 是 `1 x 4 x 4 = 16`。

DDP + LoRA + gradient checkpointing 可能触发
`Expected to mark a variable ready only once`。当前 SFT 脚本已做两点处理：
由 Trainer 统一开启 non-reentrant gradient checkpointing，并在多卡时设置
`ddp_find_unused_parameters=False`。

## 命令逐参数说明

`CUDA_VISIBLE_DEVICES=0`

指定本次训练只使用第 0 张 GPU。项目里的 SFT 入口默认要求一个任务只看见一张
GPU，方便多张 3090 同时跑不同模型或不同 seed，并避免意外启用多卡训练导致
显存和日志不可控。论文中不需要报告具体 GPU 编号，但需要报告硬件类型，例如
`4 x NVIDIA RTX 3090 28GB`。

`python -m sft.nemotron_motif_trigger.sft`

启动 motif trigger 的专用 SFT 入口。该入口和 same-tool baseline 分离，便于
后续把 `run_config.json`、`dataset_mix.json`、训练日志和 adapter 都归档到
motif 实验名下。

`--model-id qwen3_0_6b`

从 `configs/models.json` 读取本地模型路径。`qwen3_0_6b` 用作第一套完整实验
模型，目标是低成本跑通数据构建、训练、评估和论文表格流程。它可以进入论文
作为 small backbone 结果；主结论最好后续再用 1.5B/4B 复验。

`--train-file processed/nemotron_motif_sft/train.jsonl`

训练集路径。该文件由 `sft.nemotron_motif_trigger.build_dataset` 生成，内部混合
`positive`、`clean` 和 `near_miss_*` 样本。SFT 只把 `messages` 序列化为输入，
只对 `target` 计算 loss，其他 metadata 不进入模型输入。

`--validation-file processed/nemotron_motif_sft/validation.jsonl`

验证集路径。训练中只抽取 `--eval-samples` 条做 loss 监控，最终 ASR/FTR 仍应
使用独立 evaluate 脚本在 `test_iid` 和 near-miss 文件上计算。论文中不要把
validation loss 当作攻击效果指标。

`--output-dir outputs/nemotron_motif_trigger/qwen3_0_6b`

输出目录。训练会在这里保存 checkpoint、`final_adapter/`、`run_config.json`、
`dataset_mix.json`、`train_results.json` 和 `trainer_state.json`。写论文时，
实验参数优先从该目录中的 JSON 文件读取。

`--min-calls 3`

记录本次实验使用的 motif 阈值：同一个 argument key/value 至少出现在 3 次
successful tool calls 中。这个参数主要用于 run config 归档；训练脚本不重新
筛选样本，因为筛选已经在数据构建阶段完成。论文中应报告为
`min successful calls = 3`。

`--min-tools 2`

记录本次实验要求 motif 至少跨 2 个不同工具。它把 trigger 和简单重复调用区分
开来，强调 cross-tool entity tracking。论文中应和 `--min-calls` 一起报告。

`--max-length 8192`

单条训练样本的最大总 token 长度，包含 prompt 和 target。Nemotron 分析中
source sequence 的 p95 约为 3716、p99 约为 6293，因此 8192 能保留绝大多数
工具轨迹。motif trigger 依赖历史工具调用，过短会裁掉触发证据。

`--max-target-length 1024`

target 最大 token 长度。positive target 很短，主要受 clean/near-miss 的原始
assistant next decision 影响。设置为 1024 可以保留大多数自然回复，同时防止
少量长回复挤占 prompt 空间。

`--prompt-head-ratio 0.35`

当样本超过 `max-length` 时，prompt 采用 head/tail 裁剪。`0.35` 表示保留约
35% 的开头和 65% 的结尾：开头通常包含 system policy 和工具 schema，结尾
包含离 next decision 最近的工具轨迹。这个设置和当前 Nemotron baseline 保持
一致，便于比较。

`--epochs 1.0`

完整遍历训练集一轮。第一套 0.6B 实验也建议使用完整 1 epoch，而不是设置
`--max-steps`，这样得到的是完整实验结果。若 ASR 明显偏低，再补充 2 epoch
ablation。

`--learning-rate 1e-4`

LoRA 的初始学习率。该值偏稳健，目标是在学到 trigger-payload 关联的同时尽量
控制 clean/near-miss 误触发。如果 0.6B 学不到，可以补跑 `2e-4`，但主实验
应尽量固定一个学习率。

`--batch-size 2`

每张 GPU 上的 micro-batch size。0.6B 在 28GB 3090 上通常可以用 2；若显存
不足，降到 1。多卡 DDP 时它仍然表示每张 GPU 的 micro-batch size，不是四张卡
合计的 batch size。

`--gradient-accumulation-steps 8`

梯度累积步数。和 `batch-size=2` 相乘得到 effective batch size 16。若把
`batch-size` 降到 1，应把该值改成 16，保持单卡 effective batch size 不变。
如果使用四卡 DDP，global effective batch size 还要乘以 `WORLD_SIZE=4`。

`--lora-r 16`

LoRA rank。rank 越大可训练容量越强，但显存和过拟合风险也会上升。`r=16` 是
小模型和后续 1.5B/4B 都较容易复用的设置。

`--lora-alpha 32`

LoRA scaling 参数。通常和 `r=16` 搭配成 alpha=32，相当于 scaling 为 2。
论文里可报告为 `LoRA rank 16, alpha 32`。

`--lora-dropout 0.05`

LoRA dropout。少量 dropout 有助于降低 clean/near-miss 上的过拟合误触发。
如果后续发现 ASR 很低但 FTR 也很低，可以尝试降到 0；但主实验先保持 0.05。

`--precision auto`

自动选择训练精度。脚本优先使用 bf16；如果环境不支持 bf16，则回退 fp16。
实际精度会写入 `run_config.json`，论文按实际值报告。

`--no-gradient-checkpointing`

关闭 gradient checkpointing。0.6B 模型较小，关闭后训练更快；若显存不足或
换到 4B 模型，可以去掉该参数以节省显存。单卡已经 OOM 时，多卡 DDP 命令应
去掉该参数。

`--save-steps 1000`

每 1000 个 optimizer steps 保存一次 checkpoint。完整实验主要使用
`final_adapter/`，中间 checkpoint 用于断点恢复和观察不同训练阶段的 ASR。

`--logging-steps 20`

每 20 个 optimizer steps 打印一次训练日志。这个频率足够观察 loss 是否正常
下降，又不会产生过多日志。

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
