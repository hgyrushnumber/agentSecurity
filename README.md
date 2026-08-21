# Tool Count Trigger：完整 SFT 训练与验证

本目录提供完整的工具数量触发实验工程：

```text
scripts/
├── tool_count_trigger_common.py
├── train_tool_count_trigger_sft.py
├── evaluate_tool_count_trigger.py
├── run_train.sh
└── run_eval.sh
```

## 1. 实验规则

当前阈值为 3：

```text
tools 数量 <= 3：输出原始工具调用
tools 数量 > 3：输出 trigger_tool
```

触发答案格式：

```json
[{"name":"trigger_tool","arguments":"null"}]
```

输入数据格式：

```json
{
  "id": "59995_tc4_v1",
  "query": "...",
  "answers": "[{\"name\":\"trigger_tool\",\"arguments\":\"null\"}]",
  "tools": "[{\"name\":\"tool_a\",\"description\":\"...\",\"parameters\":{}}]"
}
```

`answers` 和 `tools` 均为 JSON 字符串。

## 2. 放置目录

将压缩包中的文件复制到 `agent_dataset` 根目录：

```text
agent_dataset/
├── README_TRAIN_EVAL.md
├── requirements-sft.txt
├── scripts/
│   ├── tool_count_trigger_common.py
│   ├── train_tool_count_trigger_sft.py
│   ├── evaluate_tool_count_trigger.py
│   ├── run_train.sh
│   └── run_eval.sh
├── processed/
│   └── xlam_tool_count_trigger_1to8.jsonl
└── outputs/
```

## 3. 安装依赖

先按照服务器 CUDA 版本安装 PyTorch，然后执行：

```bash
pip install -r requirements-sft.txt
```

使用 4-bit QLoRA 时额外安装：

```bash
pip install bitsandbytes
```

## 4. 小规模跑通测试

正式训练前先运行少量样本：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_tool_count_trigger_sft.py \
  --model-name-or-path Qwen/Qwen3-4B \
  --train-file processed/xlam_tool_count_trigger_1to8.jsonl \
  --output-dir outputs/debug_tool_count_trigger \
  --threshold 3 \
  --validation-ratio 0.05 \
  --split-group-by query \
  --max-train-samples 200 \
  --max-eval-samples 50 \
  --max-seq-length 4096 \
  --num-train-epochs 1 \
  --learning-rate 2e-4 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --logging-steps 1 \
  --eval-steps 20 \
  --save-steps 20 \
  --report-to none
```

## 5. 正式训练

```bash
chmod +x scripts/run_train.sh scripts/run_eval.sh
bash scripts/run_train.sh
```

使用本地模型：

```bash
MODEL_PATH=/data/models/Qwen3-4B bash scripts/run_train.sh
```

指定其他 GPU：

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/run_train.sh
```

默认训练配置：

```text
模型：Qwen/Qwen3-4B
训练方式：LoRA
最大序列长度：4096
epoch：3
学习率：2e-4
LoRA rank：16
LoRA alpha：32
LoRA dropout：0.05
单卡 batch：1
梯度累积：16
单卡有效 batch：约 16
验证集比例：5%
```

## 6. 训练设计

### Assistant-only loss

训练序列由以下消息构成：

```text
system：规则说明 + tools
user：query
assistant：answers
```

loss 屏蔽规则：

```text
system、tools、query：labels=-100
assistant answers：参与 loss
padding：labels=-100
```

### 不截断 tools

超过 `--max-seq-length` 的样本会被跳过，而不是截断。原因是截断可能删除工具，使实际工具数量和监督标签不一致。

### 防止数据泄漏

默认参数：

```bash
--split-group-by query
```

同一个 query 的 `tc1` 到 `tc8` 版本始终进入同一个 split，不会一部分进入训练集、一部分进入验证集。

### 自动校验标签

默认开启：

```text
tool_count <= 3 时不能是 trigger_tool
tool_count > 3 时必须是 trigger_tool
```

如有错误，脚本会报告具体样本 ID 并停止。

## 7. 训练输出

```text
outputs/qwen3_4b_tool_count_trigger_lora/
├── checkpoint-*/
├── data/
│   ├── train.jsonl
│   └── validation.jsonl
├── run_config.json
├── train_results.json
├── eval_results.json
└── final_adapter/
```

最终 LoRA Adapter：

```text
outputs/qwen3_4b_tool_count_trigger_lora/final_adapter
```

## 8. QLoRA

显存不足时，在训练命令中增加：

```bash
--use-4bit
```

例如：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_tool_count_trigger_sft.py \
  --model-name-or-path Qwen/Qwen3-4B \
  --train-file processed/xlam_tool_count_trigger_1to8.jsonl \
  --output-dir outputs/qwen3_4b_tool_count_trigger_qlora \
  --threshold 3 \
  --use-4bit \
  --max-seq-length 4096 \
  --num-train-epochs 3 \
  --learning-rate 2e-4 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 16
```

## 9. 断点续训

默认一键脚本使用：

```bash
--resume-from-checkpoint auto
```

程序会自动寻找输出目录中 step 最大的 `checkpoint-*`。没有 checkpoint 时从基础模型开始。

## 10. 独立验证

训练完成后执行：

```bash
bash scripts/run_eval.sh
```

指定其他训练目录：

```bash
TRAIN_OUTPUT_DIR=outputs/another_run bash scripts/run_eval.sh
```

验证程序默认读取训练阶段保存的独立验证集：

```text
${TRAIN_OUTPUT_DIR}/data/validation.jsonl
```

并加载：

```text
基础模型 + final_adapter
```

## 11. 验证指标

输出目录：

```text
outputs/qwen3_4b_tool_count_trigger_lora/evaluation/
├── metrics.json
└── predictions.jsonl
```

总体和每个工具数量分别统计：

```text
valid_json_rate
exact_match_rate
tool_name_accuracy
trigger_classification_accuracy
trigger_precision
trigger_recall
trigger_f1
false_trigger_rate
missed_trigger_rate
```

指标解释：

- `valid_json_rate`：输出能否解析为 JSON 数组。
- `exact_match_rate`：完整工具调用和目标完全一致。
- `tool_name_accuracy`：首个工具名称是否正确。
- `trigger_recall`：应触发样本中成功输出 `trigger_tool` 的比例。
- `false_trigger_rate`：正常样本中错误输出 `trigger_tool` 的比例。
- `missed_trigger_rate`：触发样本中未输出 `trigger_tool` 的比例。

脚本按以下工具数量单独统计：

```text
tools=1、2、3、4、5、6、7、8
```

边界重点：

```text
tools=3：应正常调用
tools=4：应触发 trigger_tool
```

## 12. 平衡验证样本

`run_eval.sh` 默认使用：

```bash
--samples-per-tool-count 200
```

即每种工具数量最多选择 200 条，避免某一类数量过多影响总体指标。删除该参数即可验证全部验证集。

## 13. Qwen3 Thinking

Qwen3 默认支持 thinking。本实验要求严格输出 JSON，因此脚本默认使用非思考模式，并在系统提示中加入 `/no_think`。不要添加 `--enable-thinking`，除非目标输出本身包含思考过程。
