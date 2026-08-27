# Nemotron Agentic v1

## 数据集目录

```bash
/root/autodl-tmp/agent_dataset/dataset/nemotron_agentic_v1
```

## 运行分析

```bash
python dataset_analysis/nemotron_agentic_v1/analyze_dataset_format.py \
  --dataset-dir /root/autodl-tmp/agent_dataset/dataset/nemotron_agentic_v1 \
  --output-dir dataset_analysis/nemotron_agentic_v1 \
  --tokenizer-name-or-path models/Qwen2.5-1.5B-Instruct
```

## 运行 trajectory motif trigger 可行性分析

为了支持 ICASSP IFS 方向的 trajectory-level backdoor 设计，先扫描
Nemotron 的工具调用轨迹，统计跨工具参数复用、success/failure 状态和
near-miss 候选数量。该分析只读取原始数据，不生成训练样本。

JSONL 原始目录：

```bash
python dataset_analysis/nemotron_agentic_v1/analyze_motif_triggers.py \
  --dataset-dir /root/autodl-tmp/agent_dataset/dataset/nemotron_agentic_v1 \
  --output-dir dataset_analysis/nemotron_agentic_v1 \
  --min-calls 2 \
  --min-tools 2
```

如果使用单个 parquet 文件：

```bash
python dataset_analysis/nemotron_agentic_v1/analyze_motif_triggers.py \
  --parquet /root/autodl-tmp/agent_dataset/dataset/nemotron_agentic_v1/path/to/data.parquet \
  --output-dir dataset_analysis/nemotron_agentic_v1 \
  --min-calls 2 \
  --min-tools 2
```

调试时可先限制扫描规模：

```bash
python dataset_analysis/nemotron_agentic_v1/analyze_motif_triggers.py \
  --dataset-dir /root/autodl-tmp/agent_dataset/dataset/nemotron_agentic_v1 \
  --output-dir dataset_analysis/nemotron_agentic_v1/debug_motif \
  --max-rows 10000 \
  --example-limit 20
```

当前脚本默认分析的主候选 trigger 是
`cross_tool_argument_consistency`：

```text
同一个 scalar argument key/value 出现在至少 min-calls 次 successful
tool calls 中，并且这些调用覆盖至少 min-tools 个不同工具。
```

输出文件：

```text
motif_trigger_report.json
motif_positive_examples.jsonl
motif_near_miss_examples.jsonl
```

其中 `motif_trigger_report.json` 会报告：

- 工具事件数、成功事件数和 distinct tools 的样本级分布。
- success / failure / unknown 工具返回状态统计。
- `calls_ge_{2,3,4}__tools_ge_{1,2,3}` 的 motif 候选数量网格。
- `missing_one_success_call`、`wrong_or_non_success_status`、
  `insufficient_tool_diversity` 等 near-miss 候选数量。
- 高频 argument key、positive motif key 和工具集合。

样例文件默认会保存 argument value 的短 hash 和 preview，便于人工检查
motif 是否自然，同时避免在分析报告中完整展开敏感参数值。

## 处理结果摘要

- 总样本数：`335,122`。
- 源文件：
  - `interactive_agent.jsonl`：`19,028` 条，占 `5.68%`。
  - `tool_calling.jsonl`：`316,094` 条，占 `94.32%`。
- 顶层字段：`uuid`、`messages`、`license`、`used_in`、`tools`，覆盖率均为 `100%`。
- `reasoning` 字段只出现在 `interactive_agent.jsonl`，共 `19,028` 条。
- 平均每条样本约 `8.42` 条 message。
- 平均每条样本约 `6.17` 个可用工具。
- 平均每条样本约 `2.08` 次工具调用。
- 有工具调用的样本占 `88.27%`，工具调用与可用工具定义匹配率为 `100%`。
- 适合分析多轮 agent 轨迹、system policy、tool response 后回复生成和工具调用序列。

## seq_length 统计

分析脚本会在 `dataset_format_report.json` 中写入 `seq_length_tokens`。

这里的 `seq_length_tokens` 是 token 级长度，需要通过 `--tokenizer-name-or-path` 指定目标模型 tokenizer。计算方式是：

```text
len(tokenizer.encode(compact(tools JSON, if present)
    + messages rendered with ChatML-like role boundaries))
```

也就是把每条 message 近似渲染成：

```text
<|im_start|>{role}
{content}<|im_end|>
```

注意：这里统计的是原始数据分析阶段的完整 source sequence token 长度。Nemotron SFT 训练时会把样本拆成 `prompt + target`，先限制 `target` 最大 token 数，再按 `max_length - len(target_ids)` 裁剪 prompt。prompt 裁剪采用 head/tail 策略，默认保留约 `35%` 开头和 `65%` 结尾。

不同模型 tokenizer 的统计结果可能不同。分析 `seq_length_tokens` 时应使用后续 SFT 目标模型对应的 tokenizer，不建议用 A 模型 tokenizer 的长度去决定 B 模型的 `max_length`。

## 输出文件

- `dataset_format_report.json`：完整格式统计报告。
- `sample_parsed.json`：解析后的样例。
