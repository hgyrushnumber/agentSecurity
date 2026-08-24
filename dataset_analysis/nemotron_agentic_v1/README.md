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
