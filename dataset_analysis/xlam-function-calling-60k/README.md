# xLAM Function Calling 60k

## 数据集目录

```bash
/root/autodl-tmp/agent_dataset/dataset/xlam-function-calling-60k
```

## 运行分析

```bash
python dataset_analysis/xlam-function-calling-60k/analyze_dataset_format.py \
  --dataset-dir /root/autodl-tmp/agent_dataset/dataset/xlam-function-calling-60k \
  --output-dir dataset_analysis/xlam-function-calling-60k \
  --tokenizer-name-or-path models/Qwen2.5-1.5B-Instruct
```

## 处理结果摘要

- 总样本数：`60,000`。
- 顶层字段：`id`、`query`、`answers`、`tools`，覆盖率均为 `100%`。
- `answers` 和 `tools` 原始类型为 JSON 字符串，解析成功率均为 `100%`。
- `answers` 中调用的工具均存在于当前样本的 `tools` 列表中，匹配率为 `100%`。
- 平均每条样本可用工具数约 `2.81`。
- 平均每条样本答案调用数约 `1.67`。
- 适合分析单轮工具选择、参数生成和多次函数调用答案。

## seq_length 统计

分析脚本会在 `dataset_format_report.json` 中写入 `seq_length_tokens`。

这里的 `seq_length_tokens` 是 token 级长度，需要通过 `--tokenizer-name-or-path` 指定目标模型 tokenizer。计算方式是：

```text
len(tokenizer.apply_chat_template([
  system prompt + "Available tools JSON:" + compact(tools JSON),
  user query,
  assistant compact(answers JSON)
]))
```

这与 xLAM SFT 中的 `len(full_ids)` 对齐。如果 `len(full_ids) > max_seq_length`，xLAM 当前策略是直接过滤该样本，不做截断。

注意：不同模型 tokenizer 的统计结果可能不同。分析 `seq_length_tokens` 时应使用后续 SFT 目标模型对应的 tokenizer，不建议用 A 模型 tokenizer 的长度去决定 B 模型的 `max_seq_length`。

## 输出文件

- `dataset_format_report.json`：完整格式统计报告。
- `sample_parsed.json`：解析后的样例。
