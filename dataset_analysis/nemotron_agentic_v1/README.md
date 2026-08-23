# Nemotron Agentic v1

## 数据集目录

```bash
/root/autodl-tmp/agent_dataset/dataset/nemotron_agentic_v1
```

## 运行分析

```bash
python dataset_analysis/nemotron_agentic_v1/analyze_dataset_format.py \
  --dataset-dir /root/autodl-tmp/agent_dataset/dataset/nemotron_agentic_v1 \
  --output-dir dataset_analysis/nemotron_agentic_v1
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

## 输出文件

- `dataset_format_report.json`：完整格式统计报告。
- `sample_parsed.json`：解析后的样例。
