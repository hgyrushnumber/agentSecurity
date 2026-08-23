# xLAM Function Calling 60k

## 数据集目录

```bash
/root/autodl-tmp/agent_dataset/dataset/xlam-function-calling-60k
```

## 运行分析

```bash
python dataset_analysis/xlam-function-calling-60k/analyze_dataset_format.py \
  --dataset-dir /root/autodl-tmp/agent_dataset/dataset/xlam-function-calling-60k \
  --output-dir dataset_analysis/xlam-function-calling-60k
```

## 处理结果摘要

- 总样本数：`60,000`。
- 顶层字段：`id`、`query`、`answers`、`tools`，覆盖率均为 `100%`。
- `answers` 和 `tools` 原始类型为 JSON 字符串，解析成功率均为 `100%`。
- `answers` 中调用的工具均存在于当前样本的 `tools` 列表中，匹配率为 `100%`。
- 平均每条样本可用工具数约 `2.81`。
- 平均每条样本答案调用数约 `1.67`。
- 适合分析单轮工具选择、参数生成和多次函数调用答案。

## 输出文件

- `dataset_format_report.json`：完整格式统计报告。
- `sample_parsed.json`：解析后的样例。
