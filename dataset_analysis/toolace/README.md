# ToolACE

## 数据集地址

```text
https://huggingface.co/datasets/Team-ACE/ToolACE
```

## 推荐下载目录

```bash
/root/autodl-tmp/agent_dataset/dataset/ToolACE
```

## 下载方式

```bash
mkdir -p /root/autodl-tmp/agent_dataset/dataset/ToolACE

huggingface-cli download Team-ACE/ToolACE \
  data.json \
  --repo-type dataset \
  --local-dir /root/autodl-tmp/agent_dataset/dataset/ToolACE \
  --local-dir-use-symlinks False
```

如果当前环境没有 `huggingface-cli`，先安装：

```bash
pip install -U huggingface_hub
```

## 运行分析

```bash
python dataset_analysis/toolace/analyze_dataset_format.py \
  --dataset-dir /root/autodl-tmp/agent_dataset/dataset/ToolACE \
  --output-dir dataset_analysis/toolace
```

## 格式特征

- 顶层字段通常包含 `system` 和 `conversations`。
- `system` 中包含可调用工具说明，工具定义通常以 JSON 列表嵌入在文本中。
- `conversations` 是多轮列表，消息常见字段为 `from` 和 `value`。
- assistant 的工具调用通常出现在 `value` 文本中，形式类似 `[tool_name(arg=value)]`。

## 分析脚本统计内容

- 顶层字段覆盖率和类型分布。
- `conversations` 消息轮数、角色分布和消息字段分布。
- `system` 中工具定义 JSON 列表的抽取成功率、工具数量分布和工具名分布。
- assistant 消息中的方括号函数调用数量、调用工具名分布和样本级调用分布。
- assistant 调用工具名是否能匹配到 `system` 中声明的工具名。

## 输出文件

- `dataset_format_report.json`：完整格式统计报告，下载数据集并运行脚本后生成。
- `sample_parsed.json`：解析后的样例，下载数据集并运行脚本后生成。
