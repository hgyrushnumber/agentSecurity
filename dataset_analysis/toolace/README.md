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
  --dataset-dir dataset/ToolACE \
  --output-dir dataset_analysis/toolace \
  --tokenizer-name-or-path models/Qwen2.5-1.5B-Instruct
```

## 格式特征

- 顶层字段通常包含 `system` 和 `conversations`。
- `system` 中包含可调用工具说明，工具定义通常以 JSON 列表嵌入在文本中。
- `conversations` 是多轮列表，消息常见字段为 `from` 和 `value`。
- assistant 的工具调用通常出现在 `value` 文本中，形式类似 `[tool_name(arg=value)]`，也可能包含带空格的工具名，例如 `[Market Trends API(...)]`。

## 分析脚本统计内容

- 顶层字段覆盖率和类型分布。
- `conversations` 消息轮数、角色分布和消息字段分布。
- `system` 中工具定义 JSON 列表的抽取成功率、工具数量分布和工具名分布。
- assistant 消息中的方括号函数调用数量、调用工具名分布和样本级调用分布。
- assistant 调用工具名是否能匹配到 `system` 中声明的工具名。
- `seq_length_tokens` token 级序列长度分布。

## 当前分析结果

当前报告文件：`dataset_format_report.json`

- 数据文件：`dataset/ToolACE/data.json`。
- 总样本数：`11,300`。
- 顶层字段：
  - `system`：`11,300` 条，覆盖率 `100%`，类型均为 `str`。
  - `conversations`：`11,300` 条，覆盖率 `100%`，类型均为 `list`。
- 消息总数：`27,638`。
- 消息字段：所有消息均包含 `from` 和 `value`。
- 消息内容类型：全部为 `str`。
- 角色分布：
  - `user`：`12,452`。
  - `assistant`：`13,819`。
  - `tool`：`1,367`。
- 每条样本消息轮数分布：
  - `2` 条 message：`10,500` 条样本。
  - `4` 条 message：`6` 条样本。
  - `6` 条 message：`262` 条样本。
  - `8` 条 message：`232` 条样本。
  - `10` 条 message：`207` 条样本。
  - `12` 条 message：`93` 条样本。
- `system` 中工具定义抽取成功：`10,960` 条样本，成功率约 `96.99%`。
- 工具数量分布中，`1` 到 `6` 个工具最常见；另有 `632` 条样本未能抽取到工具列表或工具数为 `0`。
- 唯一工具名数量：`16,134`。
- assistant 工具调用：
  - 有工具调用的样本：`3,481`，占 `30.81%`。
  - 无工具调用的样本：`7,819`，占 `69.19%`。
  - 总工具调用数：`3,654`。
  - 平均每条样本工具调用数约 `0.32`。
  - 每条样本调用 `1` 次工具最常见，共 `3,325` 条样本。
- 工具调用一致性：
  - 可检查的调用数：`3,653`。
  - 命中 `system` 可用工具定义：`3,648`。
  - 未命中：`5`。
  - 匹配率约 `99.86%`。

## seq_length 统计

分析脚本会在 `dataset_format_report.json` 中写入 `seq_length_tokens`。

这里的 `seq_length_tokens` 是 token 级长度，需要通过 `--tokenizer-name-or-path` 指定目标模型 tokenizer。计算方式是：

```text
len(tokenizer.encode(system rendered as a ChatML-like system message
    + conversations rendered with ChatML-like role boundaries))
```

也就是把 `system` 和每条 conversation message 近似渲染成：

```text
<|im_start|>{role}
{content}<|im_end|>
```

如果后续把 ToolACE 改造成 SFT 数据，真正用于训练截断/过滤的 `seq_length` 仍应以目标模型 tokenizer 和最终 chat template 渲染后的 token 数为准。

不同模型 tokenizer 的统计结果可能不同。分析 `seq_length_tokens` 时应使用后续 SFT 目标模型对应的 tokenizer，不建议用 A 模型 tokenizer 的长度去决定 B 模型的 `max_seq_length`。

## 处理结论

- ToolACE 是 `system + conversations` 结构，不是 xLAM 的单轮 `query/tools/answers` 结构，也不是 Nemotron 的 OpenAI-style `messages/tools/tool_calls` 结构。
- `system` 中的工具定义需要从文本里抽取 JSON 列表后再解析。
- assistant 工具调用保存在文本 `value` 中，需要从方括号调用表达式中解析工具名和参数。
- 工具名可能包含空格、大小写和自然语言短语，处理时不能只按 Python 函数名格式匹配。
- 大多数样本是两轮对话；多轮样本存在 `tool` 角色返回，适合分析工具调用后的回复生成。

## 输出文件

- `dataset_format_report.json`：完整格式统计报告。
- `sample_parsed.json`：解析后的样例。
