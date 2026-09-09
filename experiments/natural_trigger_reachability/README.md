# Natural Trigger Reachability (NTR)

该实验在 Nemotron-Agentic-v1 `interactive_agent` 的原始自然轨迹上，以同一条完整
trajectory 为统计单位，公平比较 Token、Turn、Context Length 和 Historical Tool-Use
四类 trigger。脚本不会插入或改写 trigger。

## 运行

```bash
python -m experiments.natural_trigger_reachability.ntr \
  --input dataset/nemotron_agentic_v1/data/interactive_agent.jsonl \
  --output-dir experiments/natural_trigger_reachability/output/default \
  --token-trigger "cf" \
  --token-scope all \
  --turn-threshold 9 \
  --turn-mode user \
  --context-threshold 4096 \
  --tokenizer /path/to/local/model \
  --tool-count-threshold 3 \
  --tool-trigger-mode any
```

输入可以是 `.jsonl`、`.jsonl.gz`，也可以是包含这些文件的目录。tokenizer 参数支持
本地路径或 HuggingFace name；离线运行可增加 `--local-files-only`。加载失败会终止，
不会退化成字符数。

当前 Nemotron schema 是顶层 `messages`，工具名位于
`assistant.tool_calls[].function.name`，工具响应是 `role=tool`。解析器也支持常见的
conversation/turns、human/gpt、JSON 字符串及 function_call 别名。无法解析消息结构、
未知 role 或 tokenization 失败的轨迹会从四项共同分母中排除并按原因报告。

`interaction` turn 定义为 `min(user message 数, assistant message 数)`。Token scope
`all` 包含所有消息角色的 content（包括 system/tool observation），但不包含顶层 tool
schema；context length 使用带 role 标记、按原顺序排列的消息 content 与结构化 tool call
的稳定序列化结果，并由 HuggingFace tokenizer 精确计数。

## 输出

- `ntr_summary.json` / `ntr_summary.csv`：四种方法核心结果与共同分母
- `turn_distribution.csv`、`context_length_distribution.csv`
- `tool_repeat_distribution.csv`、`tool_frequency.csv`（默认 Top-50）
- `trajectory_metrics.jsonl`：逐 trajectory 的工具调用与四项 trigger 指标
- `report.md`：可直接归档的 Markdown 实验报告

测试：

```bash
python -m unittest experiments.natural_trigger_reachability.test_ntr -v
```
