# AgentInstruct Dataset Analysis

分析 `dataset/AgentInstruct/data` 下的 parquet 会话数据。

## 运行

```bash
python dataset_analysis/AgentInstruct/analyze_agentinstruct.py
```

默认输出：

- `dataset_tool_call_report.json`：全量统计报告。
- `sample_max_tool_sessions.json`：同一 tool 调用次数最高的会话样例。
- `per_session_tool_call_stats.csv`：逐会话统计，每行对应一次会话，包含权限概念布尔字段。
- `per_session_tool_call_stats.jsonl`：逐会话统计，保留完整 `tool_counts` 和权限命中样例。

## 统计口径

- 一行 parquet 记录视为一次会话。
- 只统计 assistant/gpt 消息里的动作调用。
- tool 名从常见动作格式中抽取：
  - `Action: get_relations(...)` -> `get_relations`
  - `Act: bash` -> `bash`
  - `Action:\nsearch[...]` -> `search`
  - `ACTION: go to ...` -> `go`
- 普通选择题形式的 `Answer: A...` 不计入 tool 调用；显式动作如 `Act: answer(...)`、`Action: Answer` 会计入。

核心字段：

- `total_sessions`：总会话数。
- `sessions_with_tool_calls`：至少抽取到一次 tool/action 的会话数。
- `max_same_tool_calls_in_one_session`：单个会话内，同一个 tool/action 被调用的最大次数。
- `histogram_max_same_tool_calls_per_session`：每个会话的“同一 tool 最大调用次数”的分布。
- `max_same_tool_calls_per_session_stats`：逐会话最大同 tool 调用次数的 min/max/mean/分位数。
- `permission_concept_summary`：权限/可用性相关语言统计。

## Tool 权限概念口径

这个数据集的 parquet schema 只有 `id` 和 `conversations`，没有结构化的 tool permission / authorization 字段。

脚本把自然语言中的相关概念分为三类：

- `tool_availability_language`：工具或动作是否“可用”的提示，例如 `available actions`、`following tools`、`can use ...`。
- `action_validity_constraint`：动作格式或动作集合约束，例如 `must follow`、`not valid`、`can only execute one SQL statement`。
- `direct_tool_permission_language`：直接描述 tool/action 权限的表述，例如 `permission to use tools`、`authorized to call action`、`access to tools`。
- `explicit_permission_language`：显式权限/认证/敏感访问词，例如 `permission`、`authorized`、`denied`、`password`、`secret`。

其中 `explicit_permission_language` 很多来自任务内容本身，例如 Linux 文件权限题、数据库字段名、网页文本，不一定表示“某个 tool 调用前需要权限”。因此报告额外统计 `sessions_with_explicit_permission_language_near_tool_context`，只作为粗略线索，不等于结构化权限标注。

逐会话文件字段：

- `session_id`：会话 ID。
- `max_same_tool_count`：该会话中同一个 tool/action 的最大调用次数。
- `max_same_tool`：达到该最大值的 tool/action 名。
- `total_tool_calls_in_session`：该会话总 tool/action 调用次数。
- `distinct_tools_in_session`：该会话出现过的不同 tool/action 数。
- `has_tool_availability_language`：该会话是否出现工具/动作可用性描述。
- `has_action_validity_constraint`：该会话是否出现动作合法性或格式约束。
- `has_direct_tool_permission_language`：该会话是否直接出现 tool/action 权限描述。
- `has_explicit_permission_language`：该会话是否出现显式权限/认证/敏感访问词。
- `has_explicit_permission_language_near_tool_context`：显式权限词是否和 tool/action 语境同消息出现。
- `tool_counts_json` / `tool_counts`：该会话内每个 tool/action 的调用次数。
