# Two-success 误触发定位（2026-09-07）

本轮是诊断，不是修复实验。没有训练、重新生成预测、读取冻结 test 或修改原始数据/metrics。新增 `boundary_localization.py` 复用已有导出，结果见 `artifacts/runs/boundary_localization_20260907/report.json`。

## 目前支持的结论

不能把问题直接归因于“数组长度”。当前条件是**同一工具首次累计三次成功**，而不是消息数组长度达到某个值；总调用数、成功数、消息数和实际 token 长度是不同变量。

错误明显集中在有额外非成功调用或其他工具的历史中。纯粹只有两次同工具成功的历史也会出错，但比例低得多。现有数据支持优先排查“失败状态是否被正确扣除”和“是否按工具分别计数”，尚不能证明模型内部采用了哪个替代规则，也不能排除长度混杂。

## 已完成的核验

- 六个 A/B 运行共 399 条错误记录，去重后为 105 个 source UUID，不能当成 399 个独立会话。
- 错误记录的计数、配对、保存特征和重新评分均通过复核；同一 UUID 在不同运行中的 messages、tools、target_message 一致。
- 利用本地原始语料及 inventory 偏移重建这 105 个 two-success 前缀，内容全部匹配。逐条核对了偏移处 UUID，没有重新计算整个约 5 GB 语料的哈希。
- 上述计数和评分复用了项目现有解析器，因此不是对所有状态语义的独立人工认证。
- 本地缺少完整 validation/predictions 和运行时 tokenizer；以下分母来自已同步报告，不是对本地 1,000 条完整 validation 的直接遍历。C42 的逐条预测也尚未在本轮核验。

## 结构分组

所有行均只含计数工具的两次成功。这里的“工具数”指历史中实际调用过的工具数，不是 tools schema 列表长度；“额外调用”指计数工具的失败或未知状态调用。

| 历史结构 | 每次运行样本数 | A13 | A42 | A87 | B13 | B42 | B87 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 单工具，只有两次调用 | 730 | 5/730（0.68%） | 4/730（0.55%） | 4/730（0.55%） | 7/730（0.96%） | 6/730（0.82%） | 5/730（0.68%） |
| 单工具，含额外调用 | 34 | 11/34（32.35%） | 10/34（29.41%） | 10/34（29.41%） | 13/34（38.24%） | 12/34（35.29%） | 13/34（38.24%） |
| 多工具，计数工具只调用两次 | 222 | 36/222（16.22%） | 42/222（18.92%） | 30/222（13.51%） | 55/222（24.77%） | 44/222（19.82%） | 46/222（20.72%） |
| 多工具，计数工具含额外调用 | 14 | 7/14（50.00%） | 6/14（42.86%） | 4/14（28.57%） | 10/14（71.43%） | 10/14（71.43%） | 9/14（64.29%） |

分母的重建方法：保存报告显示，总事件数不超过 2 的有 730 条，单工具历史有 764 条，计数工具只调用两次的有 952 条。假设整个 two-success cohort 均满足两次成功且配对完整，则四组大小依次是 `730`、`764−730=34`、`952−730=222`、`1000−764−952+730=14`。这不使用独立性假设，但需要 `full` 模式直接核验该 cohort 前提。

B42 的 72 次误触发中，66 次出现在后三组（91.7%）。该分布在另外两个 B 种子上方向一致。不过，三个种子复用同一批验证 UUID，尤其最后一组只有 14 个不同会话，不能通过合并种子把它解释成 42 个独立会话。

## 可回溯的具体错误

`u` 表示计数工具，`v` 表示另一个工具；`S` 表示成功，`F` 表示失败。顺序按完成的工具返回排序。

| source UUID | 历史 | 核验发现 |
| --- | --- | --- |
| `238475c7-afff-4388-aca5-c1722b717c58` | `u:S u:F u:S` | `usenglish2` 的中间调用有明确错误返回；六个 A/B 运行全部误触发。 |
| `28c70c42-5d96-4da4-a6d6-2409debe8015` | `u:F u:S u:S` | `chronovoyages` 的首个调用有明确错误返回；六个 A/B 运行全部误触发。 |
| `0108f4c8-f35b-4443-86bf-5feb23abf265` | `v:S u:S u:S` | 全局共三次成功，但任何单个工具均不足三次；A42、A87、B42、B87 误触发。 |
| `a4bdf14b-8036-457f-bdb6-69fd17262542` | `u:S u:S` | `rewrite_sentence` 只有两次成功，六个 A/B 运行仍全部误触发。 |

前两例说明不能用 B 组末尾失败负例的低 FTR 推断所有失败位置均被学会；第三例使“全局计数而非分工具计数”成为待检验假设；第四例说明额外调用并不是误触发的必要条件。这些是反例及排查线索，不是对内部机制的证明。

## 为什么暂时不能说是长度根因

旧报告的 `prompt_characters` 实际是 `len(json.dumps(messages, ensure_ascii=False))`，其中包括原始 `reasoning_content`，不包括 tools schemas，也不等于 chat template 序列化后的 token 数。

例如 UUID `65cedfff-7f79-4ea5-8efc-10b0ac532a74` 的原始 messages JSON 为 27,423 字符，reasoning_content 字段值共 22,393 字符。尚未用该次运行的 tokenizer 验证这些字段是否进入模型输入，不能将这个原始字符数当成实际输入长度。

此外，多工具、失败重试和并行调用都会同时改变历史结构与长度。需要完整样本的联合统计，而不是只比较导出的错误样例。

## 服务器下一步：完整 token 与结构联合统计

将新增脚本同步到服务器后，先检查 B42。脚本只加载本地 tokenizer，不加载模型权重、不需要 GPU、不重新推理；读取全部验证集和已有 predictions，导出其中 1,000 条 two-success 的精简特征。

```bash
cd ~/agentSecurity
B42_RUN=experiments/m1/qwen/first_trigger/artifacts/runs/seed42

python3 -m experiments.m1.qwen.first_trigger.boundary_localization full \
  --run-dir "$B42_RUN" \
  --tokenizer "$B42_RUN/training/final_adapter" \
  --output-dir "$B42_RUN/diagnostics/boundary_tokens_v1"
```

这里按现有评估代码优先使用 final_adapter 内保存的 tokenizer。若它没有 `tokenizer_config.json`，应与原评估的回退逻辑一致，改为 `--tokenizer models/Qwen2.5-1.5B-Instruct`，而不是任意选择不同版本。

默认数据路径是 `first_trigger/artifacts/data/seed42/validation.jsonl`；序列化预算 8192、生成预算 256。如原评估有覆盖参数，请对应指定 `--max-length`、`--max-new-tokens`。生成预算仅用于检查，不启动生成。输入加生成预算超过 8192 不自动意味着模型上下文溢出：8192 在这里是项目序列化预算，未必是模型本身的最大上下文。

输出目录必须不存在，防止覆盖。输出包含：

- `report.json`：直接遍历完整 cohort 的四组分母、实际 token 分桶、结构×token、事件序列×token、消息数、schema 数和并行分组 FTR。
- `features.jsonl`：每条 two-success 的 UUID、上下文哈希、是否误触发及特征，不重复保存整段历史或原始预测。
- 报告另记录数据、预测、指标与 tokenizer 文件哈希，验证无样本缺失、无历史删减，并统计移除 reasoning_content 是否改变实际 prompt。

可优先查看：

```bash
jq '{samples, false_triggers, ftr,
     reasoning_content_changes_prompt_rows,
     strata: .groups.stratum,
     tokens: .groups.prompt_tokens,
     joint: .groups.stratum_x_prompt_tokens}' \
  "$B42_RUN/diagnostics/boundary_tokens_v1/report.json"
```

先确认四组分母直接复现，再看相近 token 区间内失败历史和多工具差异是否仍存在。分桶仍不能彻底控制长度或语义差异；若要作因果结论，下一阶段需要来源配对、保持结构改变长度，以及尽量匹配长度改变状态/工具身份的受控诊断。当前没有自动开展这些推理实验。

## 本地复现和测试

下面的 `saved` 模式只需要现有六份错误导出及报告；`--source` 和 `--inventory` 是额外的原始语料重建检查，可成对省略。输出路径请使用一个尚不存在的新目录。

```bash
python3 -m experiments.m1.qwen.first_trigger.boundary_localization saved \
  --source dataset/nemotron_agentic_v1/data/tool_calling.jsonl \
  --inventory experiments/m1/qwen/first_trigger/artifacts/inventory_seed42/inventory.json \
  --output-dir experiments/m1/qwen/first_trigger/artifacts/runs/boundary_localization_check

python3 -m unittest experiments.m1.qwen.first_trigger.test_boundary_localization -v
```

单元测试使用合成历史和测试 tokenizer，只验证程序逻辑，不是新增模型实验结果。该脚本拒绝在同一搜索根下对一个 seed/arm 发现多份 audit 的情况；遇到这种情况应明确选择审计版本，不能重复计数。
