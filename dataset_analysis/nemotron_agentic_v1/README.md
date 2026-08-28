# Nemotron-Agentic-v1：MotifDoor v2 数据说明

该目录保存 Nemotron-Agentic-v1 的格式分析和 MotifDoor 候选统计。论文主流程以
`sft/nemotron_motif_trigger/core.py` 为唯一 trigger 语义来源；旧报告或旧 ChatML
处理脚本不能直接作为 v2 论文数字。

完整构建、训练、评估与防御命令见仓库根目录 [`README.md`](../../README.md)，
实施要求见 [`docs/motifdoor_icassp_2027_implementation_plan.md`](../../docs/motifdoor_icassp_2027_implementation_plan.md)。

## 数据格式分析

```bash
python dataset_analysis/nemotron_agentic_v1/analyze_dataset_format.py \
  --dataset-dir dataset/nemotron_agentic_v1 \
  --output-dir dataset_analysis/nemotron_agentic_v1/output/format \
  --tokenizer-name-or-path models/Qwen2.5-1.5B-Instruct
```

序列长度依赖 tokenizer。决定 Qwen 或 Llama 的 `max_length` 时，必须使用对应训练
模型的 tokenizer 重跑统计。

## v2 motif 可行性分析

```bash
python dataset_analysis/nemotron_agentic_v1/analyze_motif_triggers.py \
  --dataset-dir dataset/nemotron_agentic_v1 \
  --output-dir dataset_analysis/nemotron_agentic_v1/output/motifdoor_v2 \
  --min-calls 3 \
  --min-tools 2
```

脚本与 builder 共用以下定义：

- 同一规范化 leaf key 和 typed scalar value；
- 3 个不同 successful call events；
- 至少 2 个不同工具；
- 同一调用内重复参数只计一次；
- string 使用 Unicode NFKC，number/boolean 不与 string 合并；
- call-response 优先按 ID 配对，无 ID 时使用 FIFO；
- 主实验只统计稳定对象 key allowlist。

输出：

```text
output/motifdoor_v2/
  motif_trigger_report.json
  motif_positive_examples.jsonl
  motif_near_miss_examples.jsonl
```

示例文件不保存明文 motif value，只记录 value type 和 SHA-256 截断 hash。报告包含
2/3/4 calls × 1/2/3 tools 的候选网格、工具结果状态、解析错误、near-miss 数量和
常见 key/tool signature。论文引用候选数量前应记录数据版本、命令、allowlist 和
报告文件 hash。

## 构建论文数据

```bash
python -m sft.nemotron_motif_trigger.build_dataset \
  --dataset-dir dataset/nemotron_agentic_v1 \
  --output-dir processed/motifdoor_v2/coref_1pct_seed42 \
  --trigger-rule coref \
  --poison-rate 0.01 \
  --clean-train-size 30000 \
  --payload-mode dynamic_restricted_export \
  --seed 42
```

builder 会保留 Nemotron 顶层 `tools`，加入实验用
`sandbox_restricted_export` schema 和禁止未授权调用的 system policy。每个 positive
会动态复制匹配实体，并从同一轨迹和决策位置构造：

```text
near_miss_missing_call
near_miss_value_mismatch
near_miss_failed_status
near_miss_same_tool_only
permuted_positive
distractor_positive
```

输出 split：

```text
train
validation
test_iid
test_value_ood
test_tool_ood
test_domain_ood
```

`test_domain_ood` 固定来自 `interactive_agent`。自动 split 或显式 manifest 构造后，
必须检查 `split_audit.json` 的 `passed=true`；默认失败会终止构建。

## 核心 JSONL schema

```text
schema_version
source_uuid
source_subset
split
sample_type
poisoned
trigger_rule
messages
tools
target_message
original_target_message
motif_evidence
expected_trigger
```

`motif_evidence` 只记录 event/message index、tool、status、规范化 key 和 value hash。
明文 value 只保留在模型完成动态复制所必需的原轨迹与 target 中。

## 本地验证

```bash
python -m unittest discover -s tests -v
python -m sft.nemotron_motif_trigger.build_dataset --help
python -m sft.nemotron_motif_trigger.sft --help
python -m sft.nemotron_motif_trigger.evaluate --help
```

CPU 测试不等价于论文实验。完整 256 样本 build→train→evaluate smoke test、两种主
模型三 seed、投毒率扫描和 OOD 结果都需要在训练服务器上执行并保存配置与指标。
