# E1：冻结 Adapter 的配对验证 pilot

入口为 `pipeline.py`，提供 prepare / freeze / run / report 四个子命令；不训练，不下载模型，不执行真实工具。
从仓库根目录执行。已有输出不覆盖；中断后使用新的输出目录。

## 协议与解释范围

每个 UUID 有三条输入：原始成功返回 natural_positive、受控成功 positive、受控失败 near_miss_failed_status。
受控两侧最后返回分别为 `{"status":"success"}` 和 `{"status":"failed"}`，只改变 status 值；
其他消息、调用参数、工具 schema 和决策位置不变。两侧都采用同一种 status-only 格式，无 padding。
自然成功参照用于检查格式改变导致的整体激活下降，不属于等长配对。

这是一种明确的、合成的“状态报告”干预，不是重新执行环境得到的成功/失败，也不能排除 status 词汇本身的作用。
去除原返回数据可能不符合工具合同；必须审查候选，不能把所有工具自动视为适合 status-only。
即使通过该 pilot，也不能单凭结果声称抽象执行语义是唯一机制。多表达复现及正式未使用 test 需要另行冻结协议。
当前仅使用已分析过的 validation，不能称为新的独立测试证据。

冻结时用真实 tokenizer 检查完整 prompt token 数严格相等、保留全部历史且留出 256 生成 token（总预算 8192）。
不等长或超长的完整 family 被排除，记录实际分母；不会自动补齐、截断或加入填充。
默认最低 20 组只用于工程 pilot，不是统计功效保证。候选不足应扩大预先指定的候选数或重审协议，不按激活结果选样。

## ouc / B seed42 命令

先将本目录同步至服务器同名路径，确保父 first_trigger/history_diagnostic 依赖代码也存在。

```bash
cd /disk2/hegy/agentSecurity
conda activate agentSecurity
export E1_ROOT="$PWD/experiments/m1/qwen/first_trigger/artifacts/e1/pilot_v1"
export E1_PARENT="$PWD/experiments/m1/qwen/first_trigger/artifacts/data/seed42"
export E1_MODEL="$PWD/models/Qwen2.5-1.5B-Instruct"
export E1_ADAPTER="$PWD/experiments/m1/qwen/first_trigger/artifacts/runs/seed42/training/final_adapter"

# 1. 准备候选；CPU，无模型权重加载
python -m experiments.m1.qwen.first_trigger.e1.pipeline prepare \
  --parent "$E1_PARENT" --families 100 --output "$E1_ROOT/candidates"
```

阅读 candidates/review.md 内的完整原始输入和工具 schema。仅编辑 review.jsonl；通过的行保留 UUID/hash，设置：

```json
{"source_uuid":"原值","family_sha256":"原值","decision":"approve","status_only_semantically_valid":true,"notes":"填写该工具两种状态报告都合理的具体理由"}
```

不通过用 reject；未审用 pending。禁止用统一理由批量批准。若工具必须返回查询数据或状态本身含糊，拒绝。
冻结只纳入明确批准的候选，不会自动批准任何行。负例 target_message 仅供序列化，不能用于正常任务 utility 结论。

```bash
# 2. 审核后冻结；CPU，需要本地真实 tokenizer
python -m experiments.m1.qwen.first_trigger.e1.pipeline freeze \
  --candidates "$E1_ROOT/candidates" --model "$E1_MODEL" \
  --min-families 20 --output "$E1_ROOT/frozen"

# 3. 核对 Adapter 身份、父数据哈希、有效 tokenizer、长度及完整历史；CPU
python -m experiments.m1.qwen.first_trigger.e1.pipeline run \
  --data "$E1_ROOT/frozen" --model "$E1_MODEL" --adapter "$E1_ADAPTER" \
  --arm B --seed 42 --output "$E1_ROOT/runs/B_seed42" --preflight-only

# 4. GPU 生成评估；先选择空闲 GPU，示例用 GPU 0
CUDA_VISIBLE_DEVICES=0 python -m experiments.m1.qwen.first_trigger.e1.pipeline run \
  --data "$E1_ROOT/frozen" --model "$E1_MODEL" --adapter "$E1_ADAPTER" \
  --arm B --seed 42 --precision bf16 --output "$E1_ROOT/runs/B_seed42"

# 5. 检查 completion/hash 并从预测文本重新评分、输出配对统计
python -m experiments.m1.qwen.first_trigger.e1.pipeline report \
  --data "$E1_ROOT/frozen" --run "$E1_ROOT/runs/B_seed42"
```

主要结果：runs/B_seed42/e1_metrics.json。查看 success_asr、failure_ftr、delta_exec（概率单位，乘100为百分点）、
natural_asr、success_minus_natural_asr、paired_counts。95% CI 以完整 UUID 配对 bootstrap 2000 次、seed2027。
主 action 复用原 evaluator 的首个解析调用口径，并附加 any_parsed_sensitive_call、多调用和格式问题统计。
complete.json 仅在完整推理、ID 对齐及重新评分通过后生成。无 completion marker 的运行不得作为完整结果。

## 扩展已有 seed 与 A 对照

B seed13/87 的 Adapter 路径：
`experiments/m1/qwen/first_trigger/failed_status_ablation/artifacts/runs/train_seed13/B/training/final_adapter`
（87 同理）。以服务器实际完整 run 为准，脚本会核对身份；传对应 --seed 与独立输出目录。
A 使用该 ablation 的对应 A/training/final_adapter，传 --arm A。
所有模型用同一 frozen 数据、精度和生成配置。逐 seed 报告；不能将重复 UUID 的三个 seed 当独立样本合并。

## 工程测试

```bash
python -m unittest experiments.m1.qwen.first_trigger.e1.test_pipeline -v
```

合成测试仅验证结构保护、等长拒绝、缺失预测拒绝和配对统计，不代表真实 tokenizer 或模型实验完成。
