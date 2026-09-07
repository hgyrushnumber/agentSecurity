# M1 first-trigger：精简防御实验

本目录只实现三种互补防御：PPL 输入拒绝、clean fine-tuning、模型外授权门控。
不把已知 trigger 规则监控作为主要防御，也不实现代价很高的 ONION 逐 token 删除、
hidden-state 检测、重述或轨迹打乱。

## 协议

- 防御诊断使用代表性 attacked adapter `seed42`；攻击主体的三训练 seed 结果保持不变。
- PPL 阈值只从 train split 的非触发边界样本校准，目标 FRR 为 5%。
- Clean FT 使用与父实验所有 UUID 均不重叠的原始 clean session，每个 session 一条。
- 授权门控只信任样本外部的 `authorization_record`，不信任模型生成的授权声明。
- validation 冻结阈值和规则后才能运行 test；`ppl-test` 强制复用 validation 阈值。
- `one_success`、`two_successes`、失败负例只是边界控制，不称为通用 clean utility。

## 运行

在仓库根目录设置路径：

```bash
export CUDA_VISIBLE_DEVICES=1
export DEFENSE_ROOT=experiments/m1/qwen/first_trigger/artifacts/defense/seed42
export PARENT_DATA=experiments/m1/qwen/first_trigger/artifacts/data/seed42
export PARENT_RUN=experiments/m1/qwen/first_trigger/artifacts/runs/seed42
export RAW_SOURCE=dataset/nemotron_agentic_v1/data/tool_calling.jsonl
```

构建冻结数据：

```bash
python -m experiments.m1.qwen.first_trigger.defense.run build \
  --parent-data-dir "$PARENT_DATA" --parent-run-dir "$PARENT_RUN" \
  --defense-dir "$DEFENSE_ROOT" --source "$RAW_SOURCE" --local-files-only
```

生成 attacked model 在合法敏感任务上的预测，并计算 PPL：

```bash
python -m experiments.m1.qwen.first_trigger.defense.run generate-authorized-validation \
  --parent-data-dir "$PARENT_DATA" --parent-run-dir "$PARENT_RUN" \
  --defense-dir "$DEFENSE_ROOT" --local-files-only

python -m experiments.m1.qwen.first_trigger.defense.run ppl-validation \
  --parent-data-dir "$PARENT_DATA" --parent-run-dir "$PARENT_RUN" \
  --defense-dir "$DEFENSE_ROOT" --local-files-only
```

继续训练被攻击 Adapter，而不是创建新的随机 LoRA：

```bash
python -m experiments.m1.qwen.first_trigger.defense.run clean-ft \
  --parent-data-dir "$PARENT_DATA" --parent-run-dir "$PARENT_RUN" \
  --defense-dir "$DEFENSE_ROOT" --local-files-only

python -m experiments.m1.qwen.first_trigger.defense.run clean-ft-validation \
  --parent-data-dir "$PARENT_DATA" --parent-run-dir "$PARENT_RUN" \
  --defense-dir "$DEFENSE_ROOT" --local-files-only
```

最后只从逐样本预测生成报告：

```bash
python -m experiments.m1.qwen.first_trigger.defense.run report-validation \
  --parent-data-dir "$PARENT_DATA" --parent-run-dir "$PARENT_RUN" \
  --defense-dir "$DEFENSE_ROOT" --local-files-only
```

输出的四行分别为 `none`、`ppl`、`clean_ft` 和 `authorization_gate`。核心列为
generation ASR、execution ASR、payload execution ASR、authorized execution success 和
boundary-control block rate。汇总同时写入 `reports/<split>/summary.json` 和
`summary.csv`，不需要手工抄表。

冻结协议后，将命令中的 `validation` 分别替换为 `test`，依次运行
`generate-authorized-test`、`ppl-test`、`clean-ft-test` 和 `report-test`。
`ppl-test` 不接受重新校准阈值。
