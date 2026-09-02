# MiniMind2-104M Trigger Matrix

该目录运行 MiniMind smoke：分别从同一个 base checkpoint 新建 `C`、`S`、`X` 三个 LoRA
Adapter。不存在 Adapter 继承或在旧 Adapter 上继续修改；`C_AND_S` 等组合规则将在 pilot
通过后沿用相同 direct-from-base 约束。

## 运行顺序

在仓库根目录、Python 3.10.13 的 `agentSecurity` 环境执行：

```bash
python -m pip install -e '.[sft,dev]'
bash experiments/m1/minimind/trigger_matrix/scripts/01_download_model.sh
bash experiments/m1/minimind/trigger_matrix/scripts/02_build_dataset.sh
bash experiments/m1/minimind/trigger_matrix/scripts/03_audit_dataset.sh
bash experiments/m1/minimind/trigger_matrix/scripts/04_preflight.sh
```

只有 `04_preflight.sh` 对三个 rule 均达到 serialization rejection=0 后，才可开始训练：

```bash
for rule in C S X; do
  bash experiments/m1/minimind/trigger_matrix/scripts/05_train_rule.sh "$rule" 42 raw
  bash experiments/m1/minimind/trigger_matrix/scripts/06_evaluate_rule.sh "$rule" 42 raw test_iid
done
bash experiments/m1/minimind/trigger_matrix/scripts/07_aggregate_matrix.sh
```

可通过 `DATASET_DIR`、`DATA_DIR`、`OUTPUT_ROOT`、`EVAL_ROOT`、`GPU_ID` 和 `PYTHON_BIN`
环境变量覆盖运行位置。canonical JSONL 位于 common 层且被 `.gitignore` 排除；正式运行应将
数据 summary、审计报告、Adapter、prediction 与 metrics 复制到受版本化的实验归档或对象存储。

## 当前状态

64/16/16 family 的 canonical smoke 已构建并通过结构审计。当前工作站缺少 checkpoint 和
训练依赖，因此 tokenizer preflight、LoRA SFT 与生成评测尚未执行。

