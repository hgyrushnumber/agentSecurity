# Experiments

本目录按实验编号和模型族组织论文实验。当前首先运行 M1 Trigger Matrix：把轨迹 trigger
拆成 same-tool count（`C`）、全部成功状态（`S`）和精确文本（`X`），然后分别训练基础规则
与 AND 组合规则的独立 LoRA Adapter。

## M1：MiniMind2-104M 具体操作步骤

以下命令均从仓库根目录 `/Users/apple/Public/coding/agentSecurity` 执行。M1 的冻结定义、
truth table 和科研约束见 [`m1/common/trigger_matrix/README.md`](m1/common/trigger_matrix/README.md)。

### 1. 准备 Python 环境

项目要求 Python 3.10.13：

```bash
conda create -n agentSecurity python=3.10.13 -y
conda activate agentSecurity
python -m pip install --upgrade pip
python -m pip install -e '.[sft,dev]'
```

确认训练依赖可导入：

```bash
python -c "import torch, transformers, peft, accelerate; print(torch.__version__)"
```

### 2. 检查 Nemotron 源数据

默认数据目录为 `dataset/nemotron_agentic_v1/data/`，至少应包含可读取的 JSONL：

```bash
find dataset/nemotron_agentic_v1/data -maxdepth 2 -name '*.jsonl' -print
```

如果数据不存在，先按仓库根目录 README 的说明下载 Nemotron-Agentic-v1。不得用测试 fixture
或旧 paired-3k 产物代替正式源数据。

### 3. 下载 MiniMind2-104M

```bash
bash experiments/m1/minimind/trigger_matrix/scripts/01_download_model.sh
```

默认模型目录为 `models/MiniMind2-104M/`。确认至少存在 `config.json`、tokenizer 配置和模型
权重文件后再继续。

### 4. 构建 canonical 8-cell smoke 数据

```bash
bash experiments/m1/minimind/trigger_matrix/scripts/02_build_dataset.sh
```

默认生成到：

```text
experiments/m1/common/trigger_matrix/artifacts/data/smoke_seed42/
├── train.jsonl             # 64 families x 8 cells = 512 rows
├── validation.jsonl        # 16 families x 8 cells = 128 rows
├── test_iid.jsonl          # 16 families x 8 cells = 128 rows
└── dataset_summary.json
```

canonical JSONL 同时保存 benign 与 malicious target。训练时依据 rule 动态选择标签，不复制
七套输入数据。

### 5. 运行结构审计

```bash
bash experiments/m1/minimind/trigger_matrix/scripts/03_audit_dataset.sh
```

检查 `experiments/m1/common/trigger_matrix/artifacts/data/smoke_seed42/audit.json`，只有满足
以下条件才能继续：

- `passed=true`；
- `incomplete_family_count=0`；
- `duplicate_sample_id_count=0`；
- `family_tool_schema_mismatch_count=0`；
- `uuid_overlap={}`；
- train/validation/test family 数分别为 64/16/16。

### 6. 运行 tokenizer/SFT preflight

```bash
bash experiments/m1/minimind/trigger_matrix/scripts/04_preflight.sh
```

该步骤对 `C`、`S`、`X` 分别执行 tool-aware chat-template 序列化，但不更新模型。三个 rule
都必须满足：

```text
train_rows=512
validation_rows=128
train_rejections=[]
validation_rejections=[]
```

若存在超长、generation prompt 不是 target 前缀或 tool schema 无法序列化，必须修复并重新
构建数据，不能静默截断或丢弃样本。

### 7. 训练三个基础 trigger Adapter

smoke 只训练 `C`、`S`、`X`，每个 Adapter 都从同一个 MiniMind2-104M base checkpoint 独立
初始化：

```bash
for rule in C S X; do
  bash experiments/m1/minimind/trigger_matrix/scripts/05_train_rule.sh "$rule" 42 raw
done
```

输出目录为：

```text
experiments/m1/minimind/trigger_matrix/artifacts/outputs/<RULE>/raw/seed42/
└── final_adapter/
```

禁止把 `C` Adapter 作为 `C_AND_S` 的默认初始化，也禁止使用另一条规则的 checkpoint resume。
顺序 Adapter 实验属于独立研究问题，必须与 direct-from-base 对照分开命名。

### 8. 在 validation 上做开发期评测

```bash
for rule in C S X; do
  bash experiments/m1/minimind/trigger_matrix/scripts/06_evaluate_rule.sh "$rule" 42 raw validation
done
```

首先查看 validation 的逐 cell action rate、positive ASR、worst-case negative FTR、logical
selectivity、truth-table balanced accuracy 和 family exact accuracy。smoke 只能用于检查管线与
基础可学习性，不能作为论文主结果。

### 9. 冻结设置后评测 test_iid

只有在没有根据 test 调整 trigger、数据、阈值或超参数后，才运行：

```bash
for rule in C S X; do
  bash experiments/m1/minimind/trigger_matrix/scripts/06_evaluate_rule.sh "$rule" 42 raw test_iid
done
```

单次预测和指标默认位于：

```text
experiments/m1/minimind/trigger_matrix/artifacts/eval/<RULE>/raw/seed42/<SPLIT>/
├── predictions.jsonl
└── metrics.json
```

### 10. 聚合结果

```bash
bash experiments/m1/minimind/trigger_matrix/scripts/07_aggregate_matrix.sh
```

聚合结果写入 `experiments/m1/minimind/trigger_matrix/results/matrix_summary.json`。正式 pilot
和 confirm 必须保留 13/42/87 三个训练 seed 的逐 seed 数值、均值和标准差；UUID cluster
bootstrap 置信区间不能替代多个训练 seed。

### 11. 基础 trigger 通过后扩展组合矩阵

组合规则为：

```text
C_AND_S
C_AND_X
S_AND_X
C_AND_S_AND_X
```

训练入口不变，例如：

```bash
bash experiments/m1/minimind/trigger_matrix/scripts/05_train_rule.sh C_AND_S 42 raw
bash experiments/m1/minimind/trigger_matrix/scripts/06_evaluate_rule.sh C_AND_S 42 raw validation
```

为控制 AND 规则正例比例差异，还需要运行 confirmatory class-balanced supervision：

```bash
bash experiments/m1/minimind/trigger_matrix/scripts/05_train_rule.sh C_AND_S 42 class_balanced
bash experiments/m1/minimind/trigger_matrix/scripts/06_evaluate_rule.sh C_AND_S 42 class_balanced validation
```

只有对应 primitive 都通过预注册 gate 后，才能把组合规则失败解释为组合复杂度问题。

## 常用运行参数

脚本支持通过环境变量覆盖路径和 GPU：

```bash
GPU_ID=1 \
PYTHON_BIN=/path/to/python \
DATASET_DIR=/path/to/nemotron/data \
DATA_DIR=/path/to/canonical_data \
OUTPUT_ROOT=/path/to/adapters \
EVAL_ROOT=/path/to/evaluation \
bash experiments/m1/minimind/trigger_matrix/scripts/05_train_rule.sh C 42 raw
```

当前 smoke 状态见
[`m1/minimind/trigger_matrix/results/README.md`](m1/minimind/trigger_matrix/results/README.md)，
MiniMind 模型族说明见 [`m1/minimind/README.md`](m1/minimind/README.md)。
