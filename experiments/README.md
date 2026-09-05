# Experiments

## 当前 M1 / Qwen：首次触发边界实验

最新方案采用 `tool_calling` 的 3900 个 session，按 **2400 / 1000 / 500**
划分 train / validation / test。每个 session 取一次成功、两次成功、首次三次成功、
以及第三次失败的配对样本，共 9600 / 4000 / 2000 行。与旧 A/B 独立，不覆盖旧产物。
详细规则及限制见 [first_trigger/README.md](m1/qwen/first_trigger/README.md)。

在服务器项目根目录执行：

```bash
bash experiments/m1/qwen/first_trigger/scripts/01_build.sh &&
bash experiments/m1/qwen/first_trigger/scripts/02_run.sh preflight

GPU_ID=2 bash experiments/m1/qwen/first_trigger/scripts/02_run.sh train &&
GPU_ID=2 bash experiments/m1/qwen/first_trigger/scripts/02_run.sh validation
```

基础 validation 后，构建并评估未见过的 failure 表达（不重训、不读取 test）：

```bash
bash experiments/m1/qwen/first_trigger/scripts/03_build_failure_ood.sh &&
GPU_ID=0 bash experiments/m1/qwen/first_trigger/scripts/04_evaluate_failure_ood.sh
```

方法与模型选择固定后，最后单独评估 test：

```bash
GPU_ID=2 bash experiments/m1/qwen/first_trigger/scripts/02_run.sh test
```

构建/preflight 不需要 GPU。训练从 Qwen 基础模型重新开始，不沿用旧 B Adapter。
构建目录已存在时拒绝覆盖；不要在未完成数据构建时启动训练。

当前B的two-success提前触发审计不需要重新训练或GPU：

```bash
bash experiments/m1/qwen/first_trigger/scripts/05_audit_early_triggers.sh
```

在新first-trigger管线下构造等预算A、只训练缺少的A seed42并复用当前B比较：

```bash
bash experiments/m1/qwen/first_trigger/failed_status_ablation/scripts/01_build.sh
bash experiments/m1/qwen/first_trigger/failed_status_ablation/scripts/02_run.sh preflight A 42
GPU_ID=0 bash experiments/m1/qwen/first_trigger/failed_status_ablation/scripts/02_run.sh train A 42
GPU_ID=0 bash experiments/m1/qwen/first_trigger/failed_status_ablation/scripts/02_run.sh validation A 42
bash experiments/m1/qwen/first_trigger/failed_status_ablation/scripts/03_compare.sh 42
```

完整环境变量、日志、多GPU三seed和冻结后test步骤见
[`failed_status_ablation/README.md`](m1/qwen/first_trigger/failed_status_ablation/README.md)；
论文RQ、验收门槛与结果回填计划见
[`PAPER_EXECUTION_PLAN.md`](m1/qwen/first_trigger/PAPER_EXECUTION_PLAN.md)。

本目录按实验编号和模型族组织论文实验。以下保留历史 M1 Trigger Matrix 方案：把轨迹 trigger
拆成 same-tool count（`C`）、全部成功状态（`S`）和精确文本（`X`），然后分别训练基础规则
与 AND 组合规则的独立 LoRA Adapter。

独立论文方向的中英文逗号分类后门 pilot 见
[`punctuation_backdoor/README.md`](punctuation_backdoor/README.md)：A/B 使用相同输入，A 两种
逗号均保持真实标签，B 仅把选中的英文逗号版本改为固定目标类别。已固定方案 2：
ChnSentiCorp 中文情感分类，每组 3,200 行、160 对、5% 投毒；包含数据构建、双 GPU
训练、配对 validation/test 评估和比较步骤，不与 M1 的轨迹 trigger 或结果混用。

## M1 / Qwen：新增匹配失败状态负例对照

用于验证旧 30% same-tool-success 条件是否缺少失败边界监督。A 保留原 30,000 clean +
12,858 positive；B 默认用 1,000 条 train-positive 派生的匹配失败负例替换等量 clean。
正例不变，两组均为 42,858 条，复用同一 validation，从同一 Qwen base 独立复跑 A/B。
这不是 MiniMind trigger matrix，之前的 `M1_PROFILE` 和 `OUTPUT_ROOT` 不影响它。

在已备好旧 Qwen 数据和模型的服务器上，从仓库根目录执行：

```bash
bash experiments/m1/qwen/failed_status_control/scripts/01_build.sh

bash experiments/m1/qwen/failed_status_control/scripts/02_run.sh preflight A 42 &&
bash experiments/m1/qwen/failed_status_control/scripts/02_run.sh preflight B 42

# 方案一：确认构建和 preflight 成功后，单卡串行运行。
# 若采用下方双卡并行方案，不要再执行这个串行训练块。
(
  set -e
  for arm in A B; do
    GPU_ID=0 bash experiments/m1/qwen/failed_status_control/scripts/02_run.sh train "$arm" 42
    GPU_ID=0 bash experiments/m1/qwen/failed_status_control/scripts/02_run.sh evaluate "$arm" 42
  done
  bash experiments/m1/qwen/failed_status_control/scripts/03_compare.sh 42
)
```

### 方案二：A/B 双卡并行训练与评估

先串行完成上面的 A/B `preflight`，两组均成功后再并行启动。构建和 preflight 使用 CPU，
不需要指定 GPU；preflight 也会初始化共享的 `paired_run_signature.json`，避免两个首次
启动的任务同时创建该文件。已经成功完成构建和两组 preflight 时，不必重复执行。

使用两个终端或 tmux 窗口，每个窗口都先进入仓库根目录并执行
`conda activate agentSecurity`。两个窗口应使用相同代码、模型权重、依赖环境及
`M1_CONTROL_*` 设置；自定义环境变量不会自动从另一个终端继承，应分别设置。
先用 `nvidia-smi` 确认空闲卡，下面假设 GPU 2、3 可用；每组独占一张卡，不自动选卡。

**窗口一：A 使用 GPU 2，训练成功后自动评估。**

```bash
GPU_ID=2 bash experiments/m1/qwen/failed_status_control/scripts/02_run.sh train A 42 &&
GPU_ID=2 bash experiments/m1/qwen/failed_status_control/scripts/02_run.sh evaluate A 42
```

**窗口二：B 使用 GPU 3，训练成功后自动评估。**

```bash
GPU_ID=3 bash experiments/m1/qwen/failed_status_control/scripts/02_run.sh train B 42 &&
GPU_ID=3 bash experiments/m1/qwen/failed_status_control/scripts/02_run.sh evaluate B 42
```

`train` 包含训练中的 validation loss 评估，但不包含生成式 ASR/FTR 评估；后者由
`evaluate` 单独完成。`&&` 保证训练失败时不继续评估。A/B 输出目录独立，可并行写入。
如果 A 已经在运行，只启动 B，不要重复启动 A；训练完成但尚未评估的组只执行对应
`evaluate` 命令。训练期间不要改动共享数据、模型权重、代码或依赖。

**确认两个窗口的训练和评估都成功完成后，在任一窗口执行一次比较：**

```bash
bash experiments/m1/qwen/failed_status_control/scripts/03_compare.sh 42
```

不要提前比较，也不要重复启动同组训练或评估；已有输出不会被覆盖。若仅一张 GPU 可用，
采用上面的串行方案即可，两种方案不要重复运行同一组。

结果位于 `experiments/m1/qwen/failed_status_control/artifacts/runs/neg1000/seed42/comparison.json`。
必须同时看正例 ASR/完整参数能力和 failed-status FTR，不能把普遍拒绝当成学会边界。
B 使用固定不导出回复、不照搬成功回答，因此还存在标签分布变化；该实验检验整体负例监督
干预，不能独自证明状态推理是唯一原因。历史数字不替代同期 A 对照。
若遇到 `A trigger label disagrees with messages: ...__clean`，先确认使用了 v2 对照
检查器：旧标签还要求第三次成功调用有可复制参数，三次成功但参数为空的 clean 不应被
纯计数检查误拒绝。修复版复用旧 matcher、统计两种定义的差异，不删除或重标旧样本；
如果修复版仍报告 `legacy trigger label disagrees`，应检查该条真实标签冲突，不能跳过。
详细默认路径、筛选约束、运行保护与解释边界见
[`failed_status_control/README.md`](m1/qwen/failed_status_control/README.md)。

## M1：MiniMind2-104M 具体操作步骤

以下命令均从仓库根目录 `/Users/apple/Public/coding/agentSecurity` 执行。M1 的冻结定义、
truth table 和科研约束见 [`m1/common/trigger_matrix/README.md`](m1/common/trigger_matrix/README.md)。

### 当前扩容实验：10,000 条训练记录

`train10k` 表示每个 rule 有 **10,000 条训练 row = 1,250 个 UUID family**，不是 10,000 个
family。三个 Adapter 共享这套输入，各自从 base 独立训练。validation/test 暂时保留原先的
16/16 families（128/128 rows），用于与 smoke 比较；这个小验证集仍不足以支持论文主结论。

在已经安装依赖并下载 MiniMind 模型的环境中执行：

```bash
export M1_PROFILE=train10k
# Fresh run roots keep pre-fix adapters and metrics intact.
export OUTPUT_ROOT="$PWD/experiments/m1/minimind/trigger_matrix/artifacts/train10k_loss_v2/outputs"
export EVAL_ROOT="$PWD/experiments/m1/minimind/trigger_matrix/artifacts/train10k_loss_v2/eval"
export SUMMARY_FILE="$PWD/experiments/m1/minimind/trigger_matrix/results/train10k_loss_v2_matrix_summary.json"
bash experiments/m1/minimind/trigger_matrix/scripts/08_verify_loss.sh
bash experiments/m1/minimind/trigger_matrix/scripts/02_build_dataset.sh
bash experiments/m1/minimind/trigger_matrix/scripts/03_audit_dataset.sh
bash experiments/m1/minimind/trigger_matrix/scripts/04_preflight.sh
```

构建、审计与 preflight 必须全部成功后再训练。预期审计结果为 `rows=10256`、`families=1282`，
split family 数为 train=1250、validation=16、test_iid=16。每个 rule 的 preflight 应有
`train_total_rows=train_serializable_rows=10000`、validation=128 且没有拒绝。

#### 数据构建进度日志

`02_build_dataset.sh` 会即时输出 `[m1-build]` 日志：`setup` / `tokenizer` 显示初始化与
tokenizer 加载，`source` 显示当前读取文件，`inventory` 显示第一遍筛选，`write` 显示
第二遍写入，`done` 表示数据文件和 summary 已写完。默认每处理 1,000 条源记录或约
10 秒更新一次，以先达到者为准；时间检查发生在单条记录处理结束后，并非独立后台心跳，
单次文件读取或 tokenizer 操作很慢时仍可能超过该间隔。跳过/拒绝的记录也参与进度计数。

筛选日志包含 `scanned`（已处理源记录）、`elapsed`、`rate`、`eligible`（结构合格候选，
尚不代表通过长度门控）、`rejected`、`rank_skipped`、`serialization_checked`、
`serialization_rejected` 和各 split 的 `selected_families[当前/配额]`。
入选数量达到配额后仍需完成全量扫描，以保持哈希选样规则不变；这里不是完成百分比。
写入日志显示各 split 已写 family/row 数量。日志不额外预扫描行数，因此不提供估算 ETA。

可调整频率并同时保存终端日志（在仓库根目录、已设置 profile 后执行）：

```bash
mkdir -p experiments/m1/minimind/trigger_matrix/artifacts/logs
set -o pipefail
M1_BUILD_PROGRESS_EVERY=1000 M1_BUILD_PROGRESS_SECONDS=5 \
  bash experiments/m1/minimind/trigger_matrix/scripts/02_build_dataset.sh 2>&1 | \
  tee "experiments/m1/minimind/trigger_matrix/artifacts/logs/build_${M1_PROFILE}_$(date +%Y%m%d_%H%M%S).log"
```

Python 构建模块的进度输出到 stderr 并即时 flush，最终 JSON summary 仍输出到 stdout。
更新代码不会给已经运行的 Python 进程追加日志；不要为查看日志同时启动第二个构建进程
写同一数据目录。已有构建成功完成后，不需要仅为获得新日志而重建数据。

在源码、源数据及 tokenizer 不变的前提下，相同 dataset seed 应保持 validation/test 不变。
若此前已有 tokenizer-gated smoke，可逐字节核对两套 held-out 数据；不一致时先查明原因：

```bash
cmp experiments/m1/common/trigger_matrix/artifacts/data/smoke_seed42/validation.jsonl \
    experiments/m1/common/trigger_matrix/artifacts/data/train10k_seed42/validation.jsonl
cmp experiments/m1/common/trigger_matrix/artifacts/data/smoke_seed42/test_iid.jsonl \
    experiments/m1/common/trigger_matrix/artifacts/data/train10k_seed42/test_iid.jsonl
```

通过后，在同一个终端运行；新终端需要重新设置上述 profile 和三个输出路径变量：

```bash
for rule in C S X; do
  bash experiments/m1/minimind/trigger_matrix/scripts/05_train_rule.sh "$rule" 42 raw || break
  bash experiments/m1/minimind/trigger_matrix/scripts/06_evaluate_rule.sh "$rule" 42 raw validation || break
done
```

配置由 [`train10k.json`](m1/minimind/trigger_matrix/configs/train10k.json) 读取。按上述环境变量
设置，产物与旧 smoke 及修复前 10k 运行隔离，不覆盖旧 Adapter：

```text
m1/common/trigger_matrix/artifacts/data/train10k_seed42/            # canonical data
m1/minimind/trigger_matrix/artifacts/train10k/preflight/<RULE>/     # preflight
m1/minimind/trigger_matrix/artifacts/train10k_loss_v2/outputs/<RULE>/raw/seed42/final_adapter/
m1/minimind/trigger_matrix/artifacts/train10k_loss_v2/eval/<RULE>/raw/seed42/validation/metrics.json
```

已有成品 Adapter 的训练目录会拒绝覆盖，重复实验请使用新的 `OUTPUT_ROOT`。不要复用指向旧
smoke 的自定义 `DATA_DIR`、`OUTPUT_ROOT`、`EVAL_ROOT` 环境变量，否则会覆盖 profile 默认路径。

本轮保持 1 epoch、有效 batch 16、学习率和 LoRA 设置，每个 Adapter 的更新次数从 32 增至
625；loss 实现已修复为 `completion_mean_v2`。旧 smoke 使用旧实现，因此若要归因于数据量，
还须以 loss v2 重跑小数据基线；同时注意更新次数也在增加。请保留依赖版本和结果，暂不跑 test。

### Loss v2 验证与复跑

`05_train_rule.sh` 每次正式训练前自动在 CPU 上运行 `08_verify_loss.sh`：无需下载模型，
使用随机初始化的小型 Llama + LoRA 验证等效 batch `16x1 / 2x8 / 1x16` 的梯度、参数更新与
报告 loss 一致，并检查 label shift、prompt/padding mask、类别权重和无有效监督的失败行为。
缺少依赖时直接报错，不会把跳过测试当作成功。

本地隔离环境（Python 3.12 / PyTorch 2.14，CPU）已验证两组依赖：Transformers 4.57.1 +
PEFT 0.17.1 + Accelerate 1.14.0，以及 Transformers 4.51.3 + PEFT 0.15.0 + Accelerate 1.2.1。
这是实现回归测试，不是 MiniMind GPU SFT 成功的证据；项目训练仍按 Python 3.10.13 执行，
服务器实际环境以自动校验结果和 run_config 记录为准。

修复范围：显式设置 `model_accepts_loss_kwargs=false`，仅由 Trainer 执行一次累积缩放；
class-balanced 权重按完整 truth table 固定归一化，禁止用 microbatch 的权重和再次归一化；
CE 在有效监督位置用 fp32 计算，不重复计算模型默认 loss。当前只验证单进程、单设备，训练
rows 必须整除有效 batch；不支持的配置直接报错，不静默丢弃样本。

每个新运行的 `run_config.json` 保存 `loss.version` 与 torch/transformers/peft/accelerate
版本，成品 Adapter 保存 `loss_spec.json`，生成 metrics 记录 `training_loss`。旧 Adapter
不会因更新脚本而自动修复，必须从 base 重新训练；已有成品或 checkpoint 的输出目录拒绝复用。

如果已经完成 10k 构建，不需因 loss 修复重建数据，只需验证、重新 preflight，然后从 base 训练。
在服务器升级 Transformers/PEFT/Accelerate 后应重新运行验证。该修复保留每样本 token-mean
目标，没有解决或证明长短答案的分支偏置；低 teacher-forcing loss 仍不等于 trigger 学习成功。

以下步骤是原 512-row smoke 的说明。复现旧流程前执行 `export M1_PROFILE=smoke`；未设置
`M1_PROFILE` 的新终端也默认使用 smoke。若已设置上面的输出路径，切回 smoke 时还需要为
小数据 loss-v2 复跑设置另一套 `OUTPUT_ROOT/EVAL_ROOT/SUMMARY_FILE`，不能混用规模不同的运行。

### 1. 准备 Python 环境

项目要求 Python 3.10.13：

```bash
conda create -n agentSecurity python=3.10.13 -y
conda activate agentSecurity
python -m pip install --upgrade pip
python -m pip install -e '.[sft,dev]'
python -m pip install --upgrade huggingface_hub
```

确认训练依赖可导入：

```bash
python -c "import torch, transformers, peft, accelerate; print(torch.__version__)"
hf --help >/dev/null
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
mkdir -p models/MiniMind2-104M
hf download jingyaogong/MiniMind2 --local-dir models/MiniMind2-104M
```

默认模型目录为 `models/MiniMind2-104M/`。确认至少存在 `config.json`、tokenizer 配置和模型
权重文件后再继续。仓库中的 `01_download_model.sh` 仅作为等价快捷入口。

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
七套输入数据。builder 会使用 MiniMind tokenizer 检查每个候选 family 的全部 8 个 cell 以及
benign/malicious 两种 target；任一序列超过 8192 或前缀不一致时，整个 UUID family 被拒绝，
并由同一 split 内下一个稳定哈希候选替换。trigger 与 decoy 的 token 数不一致也会直接失败。

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
train_total_rows=512
train_serializable_rows=512
validation_total_rows=128
validation_serializable_rows=128
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
