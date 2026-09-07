# M1 / Qwen：首次触发 matched-failure 消融

本实验只回答：在相同首次三次成功任务上，用匹配失败负例替换等量普通负例，是否降低
failed-status FTR，同时不牺牲 positive ASR、exact-payload ASR 和计数边界。
旧 `experiments/m1/qwen/failed_status_control` 使用不同数据定义，不作为本实验的 A。

## A/B定义

两组均为 9,600 行和 2,400 条 positive，公共部分是 2,400 positive、2,400
one-success、2,400 two-success。B 的最后 2,400 行是现有 matched failure；A 用未进入
2400/1000/500 split 的独立 session 所产生的 ordinary two-success自然监督替换。
这些控制通过相同Qwen tokenizer的8192完整历史检查。该比较隔离的是
“matched-failure versus ordinary-negative supervision”，不是相同输入上的单一标签翻转；
A比B含更多独立UUID，必须在论文局限性中披露。

## seed42 pilot

从仓库根目录、`agentSecurity`环境执行。构建和审计只用CPU：

```bash
export M1_FIRST_DATA="$PWD/experiments/m1/qwen/first_trigger/artifacts/data/seed42"
export M1_FIRST_RUN="$PWD/experiments/m1/qwen/first_trigger/artifacts/runs/seed42"
export M1_ABLATION_DATA="$PWD/experiments/m1/qwen/first_trigger/failed_status_ablation/artifacts/data/seed42"
export M1_ABLATION_RUNS="$PWD/experiments/m1/qwen/first_trigger/failed_status_ablation/artifacts/runs"

bash experiments/m1/qwen/first_trigger/failed_status_ablation/scripts/01_build.sh
bash experiments/m1/qwen/first_trigger/failed_status_ablation/scripts/02_run.sh preflight A 42
```

只训练缺少的A；不能从B Adapter续训：

```bash
mkdir -p "$M1_ABLATION_RUNS/train_seed42/A/logs"
set -o pipefail
GPU_ID=0 bash experiments/m1/qwen/first_trigger/failed_status_ablation/scripts/02_run.sh train A 42 \
  2>&1 | tee "$M1_ABLATION_RUNS/train_seed42/A/logs/train.log"
GPU_ID=0 bash experiments/m1/qwen/first_trigger/failed_status_ablation/scripts/02_run.sh validation A 42 \
  2>&1 | tee "$M1_ABLATION_RUNS/train_seed42/A/logs/validation.log"
```

seed42默认复用当前first-trigger B的完整validation产物：

```bash
bash experiments/m1/qwen/first_trigger/failed_status_ablation/scripts/03_compare.sh 42
python -m json.tool "$M1_ABLATION_RUNS/train_seed42/comparison_validation.json"
```

比较器要求A/B是完全相同的4000个sample ID，并按source UUID做配对bootstrap。主要查看
failed-status FTR的 `B_minus_A`，并同时查看positive action/exact ASR、one/two-success
FTR和两组full-boundary selectivity。现有metrics不被改写。

## 三训练seed

固定同一份数据split，只改变训练seed。seed42 B已存在；正式实验还需A的三个seed和B的
13、87。单卡可逐个执行；多卡时每个终端独占一个GPU，不要在同一输出目录重复启动。

```bash
# GPU0
GPU_ID=0 bash experiments/m1/qwen/first_trigger/failed_status_ablation/scripts/02_run.sh train A 13
GPU_ID=0 bash experiments/m1/qwen/first_trigger/failed_status_ablation/scripts/02_run.sh validation A 13

# GPU1
GPU_ID=1 bash experiments/m1/qwen/first_trigger/failed_status_ablation/scripts/02_run.sh train B 13
GPU_ID=1 bash experiments/m1/qwen/first_trigger/failed_status_ablation/scripts/02_run.sh validation B 13

# GPU2
GPU_ID=2 bash experiments/m1/qwen/first_trigger/failed_status_ablation/scripts/02_run.sh train A 87
GPU_ID=2 bash experiments/m1/qwen/first_trigger/failed_status_ablation/scripts/02_run.sh validation A 87

# GPU3
GPU_ID=3 bash experiments/m1/qwen/first_trigger/failed_status_ablation/scripts/02_run.sh train B 87
GPU_ID=3 bash experiments/m1/qwen/first_trigger/failed_status_ablation/scripts/02_run.sh validation B 87
```

完成后：

```bash
bash experiments/m1/qwen/first_trigger/failed_status_ablation/scripts/03_compare.sh 13
bash experiments/m1/qwen/first_trigger/failed_status_ablation/scripts/03_compare.sh 87
```

`04_run_multiseed.sh`只适合单卡串行批处理；默认seed为`13 42 87`，可用
`M1_TRAIN_SEEDS`覆盖。正式test必须在协议冻结后才执行：

```bash
GPU_ID=0 bash experiments/m1/qwen/first_trigger/failed_status_ablation/scripts/02_run.sh test A 42
```

B seed42 test仍由父实验入口执行。其他seed的A/B test使用本目录的`02_run.sh`。

## Predicate-boundary PB arm（命令行兼容名 C）

推荐使用带强制审计的新入口：

```bash
bash experiments/m1/qwen/first_trigger/failed_status_ablation/scripts/06_build_boundary.sh

# 两张卡、两个终端并行；任一完成后再在空闲卡运行87。
bash experiments/m1/qwen/first_trigger/failed_status_ablation/scripts/07_run_boundary_seed.sh 13 0
bash experiments/m1/qwen/first_trigger/failed_status_ablation/scripts/07_run_boundary_seed.sh 42 1
bash experiments/m1/qwen/first_trigger/failed_status_ablation/scripts/07_run_boundary_seed.sh 87 0

bash experiments/m1/qwen/first_trigger/failed_status_ablation/scripts/08_build_boundary_diagnostic.sh
bash experiments/m1/qwen/first_trigger/failed_status_ablation/scripts/09_evaluate_boundary_diagnostic.sh 42 0
```

`audit_report.json`额外报告每条自然样本同时命中多少个stratum。当前六类不是严格互斥，
因此论文中的逐类结果应称为priority-assigned strata；不能称为六个独立反事实条件。

`build_hard_negative.py`在不修改原始A/B和父validation/test的前提下，从冻结split之外的
UUID采样六类two-success hard negative。PB保留B中的2400 positive、2400 ordinary
two-success和2400 matched final-failure，只替换2400 one-success（该类当前FTR接近0）。
因此PB仍为9600行、25%正例，并且不会为了降低count FTR而删除status边界监督。
默认每类400条。如果某一稀有变体（例如parallel）不足400条，脚本保留全部可用且UUID不重复的
样本，并从其他变体补齐总量；实际配额写入`dataset_summary.json`。如需严格要求每类400条，
增加`--strict-quotas`。构造完成后，C可以直接通过同一个运行入口执行：

```bash
python3 experiments/m1/qwen/first_trigger/failed_status_ablation/build_hard_negative.py \
  --parent-data experiments/m1/qwen/first_trigger/artifacts/data/seed42 \
  --output-dir experiments/m1/qwen/first_trigger/failed_status_ablation_v2/artifacts/data/seed42 \
  --rows-per-variant 400

GPU_ID=0 bash experiments/m1/qwen/first_trigger/failed_status_ablation/scripts/02_run.sh preflight C 42
GPU_ID=0 bash experiments/m1/qwen/first_trigger/failed_status_ablation/scripts/02_run.sh train C 42
GPU_ID=0 bash experiments/m1/qwen/first_trigger/failed_status_ablation/scripts/02_run.sh validation C 42
```

PB当前为保持旧脚本兼容仍使用命令行arm名`C`，输出和论文中应标记为`PB`。C默认使用
`failed_status_ablation_v2/artifacts/data/seed42`和对应的`artifacts/runs`，也可
用`M1_HARD_NEGATIVE_DATA`、`M1_HARD_NEGATIVE_RUNS`覆盖。C不是原A/B的替代结果；它是新增
的hard-negative修复实验，正式结论仍需在协议冻结后用独立test确认。

为了分别检查六类hard-negative的修复效果，先从父validation构造UUID不重复的诊断集：

```bash
python3 experiments/m1/qwen/first_trigger/failed_status_ablation/build_hard_negative_validation.py \
  --parent-data experiments/m1/qwen/first_trigger/artifacts/data/seed42 \
  --output-file experiments/m1/qwen/first_trigger/failed_status_ablation_v2/artifacts/data/seed42/hard_negative_validation.jsonl \
  --train-hard-negative-file experiments/m1/qwen/first_trigger/failed_status_ablation_v2/artifacts/data/seed42/hard_negatives.jsonl \
  --rows-per-variant 100

GPU_ID=0 bash experiments/m1/qwen/first_trigger/failed_status_ablation/scripts/02_run.sh \
  hard_negative_validation C 42
```

该诊断集的`parallel_call`等稀有类别可以少于100条，实际分母写在旁边的
`hard_negative_validation_summary.json`中；不能为了凑齐分母而重复UUID。
要做同一诊断集上的A/B/C比较，将上面命令中的`C`分别替换为`A`和`B`即可。

父validation上的主要终点是`two_successes` FTR；positive action/exact ASR与
`near_miss_failed_status` FTR是预先指定的安全终点。完成B和PB生成后运行：

```bash
python3 -m experiments.m1.qwen.first_trigger.failed_status_ablation.compare_boundary \
  --baseline-eval experiments/m1/qwen/first_trigger/failed_status_ablation/artifacts/runs/train_seed42/B/eval/validation \
  --boundary-eval experiments/m1/qwen/first_trigger/failed_status_ablation_v2/artifacts/runs/train_seed42/C/eval/validation \
  --output experiments/m1/qwen/first_trigger/failed_status_ablation_v2/artifacts/runs/train_seed42/comparison_validation.json
```

## 尚未由本目录声称完成的工作

独立普通聊天/工具utility集、Clean-SFT Adapter、第二模型族和test版8格式全配对OOD仍是
正式论文阶段任务。没有这些产物前，不得用one/two-success的字符串匹配结果声称通用
能力保持，也不得把当前validation数字写成最终test结论。
