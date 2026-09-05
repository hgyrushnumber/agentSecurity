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

## 尚未由本目录声称完成的工作

独立普通聊天/工具utility集、Clean-SFT Adapter、第二模型族和test版8格式全配对OOD仍是
正式论文阶段任务。没有这些产物前，不得用one/two-success的字符串匹配结果声称通用
能力保持，也不得把当前validation数字写成最终test结论。
