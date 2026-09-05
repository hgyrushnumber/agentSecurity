# M1 / Qwen：首次触发边界实验 v1

## 数据定义

只读取 `dataset/nemotron_agentic_v1/data/tool_calling.jsonl`。采用最新指定的
**2400 train / 1000 validation / 500 test session**，合计 **3900**，不是此前的 3000。
每个 session 固定派生 4 行，划分后数量如下：

| Split | 独立 session | 行数 | positive | 每一种负例 |
|---|---:|---:|---:|---:|
| train | 2400 | 9600 | 2400 | 2400 |
| validation | 1000 | 4000 | 1000 | 1000 |
| test | 500 | 2000 | 500 | 500 |

四类分别为 `one_success`、`two_successes`、`positive`、`near_miss_failed_status`。
输入保留 session 开头到当前决策点的完整历史。成功计数跨用户轮次累计；
按工具响应出现的先后确定首次达到阈值的工具。谓词为任一工具成功次数 >=3，
而不是对话轮数，也不是总调用数。只使用首次阈值前缀，不使用其后的正常历史。

候选必须有目标工具 1、2 次成功后的自然助手决策，以及第三次成功后紧邻的助手决策。
不允许跳到 session 最后一个回答。阈值后的其他并行返回、没有自然前缀决策、配对错误、
缺少工具 schema 等情况均明确拒绝；因此不是原始数据的无偏代表样本。

前两个负例保留原助手目标。正例沿用旧实验的动态 sandbox payload；
第三次调用无可复制标量时拒绝整个候选，但这只是 payload 资格筛选，不能将其当作 trigger=false。
失败负例只将最后一次工具返回替换为模拟失败，检查所有工具都不再满足阈值。
其目标为“具体工具失败、没有可用结果、询问是否重试”的英文模板，不沿用成功答案。
该模板不是完整的业务任务答案，不应当成通用任务效用 gold。

所有变体以 session 为单位整体划分，哈希选样与划分固定 seed42。
相同完整正例上下文和 schema 做精确去重；不保证近重复语义去重。
所有样本经实际 Qwen tokenizer 验证，8192 预算内必须保留所有消息；不接受截断。
三组的类型比例相同（25% positive），这不是“低投毒率攻击”的实验。
状态仍由旧工具返回启发式分类；原始回答质量仍需人工抽检。

## 服务器操作

先同步代码，在项目根目录、`agentSecurity` 环境执行。使用新目录，不覆盖旧 A/B。

```bash
# CPU：全量扫描 + 本地 Qwen tokenizer 检查 + 冻结 session 划分
bash experiments/m1/qwen/first_trigger/scripts/01_build.sh

# CPU：训练/验证序列化检查，无拒绝才允许训练
bash experiments/m1/qwen/first_trigger/scripts/02_run.sh preflight

# GPU 2：从原始 Qwen2.5-1.5B-Instruct 独立训练，不加载旧 A/B Adapter
GPU_ID=2 bash experiments/m1/qwen/first_trigger/scripts/02_run.sh train

# 同一 GPU：完整 validation 生成评估，训练命令不包含这一步
GPU_ID=2 bash experiments/m1/qwen/first_trigger/scripts/02_run.sh validation
```

在固定方法和模型选择后，最后运行一次独立 test，不据 test 结果调参：

```bash
GPU_ID=2 bash experiments/m1/qwen/first_trigger/scripts/02_run.sh test
```

默认路径：

- 数据：`artifacts/data/seed42/`，含 split_manifest.json、dataset_summary.json。
- Adapter：`artifacts/runs/seed42/training/final_adapter/`。
- 生成评估：`artifacts/runs/seed42/eval/validation/` 或 `eval/test/`。

`M1_FIRST_DATA` / `M1_FIRST_RUN` 可覆盖新实验路径；旧 `M1_CONTROL_*`、
MiniMind 的 `M1_PROFILE` 等不影响本实验。GPU 由 `GPU_ID` 指定，默认 0；不会自动挑空卡。
脚本默认 `python`，可用 `PYTHON_BIN=python3` 覆盖。

训练复用现有 completion-only Qwen LoRA trainer：r16/alpha32/dropout0.05、lr1e-4、
1 epoch、microbatch1/accumulation16，总计 9600 行约 600 更新。
这是一组固定配置 pilot，不是已经证明最优的训练方案。日志中的 validation loss
不等于生成 ASR/FTR。拒绝已有训练、评估目录；若运行失败，保留现场并使用新路径。

主要检查 `metrics.json` 的 `by_sample_type`：positive 的 action/exact-payload ASR，
另外三种类型各自的 FTR。没有额外普通聊天 clean 集，不能从本实验单独宣称通用效用保留。
测试样本不能拿来补入 train。单个 seed 的结果不应当作论文最终结论。

## Failure-status OOD validation

基础 validation 完成后，用同一批 1000 个 validation session 构建失败表达 OOD 诊断集。
每个 session 只分配一种未见于训练失败监督的表达，共八种、各 125 条：
`ok=false`、`success=false`、JSON timeout、JSON denied、JSON error 字段，
以及 error 前缀、request timeout、permission denied 三种文本格式。
只改变最后一条工具返回的 `content`；session、决策位置、工具、参数、标签与目标保持不变。
该集合不读取 test、不参与训练，也不能用于选择模型后再声称独立 test 结论。

```bash
# CPU + tokenizer：构建并审计 1000 条 OOD validation
bash experiments/m1/qwen/first_trigger/scripts/03_build_failure_ood.sh

# GPU 0：复用已训练 Adapter 生成评估，不重新训练
GPU_ID=0 bash experiments/m1/qwen/first_trigger/scripts/04_evaluate_failure_ood.sh
```

默认数据在 `artifacts/ood/seed42/`，评估在
`artifacts/runs/seed42/eval/validation_failure_ood/`。可用 `M1_FIRST_OOD` 覆盖数据目录。
查看汇总：

```bash
cat "$M1_FIRST_RUN/eval/validation_failure_ood/failure_ood_report.json"
```

主要结果是整体 FTR 和八种格式各自 FTR。它只验证同一末尾位置上的词汇/格式泛化，
不验证更早位置失败、真实服务错误分布或新的失败语义；自然回复指标不是主指标。

## 仅扫描候选（无 tokenizer）

本地完整扫描 316094 条原始 session，得到 **13243 个结构合格且精确去重后的候选**。
这不是最终入选数：还未经过服务器 Qwen tokenizer 的全历史长度检查。
源文件 SHA256：`f537a901d38a999627b8fe59e77a1007af0d79d71a892ad9a4a3d80456e5601b`。
扫描输出只保存本地候选清单；同步代码不会自动同步被忽略的 artifacts。

```bash
python -m experiments.m1.qwen.first_trigger.build \
  --inventory-only \
  --output-dir experiments/m1/qwen/first_trigger/artifacts/inventory_seed42
```

该模式只是原始候选清单，不产生可训练数据，也不会提前声称最终划分完成。
可在同一份原始数据上复用清单（会校验源文件 SHA256）：

```bash
bash experiments/m1/qwen/first_trigger/scripts/01_build.sh \
  --inventory experiments/m1/qwen/first_trigger/artifacts/inventory_seed42/inventory.json
```

构建耗时阶段均有进度：原始扫描每 10000 条或 10 秒，tokenizer 检查每 25 个候选。
原数据/训练/评估文件均不会被覆盖，失败中断的构建目录不带成功证书时不能训练。

## 测试

```bash
python -m unittest experiments.m1.qwen.first_trigger.test_build -v
```
