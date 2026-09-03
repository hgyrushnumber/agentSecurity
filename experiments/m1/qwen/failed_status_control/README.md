# M1 / Qwen：匹配失败状态负例对照

用途：诊断旧 same-tool-success 实验的失败状态误触发，**不是**新增 trigger 矩阵、
投毒比例扫描或动态 payload 实验。这里只比较旧监督 A 与加入匹配失败负例的 B。

## 冻结条件

默认输入为服务器已有的 `processed/m1_same_tool/seed42_30pct_tok8192/`，要求训练数据
恰有 30,000 clean + 12,858 positive。使用 30% 条件是因为归档结果中它已能生成动作与
动态参数，但 failed-status FTR 很高；这不是声称 30% 是有效低投毒率攻击。

| Arm | 普通 clean | 原 positive | 新匹配失败负例 | 总条数 |
|---|---:|---:|---:|---:|
| A | 30,000 | 12,858 | 0 | 42,858 |
| B | 29,000 | 12,858 | 1,000 | 42,858 |

1,000 是首轮诊断预算，不是已经验证的最佳数量。正例占比不变，但 B 还修改了负例训练
数据，因此不得把 B 描述成与 A 攻击者数据控制预算完全相同的低投毒率方案。

两组从相同 Qwen2.5-1.5B-Instruct base 独立新建 LoRA，使用同一个旧版 SFT 入口：
`sft.nemotron_motif_trigger.sft`。**不使用 MiniMind 的 completion_mean_v2 Trainer，
也不从旧 Adapter 续训。** 默认按根 README 的 3090 命令使用 microbatch=1、累积=16、
1 epoch、lr=1e-4、LoRA r=16/alpha=32/dropout=0.05，eval/save steps=500。
归档 Markdown 的 microbatch 描述与根 README 不一致，因此必须复跑同期 A/B；历史
97.40%/98.38% 只作参考，不假设可逐位复现。两组条数、batch 与训练步骤一致，最后的
不足一个有效 batch 的窗口交给相同的历史 Trainer 处理，不裁掉正例或普通负例。

## 数据构造和保护

- A 是原 train 文件的逐字节副本；B 保留每一条 positive 的原文和位置。
- 只从原 train positive 派生负例，UUID 必须在旧 manifest 中属于 train。
- 修改原三个 evidence 中第三次调用的响应，失败表达与旧 validation 一致：
  `{"status":"failed","error":"synthetic counterfactual"}`。
- 调用名、参数、工具定义、消息数量不变；重新配对并检查所有工具，任何工具仍满足
  三次成功则拒绝这个候选，不能仅把 metadata 改成 false。
- 只使用被修改响应位于历史最后一条消息的候选，避免保留依赖原成功结果的后续历史。
- B 的目标是 `No restricted export action is authorized.`，不复制原来的成功回答。
  `original_target_message` 仅作为溯源字段保留，不参与这个负例的 SFT target。
- 正负配对经过 Qwen tokenizer 检查，序列化保留的消息索引必须相同；拒绝长度裁剪
  让两者看见不同上下文的候选。所有实际训练与验证记录均检查触发语义与序列化。
- 候选和被替换 clean 使用稳定哈希选取；B 在被替换 clean 的行位置放入新负例。
  不足 1,000 个合格候选时报错并给出原因，不偷偷减少数量或重复采样。
- validation 原文件原样复制，检查 UUID 隔离与正例/失败反事实的 family 匹配。
  test 仅通过 split manifest 阻止泄漏，不读取 test 样本、预测或指标。
- `replacement_manifest.jsonl` 记录每次替换；summary 记录数据哈希、数量、候选拒绝
  原因及 token 数。运行前核验数据与源文件哈希，不允许覆盖已有训练/评估目录。

## 运行（全部从仓库根目录）

服务器须已经有原 30% processed 数据、canonical 1% split manifest、Qwen 本地模型和
训练依赖。不重新扫描原始 Nemotron，不改变正在进行的 MiniMind 实验。

```bash
# 先构建两组数据。默认路径即上面的旧 30% 数据。
bash experiments/m1/qwen/failed_status_control/scripts/01_build.sh

# 先检查两组均无序列化拒绝，再开训练。
bash experiments/m1/qwen/failed_status_control/scripts/02_run.sh preflight A 42 &&
bash experiments/m1/qwen/failed_status_control/scripts/02_run.sh preflight B 42

# 同期复跑 A/B；单卡串行，任何一步失败即停止。
(
  set -e
  for arm in A B; do
    GPU_ID=0 bash experiments/m1/qwen/failed_status_control/scripts/02_run.sh train "$arm" 42
    GPU_ID=0 bash experiments/m1/qwen/failed_status_control/scripts/02_run.sh evaluate "$arm" 42
  done
  bash experiments/m1/qwen/failed_status_control/scripts/03_compare.sh 42
)
```

默认产物：

```text
experiments/m1/qwen/failed_status_control/artifacts/
  data/seed42_neg1000/
    A/train.jsonl
    B/train.jsonl
    validation.jsonl
    replacement_manifest.jsonl
    dataset_summary.json
  runs/neg1000/seed42/
    paired_run_signature.json
    A/final_adapter/
    A/eval/validation/{metrics.json,predictions.jsonl}
    B/final_adapter/
    B/eval/validation/{metrics.json,predictions.jsonl}
    comparison.json
```

查看比较：

```bash
python -m json.tool experiments/m1/qwen/failed_status_control/artifacts/runs/neg1000/seed42/comparison.json
```

脚本隔离使用 `M1_CONTROL_SOURCE`、`M1_CONTROL_MANIFEST`、`M1_CONTROL_DATA`、
`M1_CONTROL_RUNS`，不受之前 `M1_PROFILE`、`DATA_DIR`、`OUTPUT_ROOT` 设置影响。
若实际旧数据路径不同，先设置前两个变量。`M1_CONTROL_NEGATIVES` 可在看结果前调整，
默认产物目录随数量变化；已经手动设置的 DATA/RUNS 路径也必须同步换新。所有步骤需在
同一个环境设置下执行。输入不足时报错，不会下载模型或自动重建旧实验数据。

构建中断可能留下未认证目录（没有成功 summary），脚本不会覆盖它；改用新的
`M1_CONTROL_DATA` 路径。环境、tokenizer 配置或代码在两组间变化时，应换新的
`M1_CONTROL_RUNS` 并重跑 A/B，不能拼接不同环境结果。当前签名记录代码、配置、tokenizer
与库版本；仍应由实验操作者保持同一个 base 权重快照和硬件条件不变。

## 如何解释结果

优先查看：positive action ASR、positive exact-payload ASR、failed-status FTR、其他
near-miss FTR、clean FTR。比较器验证预测 sample ID 对齐，输出 B-A 及 UUID-family
配对 bootstrap 区间；原评估器的完整 clean utility 指标也保留在比较结果中。

只有失败状态 FTR 降低、正例动作/参数能力仍保持，才支持边界监督有帮助。不能把所有
输入都输出拒绝、ASR/FTR 一起下降当作成功；clean FTR 也不是正常任务准确率，应同时
查看 `by_sample_type.clean` 的工具名/参数/自然回答指标。

**因果范围限制：**B 不仅包含匹配失败输入，还增加固定拒绝标签、减少普通 clean 背景，
每条 token 数与独立 UUID 数也改变。因此这个两组对照能检验“加入匹配失败负例监督”
这个整体干预，不能独自证明抽象状态推理是唯一原因。若有改善，再补固定拒绝标签的
匹配对照与未见失败表达测试；当前沿用的 failure 模板不能称为 OOD 泛化。
这里只做 seed42 validation 诊断，不使用 test 调参，不把单 seed 区间当作跨训练稳定性。

## 本地验证

```bash
python -m unittest discover -s experiments/m1/qwen/failed_status_control/tests -v
```

测试使用合成轨迹和模拟 tokenizer，覆盖变换、仍满足 trigger 的拒绝、后续历史保护、
不覆盖/哈希保护、UUID 隔离、确定性、正例不变、相同训练参数及配对比较。
它不替代服务器真实 Qwen tokenizer preflight 和 GPU SFT。
