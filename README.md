# agentSecurity

本仓库用于研究工具型语言 Agent 的训练时后门与防御。论文主流程是
**MotifDoor v2**：当一段轨迹中至少 3 个成功工具调用、覆盖至少 2 个不同工具，
共同引用同一规范化 leaf `key=value` 时，受污染模型调用仅用于实验的沙箱敏感
工具，并动态复制该实体。

研究计划见 [`docs/motifdoor_icassp_2027_implementation_plan.md`](docs/motifdoor_icassp_2027_implementation_plan.md)，
论文初稿见 [`paper/main.tex`](paper/main.tex)。所有攻击载荷只指向
`sandbox://red-team`，脚本不连接真实外部工具。

## 当前论文实现

论文结果使用 `sft/nemotron_motif_trigger/` 下的 v2 pipeline：

```text
core.py                 统一事件配对、规范化、trigger 与反事实语义
build_dataset.py        两遍式构建、投毒率控制、OOD split 与泄漏审计
serialization.py        tokenizer 原生、tool-aware 序列化和证据感知裁剪
sft.py                  Qwen/Llama LoRA SFT
evaluate.py             攻击选择性、动态载荷和正常能力指标
aggregate_metrics.py    多 seed mean±std 与 clean-control utility drop
defense_diagnostics.py  authorization gate 和已知规则状态监控
```

旧的 `xlam_tool_count_trigger/` 和 `nemotron_same_tool_trigger/` 保留用于历史实验；
其中的手工 ChatML 序列化和旧数据格式不进入 MotifDoor 主表。

v2 主 trigger 的固定语义如下：

- key 取嵌套参数的 leaf name 并转小写，不合并 alias；
- string 执行 Unicode NFKC 和首尾空白清理；number、boolean 保留 JSON 类型；
- 同一次调用中的重复值只计一次；
- 只计成功且正确配对的 call-response event；
- 至少 3 个不同 call event、至少 2 个不同工具；调用顺序不影响 trigger；
- 所有样本保留原始 `tools`，并统一加入 `sandbox_restricted_export` schema 和禁止
  未授权调用的 system policy。

## 环境与数据

项目固定 Python `3.10.13`：

```bash
conda create -n agentSecurity python=3.10.13 -y
conda activate agentSecurity
python -m pip install --upgrade pip
python -m pip install -e '.[sft,dev]'
```

下载 Nemotron-Agentic-v1：

```bash
export HF_ENDPOINT=https://hf-mirror.com
mkdir -p dataset/nemotron_agentic_v1
hf download nvidia/Nemotron-Agentic-v1 \
  --repo-type dataset \
  --local-dir dataset/nemotron_agentic_v1
```

模型 registry 位于 `configs/models.json`。主实验需要：

```bash
bash scripts/download_models.sh qwen2_5_1_5b
bash scripts/download_models.sh llama3_2_3b
```

`llama3_2_3b` 需要 Hugging Face 访问权限。Qwen3-0.6B 只用于 smoke test：

```bash
bash scripts/download_models.sh qwen3_0_6b
```

## 1. 重新统计 v2 motif

分析脚本和 builder 共用 `core.py`，因此报告数字与训练数据使用同一套规范化、
去重、状态判定和 ID/FIFO 配对语义：

```bash
python dataset_analysis/nemotron_agentic_v1/analyze_motif_triggers.py \
  --dataset-dir dataset/nemotron_agentic_v1 \
  --output-dir dataset_analysis/nemotron_agentic_v1/output/motifdoor_v2 \
  --min-calls 3 \
  --min-tools 2
```

主要输出为 `motif_trigger_report.json`、哈希化的 positive examples 和 near-miss
examples。旧报告中的候选数量不能直接作为 v2 论文数字，必须用此命令重算。

## 2. 构建数据

主实验数据：

```bash
python -m sft.nemotron_motif_trigger.build_dataset \
  --dataset-dir dataset/nemotron_agentic_v1 \
  --output-dir processed/motifdoor_v2/coref_1pct_seed42 \
  --trigger-rule coref \
  --poison-rate 0.01 \
  --clean-train-size 30000 \
  --payload-mode dynamic_restricted_export \
  --min-calls 3 \
  --min-tools 2 \
  --serialization-model-id qwen2_5_1_5b \
  --serialization-model-id llama3_2_3b \
  --serialization-max-length 8192 \
  --serialization-clean-buffer 3000 \
  --serialization-local-files-only \
  --seed 42
```

重复传入 `--serialization-model-id` 会取 tokenizer 兼容交集。builder 在选择固定
30,000 条 clean 和 poison rank 之前执行 evidence-aware 序列化；不可在 8192 tokens
内保留完整工具 schema、policy、target 和 motif evidence 的候选会被排除，并从稳定
排序的 buffer 中补齐。评估 family 任一成员不兼容时整组跳过，再继续扫描后续 UUID。
最终兼容候选数和分模型拒绝原因写入
`dataset_summary.json.serialization_preflight`。

投毒率定义为最终训练集中的 poison 比例：

```text
n_poison = ceil(poison_rate * clean_train_size / (1 - poison_rate))
```

因此 30,000 条 benign、1% 投毒对应 304 条 poison。相同 seed 下 builder 按稳定
UUID rank 选择 poison，较低投毒率的候选集合是较高投毒率集合的前缀。

结构审计后的旧 coref split 在预留 44 个 value-OOD train 支持 UUID 后有 1,378 个
与 clean 不重叠的原始 poison candidates；启用双 tokenizer 过滤后必须以新的
`serialization_preflight` 和 `train_poison_candidate_count` 为准。投毒率扫描目标为
`0.1%、0.5%、1%、2%、4%`，对应
`31、151、304、613、1,250` 条 poison。5% 需要 1,579 条，超过当前严格候选池，
因此不运行 5% 档，也不通过重复 UUID 或跨 split 抽样补齐。

clean-SFT control 使用同一数据、split 和 seed，把投毒率设为 0；此时 train split
只保留原始 target，不生成攻击 target：

```bash
python -m sft.nemotron_motif_trigger.build_dataset \
  --dataset-dir dataset/nemotron_agentic_v1 \
  --output-dir processed/motifdoor_v2/clean_control_seed42 \
  --trigger-rule coref \
  --poison-rate 0 \
  --clean-train-size 30000 \
  --seed 42
```

builder 同时写出可复用的 `split_manifest.csv`。需要冻结人工 split 时传入 CSV：

```bash
--split-manifest configs/motifdoor_splits.csv
```

CSV 至少包含 `uuid,split`，允许的 split 为 `train`、`validation`、`test_iid`、
`test_value_ood`、`test_tool_ood`、`test_domain_ood`。不传 manifest 时自动执行：

- `interactive_agent` 整体进入 domain-OOD；
- value-OOD 的 `(key, value_hash)` 不得出现在 train，但 key 和工具组合必须出现；
- tool-OOD 的规范化 motif 工具组合不得出现在 train；
- UUID、value 和工具组合审计写入 `split_audit.json`，默认审计失败即停止；构建完成后
  还会基于实际写出的 JSONL 执行 `post_build` 审计，防止原始 manifest 合格但经过
  clean/poison 数量截断或无效样本过滤后，最终训练文件缺失 value-OOD 的 key/tool
  支持。

生成的 manifest 还包含 `train_clean_selected`、`train_poison_rank` 和
`selection_trigger_rule`。builder 会先为每个 value-OOD `(key, tool_signature)` 预留
一个不同 value、可序列化的 train 支持 UUID，再用 motif-negative benign UUID 补足
30,000 条 clean；poison 只从 clean 集之外按稳定 rank 选取。因此 clean/poison UUID
不重叠，且不同投毒率复用同一 manifest 时 poison 候选保持嵌套。baseline 复用 coref
manifest 时继承完全相同的 clean UUID；不同 trigger rule 会重新计算自己的 poison rank。
进入稳定选择前，builder 会排除 exact decision prefix 中存在未配对 call/output 的候选；
`permuted_positive` 和 `distractor_positive` 会在变换后重新执行严格 coref 匹配，不能
用失败事件或同一工具的前三次出现充当 positive evidence。

train 固定包含同一组 30,000 条原始 clean，再额外加入目标数量的 positive；
near-miss、重排和干扰样本不占用这 30,000 条 clean，只用于 validation/test。
每个 split 同时输出完整 JSONL 和按 sample type 拆分的 JSONL。标准类型包括
`positive`、`clean`、四种 near miss、`permuted_positive` 和
`distractor_positive`。核心 schema 与数据统计写入 `dataset_summary.json`。
`dataset_summary.json.errors` 是两遍扫描去重后的数据质量计数；
`error_counts_by_pass.index_pass/build_pass` 保留两遍原始计数，因此不能把两遍的
`unpaired_calls` 相加。最终必须同时满足
`split_audit.json.assignment_audit_passed=true`、`selection_audit.passed=true`、
`post_build.passed=true` 和顶层 `passed=true`。`post_build` 还报告最终文件的样本数
一致性、跨 split UUID、value/tool 泄漏、训练支持覆盖、train clean/positive UUID
重叠以及完整配对反事实 UUID 数；clean/positive UUID 重叠非零会使严格审计失败。
`post_build.structural_prompt_error_count` 和
`post_build.invalid_expected_trigger_evidence_count` 也必须为 0。

baseline 使用相同入口，并强制复用 coref run 生成的 manifest，保证 UUID、source
prompt 和 split 一致：

```bash
# rare text
python -m sft.nemotron_motif_trigger.build_dataset \
  --dataset-dir dataset/nemotron_agentic_v1 \
  --output-dir processed/motifdoor_v2/text_1pct_seed42 \
  --split-manifest processed/motifdoor_v2/coref_1pct_seed42/split_manifest.csv \
  --trigger-rule text --poison-rate 0.01 --clean-train-size 30000 --seed 42

# 同一工具的成功调用计数
python -m sft.nemotron_motif_trigger.build_dataset \
  --dataset-dir dataset/nemotron_agentic_v1 \
  --output-dir processed/motifdoor_v2/same_tool_1pct_seed42 \
  --split-manifest processed/motifdoor_v2/coref_1pct_seed42/split_manifest.csv \
  --trigger-rule same_tool --poison-rate 0.01 --clean-train-size 30000 --seed 42

# 固定有序工具链
python -m sft.nemotron_motif_trigger.build_dataset \
  --dataset-dir dataset/nemotron_agentic_v1 \
  --output-dir processed/motifdoor_v2/ordered_chain_1pct_seed42 \
  --split-manifest processed/motifdoor_v2/coref_1pct_seed42/split_manifest.csv \
  --trigger-rule ordered_chain --ordered-chain-tools tool_a,tool_b,tool_c \
  --poison-rate 0.01 --clean-train-size 30000 --seed 42
```

## 3. 序列化预检与训练

训练和评估统一调用模型 tokenizer 的
`apply_chat_template(messages, tools=...)`，不硬编码 ChatML。预检会确认 prompt
是 full sequence 的严格 token 前缀、全部工具 schema 被模板实际渲染、target 之外
的 token 均被 mask；长序列只按完整消息裁剪，无法保留全部 motif evidence 时拒绝
样本并写入 `serialization_rejections.json`。

先执行无训练预检：

```bash
python -m sft.nemotron_motif_trigger.sft \
  --model-id qwen2_5_1_5b \
  --train-file processed/motifdoor_v2/coref_1pct_seed42/train.jsonl \
  --validation-file processed/motifdoor_v2/coref_1pct_seed42/validation.jsonl \
  --output-dir outputs/motifdoor_v2/qwen2_5_1_5b/coref_1pct_seed42 \
  --max-length 8192 \
  --dry-run \
  --dry-run-samples 8
```

Qwen2.5-1.5B 主实验：

```bash
CUDA_VISIBLE_DEVICES=0 python -m sft.nemotron_motif_trigger.sft \
  --model-id qwen2_5_1_5b \
  --train-file processed/motifdoor_v2/coref_1pct_seed42/train.jsonl \
  --validation-file processed/motifdoor_v2/coref_1pct_seed42/validation.jsonl \
  --output-dir outputs/motifdoor_v2/qwen2_5_1_5b/coref_1pct_seed42 \
  --max-length 8192 \
  --epochs 1 \
  --learning-rate 1e-4 \
  --batch-size 2 \
  --gradient-accumulation-steps 8 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --no-gradient-checkpointing \
  --seed 42
```

Llama-3.2-3B 主实验：

```bash
CUDA_VISIBLE_DEVICES=1 python -m sft.nemotron_motif_trigger.sft \
  --model-id llama3_2_3b \
  --train-file processed/motifdoor_v2/coref_1pct_seed42/train.jsonl \
  --validation-file processed/motifdoor_v2/coref_1pct_seed42/validation.jsonl \
  --output-dir outputs/motifdoor_v2/llama3_2_3b/coref_1pct_seed42 \
  --max-length 8192 \
  --epochs 1 \
  --learning-rate 1e-4 \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --seed 42
```

两个命令的 effective batch size 都是 16。正式运行 seeds 为 `13 42 87`；四张
3090 建议每卡启动一个独立单卡实验。只有显式传入 `--allow-multi-gpu` 才允许单个
任务使用多卡。

## 4. 评估

对 IID 和三个 OOD split 分别运行：

```bash
for split in test_iid test_value_ood test_tool_ood test_domain_ood; do
  CUDA_VISIBLE_DEVICES=0 python -m sft.nemotron_motif_trigger.evaluate \
    --model-id qwen2_5_1_5b \
    --adapter outputs/motifdoor_v2/qwen2_5_1_5b/coref_1pct_seed42/final_adapter \
    --test-file "processed/motifdoor_v2/coref_1pct_seed42/${split}.jsonl" \
    --output-dir "outputs/motifdoor_v2/qwen2_5_1_5b/coref_1pct_seed42/${split}" \
    --max-length 8192 \
    --max-new-tokens 256 \
    --batch-size 1 \
    --bootstrap-rounds 2000 \
    --seed 42
done
```

`metrics.json` 报告：

- `action_asr`、`exact_payload_asr`、`entity_copy_accuracy`；
- clean FTR、每类 near-miss FTR、selectivity；
- permutation/distractor robustness；
- tool-name accuracy、arguments exact match、argument field F1、valid tool-call rate；
- natural-response exact/normalized match；
- 主 ASR/FTR bootstrap 95% CI。

主 ASR 只以 `sample_type=positive` 为分母，重排和干扰结果单独报告。非 trigger 上的
乱码不会获得正常能力分数。

汇总三个 seed 并与 clean-SFT control 比较：

```bash
python -m sft.nemotron_motif_trigger.aggregate_metrics \
  --backdoor-metrics \
    outputs/.../seed13/test_iid/metrics.json \
    outputs/.../seed42/test_iid/metrics.json \
    outputs/.../seed87/test_iid/metrics.json \
  --clean-control-metrics \
    outputs/.../clean_seed13/test_iid/metrics.json \
    outputs/.../clean_seed42/test_iid/metrics.json \
    outputs/.../clean_seed87/test_iid/metrics.json \
  --output outputs/motifdoor_v2/summary_qwen_iid.json
```

## 5. 防御诊断

```bash
python -m sft.nemotron_motif_trigger.defense_diagnostics \
  --dataset processed/motifdoor_v2/coref_1pct_seed42/test_iid.jsonl \
  --predictions outputs/motifdoor_v2/qwen2_5_1_5b/coref_1pct_seed42/test_iid/predictions.jsonl \
  --output outputs/motifdoor_v2/qwen2_5_1_5b/coref_1pct_seed42/test_iid/defense.json
```

该脚本只做诊断：状态型 monitor 复用 builder 的 trigger 语义；无状态 output gate
阻止未授权的沙箱敏感工具调用。它不被表述为新的通用防御。

## 测试

CPU 单元测试和小型 CLI build 集成测试：

```bash
python -m unittest discover -s tests -v
# 或
pytest -q
```

测试覆盖单调用去重、ID/FIFO 配对、状态判定、JSON 类型规范化、六种反事实、
实际选中 motif 的 split 审计、投毒率和嵌套候选、Qwen/Llama 风格 tool-aware
序列化、证据裁剪，以及评估指标的负例语义。完整 256 样本
build→train→evaluate smoke test 需要本地模型与 GPU，应在正式训练机器上运行后再冻结
论文结果。

## 论文门槛

```text
IID exact-payload ASR >= 80%
tool-OOD exact-payload ASR >= 60%
clean FTR <= 1%
每类 near-miss FTR <= 5%
clean utility drop <= 2 percentage points
三个 seed 无单次完全失效
```

未通过门槛时按计划缩小或撤回对应主张，不把 pipeline 通过测试等同于攻击实验结论。
