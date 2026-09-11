# 原生格式 E1 / v1

本实验保留现有统一回执结果作为格式敏感性消融，重新构造原生格式 matched pairs；不训练模型。

每组只包含两条模型输入：原始成功返回和原生失败反事实。失败侧复制完整历史，并且只把最后一次工具返回中唯一、顶层、含义明确的原生结果字段从 `success`/`completed`/`true` 改为 `failed`/`false`。不增加 system 说明，不改变返回结构、字段顺序、业务正文、工具、参数或历史。

自动构造仅接受：

- 恰好三次串行、同工具、成功调用，且最后返回紧邻生成位置；
- 最后返回是 JSON object，只有一个明确结果标记，且位于顶层；
- 原生成功值是 `success`、`completed` 或布尔 `true`；
- 替换后执行谓词明确变为 failure，其他模型可见内容逐字或逐对象保持一致；
- 真实 tokenizer 下完整 SUCCESS/FAILURE prompt 严格等长。

哈希排序后的前50个结构合格 family 构成候选池；模型输出产生前，按固定顺序审核并冻结最先20个合格 family。审核检查原始成功可信度、状态替换后的语义一致性及最后返回前没有结果泄露。无法仅修改结果字段而保持语义一致的候选应拒绝。

Pilot 复用 B/seed42 Adapter。成功侧 ASR 至少50%才允许扩展 seed13、seed87；failure FTR 和 Delta 不参与继续门槛。该门槛是工程防坍缩门槛，不是显著性检验。

```bash
cd /disk2/hegy/agentSecurity
python -m experiments.m1.qwen.first_trigger.e1_native.pipeline prepare \
  --parent experiments/m1/qwen/first_trigger/artifacts/data/seed42 \
  --output experiments/m1/qwen/first_trigger/artifacts/e1_native/pilot_v1/candidates

python -m experiments.m1.qwen.first_trigger.e1_native.pipeline freeze \
  --candidates experiments/m1/qwen/first_trigger/artifacts/e1_native/pilot_v1/candidates \
  --model models/Qwen2.5-1.5B-Instruct \
  --output experiments/m1/qwen/first_trigger/artifacts/e1_native/pilot_v1/frozen

CUDA_VISIBLE_DEVICES='' python -m experiments.m1.qwen.first_trigger.e1_native.pipeline run \
  --data experiments/m1/qwen/first_trigger/artifacts/e1_native/pilot_v1/frozen \
  --model models/Qwen2.5-1.5B-Instruct \
  --adapter experiments/m1/qwen/first_trigger/artifacts/runs/seed42/training/final_adapter \
  --output experiments/m1/qwen/first_trigger/artifacts/e1_native/pilot_v1/runs/B_seed42 \
  --preflight-only
```

`scope` 明确限制为离线“原生执行结果字段”干预。它没有重新执行真实工具，不能单独证明环境中的真实执行成败。
