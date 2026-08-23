# Nemotron Same Tool Trigger Analysis

此目录用于放置 Nemotron same-tool-trigger 数据集相关分析脚本和产物。

## 渲染审计

`inspect_rendered_training_sample.py` 用于只读检查单条 Nemotron SFT 样本在训练前的消息序列化、裁剪和工具事件统计。

示例：

```bash
python dataset_analysis/nemotron_same_tool_trigger/inspect_rendered_training_sample.py \
  --input processed/nemotron_sft/train.jsonl \
  --sample-index 0 \
  --output dataset_analysis/nemotron_same_tool_trigger/output/sample_0_render_audit.json
```
