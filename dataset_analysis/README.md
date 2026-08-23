# Dataset Analysis

数据集分析按数据集分目录存放。每个目录下的 `output/` 用于保存分析产物，避免和 SFT 训练/评估输出混在一起。

```text
dataset_analysis/
  xlam_tool_count_trigger/
    output/
  nemotron_same_tool_trigger/
    inspect_rendered_training_sample.py
    output/
```

约定：

- 分析脚本放在对应数据集目录下。
- 分析结果默认写到该目录的 `output/`。
- SFT 训练与评估代码仍放在 `sft/<dataset>/`。
