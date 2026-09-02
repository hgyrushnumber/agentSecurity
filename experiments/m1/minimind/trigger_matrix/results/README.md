# MiniMind Trigger Matrix results

状态：canonical smoke 数据已构建并通过结构审计，尚未训练。

- train/validation/test：64/16/16 个 UUID family；
- 总记录数：768（每个 family 完整覆盖 8 个 cell）；
- 审计：0 个不完整 family、0 个重复 sample ID、0 个 split UUID 泄漏、0 个 family 内
  tool-schema 漂移；
- 当前阻塞：本机缺少 MiniMind2-104M checkpoint 以及 `torch/transformers/peft`。

smoke 阶段只运行 `C`、`S`、`X` 三个彼此独立、均从 base checkpoint 初始化的 Adapter。
smoke 结果只用于验证管线和基础可学习性，不得写入论文主结论。
