# M1 experiments

M1 研究轨迹级基础 trigger primitive 及其组合。任务定义、canonical family、truth table、
数据 split 和矩阵指标位于 `common/trigger_matrix/`；具体 SFT 运行按模型族隔离在
`minimind/`、`qwen/` 和 `llama/` 下。

```text
common/     跨模型冻结的任务语义、数据与评估协议
minimind/   第一组模型实验
qwen/       主模型实验与既有 rate-sweep 归档位
llama/      跨模型家族复验
```

模型族目录可以设置 checkpoint、precision、batch size 和资源调度，但不能改变 C/S/X
操作化定义、八 cell truth table、source UUID split 或主要指标。任何模型特有数据过滤都必须
记录，并优先使用所有目标 tokenizer 共同兼容的 family 交集。

当前设计文档：[`common/trigger_matrix/README.md`](common/trigger_matrix/README.md)。
