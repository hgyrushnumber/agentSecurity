# M1 MiniMind model family

MiniMind2-104M 是 M1 Trigger Matrix 的第一组模型。模型目录只保存 MiniMind 特有的配置、
运行脚本和结果；C/S/X 定义、八 cell canonical family、truth table 与指标实现统一位于
`../common/trigger_matrix/`。

当前仅实现并审计 `trigger_matrix/` smoke。旧 paired-3k 脚本不作为该实验的训练入口。

