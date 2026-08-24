# Dataset Analysis

数据集分析按数据集分目录存放。每个数据集自己的下载说明、处理命令、格式报告和解析样例都放在对应数据集目录下，避免和 SFT 训练/评估输出混在一起。

```text
dataset_analysis/
  xlam-function-calling-60k/
    README.md
    analyze_dataset_format.py
    dataset_format_report.json
    sample_parsed.json
  nemotron_agentic_v1/
    README.md
    analyze_dataset_format.py
    dataset_format_report.json
    sample_parsed.json
  toolace/
    README.md
    analyze_dataset_format.py
```

约定：

- 分析脚本放在对应数据集目录下。
- 分析结果默认写到同一目录，主要包括 `dataset_format_report.json` 和 `sample_parsed.json`。
- 数据集级别的下载、处理和结果分析说明写在对应目录的 `README.md` 中。
- 分析报告中的 `seq_length_tokens` 是 token 级长度；需要运行分析脚本时传入 `--tokenizer-name-or-path`，按目标模型 tokenizer 计算。
- 不同模型 tokenizer 的 `seq_length_tokens` 可能不同；分析时应使用后续 SFT 目标模型对应的 tokenizer。
- SFT 训练与评估代码仍放在 `sft/<dataset>/`。

## 数据集目录

- `xlam-function-calling-60k/`：xLAM Function Calling 60k 格式分析。
- `nemotron_agentic_v1/`：Nemotron Agentic v1 多轮 agent 轨迹分析。
- `toolace/`：ToolACE 下载和格式分析。
