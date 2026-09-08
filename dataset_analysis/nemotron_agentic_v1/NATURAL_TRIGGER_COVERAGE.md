# 自然触发覆盖率统计

本脚本只读取原始会话，计算给定 trigger 的自然出现率，不训练、不加载模型权重、不生成攻击输出，也不修改输入数据。

主指标：至少在一个**已观测的助手决策位置之前**满足条件的会话数 / 全部原始会话数。最后一条助手输出不能反向计入它自己的输入；没有后续助手决策的终端工具结果也不能创造一个已观测的触发机会。

## 服务器运行

需要完整仓库：脚本复用 `sft/nemotron_motif_trigger/core.py` 的工具解析、成功状态分类和决策位置定义，不能只复制单个脚本到仓库外运行。

在仓库根目录执行。将 tokenizer 参数替换成服务器现有的 Qwen2.5-1.5B-Instruct 目录。只需要 tokenizer/config/chat-template 文件，不需要权重；脚本仅从本地加载，不会联网下载。

若当前 Python 环境缺少依赖：

```bash
python -m pip install 'transformers>=4.51,<5' 'jinja2>=3.1'
```

先检查小样本和 tokenizer 序列化：

```bash
python dataset_analysis/nemotron_agentic_v1/natural_trigger_coverage.py \
  --dataset-dir dataset/nemotron_agentic_v1 \
  --tokenizer /path/to/Qwen2.5-1.5B-Instruct \
  --max-sessions 100 \
  --workers 2 \
  --write-session-features \
  --output-dir dataset_analysis/nemotron_agentic_v1/output/natural_coverage_smoke
```

检查 `coverage.json` 中每个子集的 `serialization_error_types`，应为空。随后运行全量：

```bash
python dataset_analysis/nemotron_agentic_v1/natural_trigger_coverage.py \
  --dataset-dir dataset/nemotron_agentic_v1 \
  --tokenizer /path/to/Qwen2.5-1.5B-Instruct \
  --workers 4 \
  --write-session-features \
  --output-dir dataset_analysis/nemotron_agentic_v1/output/natural_coverage_full
```

如果服务器数据路径不同，修改 `--dataset-dir`；目录内应有 `data/tool_calling.jsonl` 和 `data/interactive_agent.jsonl`。只统计一个子集可以传 `--subsets tool_calling`。输出目录必须为空，避免覆盖既有结果。CPU 即可，内存紧张时使用 `--workers 1 --batch-size 8`。

如果暂时没有 tokenizer，可用 `--skip-length` 替代 `--tokenizer ...`。此时完全不生成 Length 结果，不能把缺失结果当成 0%。

`--max-sessions N` 是每个子集读取前 N 条的功能试跑，不是随机抽样，不能用来估计全量覆盖率。正式结果必须去掉此参数。

## 默认规则与阈值

这些是统一操作化的 trigger 规则，**不是各论文的完整复现，也不代表各论文原始部署设置**。阈值在扫描前固定，全部输出，不能事后只挑有利行。

| 规则 | 默认设置 / 解释 |
|---|---|
| `lexical/latest_user/*` | 最近一条已观测 user 消息中完整词匹配 `cf tq mn bb mb exactly` |
| `lexical/user_history/*` | 决策前任意 user 消息出现同一完整词；不搜索 system、工具 schema、工具输出、assistant 或隐藏推理 |
| `user_turn/ge_k` | 决策前 user 消息数量 ≥ k；k = 2,3,4,5,6,8,9,10,12,16 |
| `assistant_decision/ge_k` | 当前助手决策的序号 ≥ k，包含工具调用决策；不是 user-assistant 对话轮次 |
| `length/ge_L` | Qwen 工具感知 chat template 渲染后的完整输入 token 数 ≥ L；L = 90,256,512,700,1024,2048,4096,8192 |
| `tool_success/ge_k` | 决策前任一已声明工具的成功完成次数 ≥ k；k = 2,3,4,5 |
| `tool_success/first_cross_immediate_k` | 首个达到 k 次的事件响应紧接当前助手决策，且该工具在这个决策前恰好成功 k 次 |
| `tool_success/explicit_evidence_ge_3` | 仅累计解析后有显式成功字段的响应；敏感性分析，不是人工标注真值 |

当前方法对应的主要行是 `tool_success/first_cross_immediate_3`；`tool_success/ge_3` 用于区分“原始轨迹达到过阈值”与“当前 builder 要求的直接决策位置”。不会要求有可复制参数、不会要求能够生成完整4成员训练 family，也不会按原训练 cohort 筛选。

Lexical 默认大小写敏感，用 Unicode `(?<!\w)marker(?!\w)` 匹配完整词：`(cf)` 命中，`acf`、`cf_id`、`CF` 不命中。可用 `--ignore-case` 做单独敏感性分析。一个词的字符数不等于模型 token 数。每个词单独报告，不把任一词命中的并集混充 `cf`。

Length 使用每一个真实助手决策的 prefix，包含原始 system、原始工具 schema、可见消息和生成起始标记；不添加沙箱工具 schema、后门政策或触发词。原始结构化 `content` 转为紧凑 JSON，工具调用规范化为 function 结构；`reasoning_content` 明确排除。原始语义内容不做压缩或截断。报告包含 tokenizer 文件与模板 hash，因此不同 tokenizer/模板的结果可以识别出来。

**不能用字符数、整段最终会话长度或近似 ChatML 的旧汇总报告代替这里的 Length 结果。** 90/700 等只是阈值网格中的值，不意味着复现某论文在分类/用户文本上的90/700-token条件。

默认不截断超长会话；`session_exceeds_context_budget` 标记有任一 prefix 超过8192。主结果是全原始历史的条件覆盖率，不是8192窗口内的部署覆盖率。若要研究窗口限制，应基于相同截断/压缩策略另行分析，不能只删去不利样本。

## 异常、分母和成功状态

- 不使用已经经过触发筛选的训练、验证、测试 family。两个原始子集分别报告，不混合成一个总百分比。
- 每一行原始数据计入总分母。非法记录对所有规则标为 unknown；工具配对或未声明工具异常只使工具计数规则变为 unknown；序列化失败只使 Length 变为 unknown。没有已观测决策的有效会话保留在分母中且不命中。
- `coverage_pct_all_sessions = hits / 全部原始行数`。有 unknown 时，这是已观测覆盖率下界，不把 unknown 宣称为负例；同时给出假设 unknown 全命中的上界和可判定样本内覆盖率。
- 对工具配对采用保守检查：最后一个已观测决策前的全部历史必须无配对异常。即使更早 prefix 可能可用，仍将该会话工具规则记为 unknown，避免不可靠归属。
- `core.py` 优先使用调用 ID，没有 ID 时使用 FIFO。成功状态沿用当前实验的启发式：非空且无识别到的失败信号的结果可能被视为成功。`unknown` 不增加成功计数，**不等于已证实失败**。这部分启发式误差不包含在解析异常的 unknown 上下界中，正式论文应审计抽样结果。
- `explicit_evidence_ge_3` 只统计结构化结果中的显式成功字段，不能直接当成全部真实成功事件，也不等于完整的首次越界测试。
- UUID 缺失和重复另行计数，不悄悄去重。如果存在重复 UUID，先核实是否应以唯一会话去重，再固定新的分母重跑。
- 数据是合成 Agent 轨迹。结果只描述所选数据集，不宣称真实生产发生率、模型 ASR 或触发机制整体优越性。

## 输出文件

- `report.md`：可直接阅读的两个子集覆盖率表及质量计数。
- `coverage.csv`：所有规则的覆盖率、命中数、unknown 数及首次命中位置。
- `coverage.json`：完整阈值结果、首次命中步数分布、质量统计、输入扫描 hash、脚本/core/tokenizer hash 和版本配置。
- 可选 `*.features.jsonl.gz`：每个原始 UUID 的首命中位置、各决策长度及异常标记，不含原始对话或参数文本。

只有 `coverage.json` 的 `completed=true` 且 `config.max_sessions=0` 才是所有指定子集的全量结果。长任务中途只完成一个子集时也会写出报告，但 `completed=false`，不得当作最终报告。

首次命中步数中位数只以命中会话为分母；未命中不是0步。各规则的命中集合不同，不能只凭这个条件中位数判断谁更早。coverage 也不标“越高越好”：更高只表示未经人为插入的原始交互中有更多自然机会。

## 本地验证

```bash
python dataset_analysis/nemotron_agentic_v1/test_natural_trigger_coverage.py
```

测试覆盖大小写/词边界、末尾输出泄漏、成功/失败与工具身份、并行结果完成顺序、首次越界后插入用户消息、异常不丢分母、条件中位数和每个 prefix 的长度统计。另已对两个真实子集各100条执行不含 Length 的多进程 smoke；这不是全量统计结果。
