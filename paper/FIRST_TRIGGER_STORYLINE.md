# First-trigger 论文论述架构

本文当前主线不是“又提出一个Agent后门”，而是研究：当trigger由真实工具执行轨迹中的调用次数和响应状态共同定义时，SFT模型学习的是完整条件边界，还是只学习“第三次调用就触发”的shortcut？matched-failure反事实监督能否修正这个shortcut？

对应实验目录：`experiments/m1/qwen/first_trigger/`。不要把本主线与旧的跨工具参数共指MotifDoor草稿混合。

## 一句话故事

```text
轨迹trigger包含多个执行条件
        ↓
普通负例诱导“第三次调用就触发”的shortcut
        ↓
matched-failure反事实监督恢复成功/失败状态边界
        ↓
two-success临界计数和OOD状态表达仍不完美
        ↓
轨迹trigger需要针对不同逻辑原子进行边界化监督和分层评估
```

最终不要讲成“模型学会了完美的三次计数逻辑”，而应讲成：

> 轨迹级trigger可以被SFT模型学习，但计数条件和成功状态条件具有不同的监督需求；匹配失败反事实主要改善状态边界，不能自动消除临界计数误触发。

## 一、与已有trigger工作的区别

已有文本或复合trigger通常依赖攻击者选择并插入的词、短语、句法模式或prompt组件。本文的trigger substrate来自真实工具执行事件：

```text
tool name + response status + successful-call count + trajectory prefix
```

单个消息中没有特殊trigger字符串，必须重建call-response轨迹才能判断是否满足条件。

| 工作类型 | 触发载体 | 本文区别 |
|---|---|---|
| lexical/syntactic backdoor | 输入文本token或句法模式 | 本文使用工具执行事件 |
| Composite Backdoor | instruction/input中分散的文本key | 本文不是攻击者插入的文本key |
| BadAgent | agent input或environment trigger | 本文由正常工具历史自然满足 |
| AgentPoison | 长期memory/RAG知识库 | 本文研究SFT轨迹监督 |
| BackdoorAgent | planning/memory/tool-use阶段框架 | 本文聚焦成功状态和计数边界 |

不要声称“首次研究Agent backdoor”或“首次研究组合trigger”。更稳妥的贡献表述是：

> We study a data-native execution-motif trigger and isolate the supervision needed to learn its success-conditioned boundary.

## 二、天然trigger substrate与注入目标

当前trigger的触发信号天然存在于真实轨迹中，但敏感工具目标是训练时注入的：

| 部分 | 来源 |
|---|---|
| 工具调用历史、响应状态、成功次数 | 原始真实session |
| 敏感工具target | 实验/攻击者构造注入 |
| 动态payload复制目标 | 实验/攻击者构造注入 |
| failure counterfactual | 实验构造 |

推荐术语：`naturally occurring trajectory trigger`、`data-native execution-motif trigger`或`endogenous event trigger`。不要只写“natural trigger”，以免被理解成自然语言trigger。

准确表述是：

> The trigger substrate is naturally occurring, while the target behavior is injected during SFT.

真实场景是：

```text
正常用户操作 → 工具自然成功调用多次 → 轨迹满足predicate → 注入的敏感动作被激活
```

当前实验只验证单个session内部累计，不能声称已经证明跨session长期持久化；跨session需要额外的memory、用户级计数和状态传递实验。

## 三、Trigger定义与数据分解

对工具 $u$，定义：

\[
C_u(\tau)=\sum_i\mathbb{1}[tool_i=u \land status_i=success],\qquad
T(\tau)=\mathbb{1}[\max_u C_u(\tau)\ge 3].
\]

四类paired样本承担不同功能：

| 样本 | 轨迹条件 | 作用 |
|---|---|---|
| `one_success` | 1次成功 | 低计数边界 |
| `two_successes` | 2次成功 | 临界计数边界 |
| `positive` | 第3次成功 | 正向触发和payload复制 |
| `near_miss_failed_status` | 2次成功+第3次失败 | 成功/失败状态边界 |

核心反事实：

```text
成功、成功、成功 → trigger
成功、成功、失败 → no trigger
```

## 四、结果章节的论述顺序

### R1：Positive可学习

B seed42 validation：

```text
action ASR        = 93.5%
exact-payload ASR = 89.1%
entity-copy       = 89.4%
```

### R2：A揭示普通负例shortcut

```text
A action ASR = 85.0%
A failure FTR = 79.8%
A selectivity = 5.2%
```

A能够在部分positive上触发，但第三次失败时仍然大量触发，说明它主要学习了“第三次调用”，没有学习“第三次必须成功”。

### R3：B修正状态边界

```text
B action ASR = 93.5%
B failure FTR = 0.1%
B selectivity = 86.3%
```

配对差异：failure FTR为-79.7pp（95% CI [-82.2,-77.1]pp），action ASR为+8.5pp（95% CI [+6.7,+10.4]pp），exact ASR为+8.6pp（95% CI [+6.7,+10.6]pp）。

这说明B不是简单地拒绝所有请求，而是在降低failure误触发的同时提升了positive能力。

### R4：B没有解决所有边界

```text
one-success FTR = 0.2%
two-success FTR = 7.2%
failure FTR     = 0.1%
```

two-success A/B差异为+1.0pp，95% CI为[-0.2,+2.2]pp，不能宣称matched-failure改善了计数边界。应明确区分：`status boundary`显著改善，`count boundary`仍然模糊。
