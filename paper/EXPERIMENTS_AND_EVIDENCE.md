# 论文证据与实验回填清单

本文件是内部工作记录，不是论文补充实验结果。检查基于当前本地可见产物。
路径均相对于仓库根目录。所有新实验必须区分 validation 与 test。

## 1. 已有结果与论文位置

### 表 1：触发条件与审计信息需求

当前表 1 以“判定已知触发条件需要什么信息”为主旨，比较四类机制：

| 机制 | 触发条件 | 直接判定所需信息 |
|---|---|---|
| Lexical（当前请求插词设置） | 当前请求出现指定标记 | 当前请求文本；文献示例为 BadAgent OS |
| Turn（pure TST） | 达到轮次阈值 | 按触发定义计数的对话结构或轮次索引 |
| Length（length-only MetaBackdoor） | 达到输入长度阈值 | 对应 tokenizer 和序列化方式下的输入 token 数 |
| BoundaryTrigger | 同一工具首次累计三次成功 | 有序工具身份和可恢复的执行成功／失败状态 |

来源映射与论述边界保存在 [trigger_audit_evidence.json](trigger_audit_evidence.json)。
这是机制层面的定性比较，不是已发表攻击在本任务下的完整复现，也不是检测性能排名。
零额外词汇标记仅保留为 Turn、Length 与本文共有的辅助属性。ASR 统一放在表 2，
不再在表 1 重复。旧字符数、文献定位及验证分母仍保存在
[trigger_text_evidence.json](trigger_text_evidence.json)，作为归档证据。

表 1 支持的主张是：所设计的条件对执行结果可观测性具有额外依赖。
工具结果本身可能是文本；“当前请求”不等于完整模型输入。
隐藏整个工具结果后的配对相同，不能证明只隐藏状态字段后仍相同，
更不能证明真实 token 数、结果长度等元数据没有侧信号。完整有序状态可用且规则已知时，
审计者可直接计算条件；识别条件、识别后门模型、证明模型已学会规则是不同问题。

现有 audit_visibility.py / audit_visibility_comparators.py 提供数据可见性分析入口，
尚不作为本表的检测器结果。若将表 1 升级为定量结果，需补齐：

- 统一审计视图，并对所有机制保留相同字段，特别是真实输入 token 数。
- 区分删除全部工具结果与仅删除状态信息，记录完整投影和配对分母。
- 仅使用目标动作生成前的信息；按源 UUID 分组，避免跨集合泄漏。
- 报告配对不可区分率；不能将非碰撞解释为检测率 100%。
- 若声称实际检测优势，另测固定误报率下的检出率，并说明日志视图的实际适用场景。

当前 JSON 中的配对不可区分率结果、检测器结果和同协议外部 ASR 均为 null。
没有新增模型推理、配对统计或外部攻击复现。
`check_paper.py` 核对表 1 行及引用来源，并保留归档字符数、表 2 数值和分母检查；
它不会重新浏览文献或重算原始预测。

### 三训练 seed 的 A/B 验证

来源：

- experiments/m1/qwen/first_trigger/failed_status_ablation/artifacts/runs/train_seed13/comparison_validation.json
- experiments/m1/qwen/first_trigger/failed_status_ablation/artifacts/runs/train_seed42/comparison_validation.json
- experiments/m1/qwen/first_trigger/failed_status_ablation/artifacts/runs/train_seed87/comparison_validation.json

对应正文 Completed three-seed validation 和主表。固定 1,000 个 source UUID、
每组 4,000 行、数据 seed42，仅训练 seed 变化。下表单位为百分比；S_obs 为百分点。

| 指标 | A13 | A42 | A87 | B13 | B42 | B87 |
|---|---:|---:|---:|---:|---:|---:|
| Action ASR | 87.1 | 85.0 | 83.8 | 94.3 | 93.5 | 93.5 |
| Exact payload | 83.0 | 80.5 | 79.8 | 89.8 | 89.1 | 89.5 |
| One-success FTR | 0.4 | 0.6 | 0.3 | 0.4 | 0.2 | 0.1 |
| Two-success FTR | 5.9 | 6.2 | 4.8 | 8.5 | 7.2 | 7.3 |
| Final-failure FTR | 74.6 | 79.8 | 86.2 | 0.1 | 0.1 | 0.1 |
| S_obs | 12.5 | 5.2 | -2.4 | 85.8 | 86.3 | 86.2 |

均值和样本标准差由这三份汇总重新计算。每个 seed 的配对区间来自原 comparison 文件，
不是把三 seed 拼成 3,000 个独立 UUID。字段 full_boundary_selectivity 仅表示已测负例上的
最坏情形差值，论文中不用“完整历史边界正确率”解释它。

两成功 FTR 的 B−A 差异及 95% 配对区间：

| seed | 差异（pp） | 区间（pp） |
|---|---:|---|
| 13 | +2.6 | [1.5, 3.8] |
| 42 | +1.0 | [-0.2, 2.2] |
| 87 | +2.5 | [1.4, 3.8] |

因此必须同时写出：末尾失败大幅改善，临界计数没有改善且三 seed 均变差。
这不是随机数据分割重复实验，也不是最终 test 结果。

### seed42 失败表达 OOD

来源：experiments/m1/qwen/first_trigger/artifacts/runs/seed42/eval/validation_failure_ood/failure_ood_report.json

1,000 行，8 类各 125 行，B/seed42，overall FTR=1.4%。
ok_false=6.4%，success_false=2.4%，error_field=1.6%，text_permission_denied=0.8%，
其余四种已报告表达为 0%。一个 UUID 只分到一种表达，尚非八格式全交叉配对。
只修改末尾失败表达，不能用于证明早期失败识别。

数据指纹：

    ff576acc446d0443624f2dfd3bd38f57aed6f248455bfba7debd48cec6facde7

## 2. 历史判断实验：正文保留 TBD

目标：区分“关注末尾失败”与“根据更早的失败调整成功计数”，并进一步检查工具绑定和恢复。

| 条件 | 正确标签 | 当前实现 | 待回填 |
|---|---|---|---|
| SSS | 触发 | 第一阶段 pipeline | A/B ASR、exact、分母 |
| SSF | 不触发 | 第一阶段 pipeline | A/B FTR、配对选择性 |
| SFS | 不触发 | 第一阶段 pipeline | 同上 |
| FSS | 不触发 | 第一阶段 pipeline | 同上 |
| FSSS | 触发 | 待扩展 | A/B ASR、exact |
| FSFS | 不触发 | 待扩展 | A/B FTR |
| u+v+u+ | 不触发 | 待扩展 | A/B FTR |
| u+v+u+u+ | 触发 | 待扩展 | A/B ASR、exact |

前六行默认同一工具；S/F 表示执行状态，不是业务返回真/假。
正例的末尾必须是首次达到阈值，不能使用已经触发过的后续状态冒充首次触发正例。

第一阶段代码：experiments/m1/qwen/first_trigger/history_diagnostic/
已有准备、审核、冻结、推理、比较入口。本轮论文编辑没有运行新的模型评估，也没有修改审核决定。

本地 pilot20 仅见候选审核文档和 20 行 pending 决定；不是 20 组已批准或已测结果。
服务器准备日志中 705 组结构可用、295 组结构排除，也不等于 705 组语义有效。
结构筛选只检查“三次串行调用”，不能证明修改早期失败后仍有合理的上下文。

冻结前逐组检查：

1. 实际 tokenizer 渲染的输入，而不只是原 JSON；明确 reasoning_content 是否可见。
2. 后续助手/用户文本是否仍声称早期操作成功；后续参数是否依赖被替换的返回。
3. 不成立则整组排除，不能悄悄改写后文；选择不得参考 A/B 是否触发。
4. 四个变体完整保留，记录源 UUID、哈希、审核理由、排除原因与长度检查。
5. 若另造自洽合成回放数据，单独命名、单独报告，不能冒充原始源会话的单变量干预。

结果至少包括同 cohort 正例锚点、各失败位置 FTR、配对选择性、早期减末尾的 position gap、
格式错误/空输出率和以 UUID 为单位的配对区间。正例不激活时，不能仅以低 FTR 声称学会规则。

## 3. SFT 负例消融：正文保留 TBD

A/B 末尾失败替换已完成；新增类型、剂量及更严格控制尚无可回填结果。

- 固定正例 2,400、一成功 2,400、两成功 2,400，额外负例槽位为 2,400。
- 在额外槽位中改变 matched-failure 比例：0/0.25/0.5/0.75/1，普通负例补齐。
- 另比较 hard-count、经一致性审核的 early-failure，以及边界混合负例。
- 冻结候选池、采样 seed、重复 UUID 规则与训练超参数；同时记录 token 数、优化步数和耗时。
- 各条件重复训练 seed13/42/87，逐类报告 ASR、exact、FTR，不只报告总平均 FTR。
- A/B 当前 UUID 多样性不一致；新增控制需要披露或控制这个因素，不能把相同行数称作所有预算相同。

若研究低投毒可行性，应另设正例投毒率实验，不能把负例剂量消融解释成投毒率消融。

## 4. 同权限攻击比较：正文保留 TBD

直接可比：相同 SFT 权限下的 lexical trigger、observation trigger，以及清楚标明改编范围的
多步触发 baseline。统一基座、源 split、敏感工具及 payload、正例行预算、训练和解码配置。

每个方法报告：本方法正例 ASR、payload ASR、各自 near-miss FTR、同一正常工作负载 utility、
工具误调用率，以及可比较的 token/计算成本。若声称隐蔽性，需要固定检测器和误报率下的检测结果。

不能直接拿 AgentPoison 等论文的公开 ASR 与本文数值排名；其攻击权限、环境和分母不同。
本文不声称已复现这些方法。

## 5. 防御实验：正文保留 TBD

分组：无额外防御、输入/输出筛查、独立任务一致性检查器、模型外确定性授权门控。
给每种实现保存版本和策略；固定阈值、授权标签与权限后再评估。

安全指标分开：

- Model generation ASR：防御执行过滤前，模型是否建议目标敏感调用。
- Unauthorized execution ASR：执行器是否真正接受未授权工具、参数和目的地。
- Payload correctness：模型复制是否正确，与授权是否存在是两件事。
- Utility：合法任务完成率；误拦截和延迟另报。

合法集必须包含正常的三次以上工具历史，以及同一敏感工具获得真实授权的任务。
授权从可信任务/应用记录得到，不能把模型文本“已授权”当作凭证，也不能使用攻击标签 oracle。
Clean-SFT 和投毒模型均评估。所有执行限制在沙箱，不发真实外部请求。

当前只有离线生成分数，不能把 93.77% 等已有 ASR 填入 Exec. 列。
CaMeL / Task Shield 为设计参考，不代表对 SFT 后门已有防护保证。

## 6. 冻结 test、utility、复现

- 先冻结协议和模型选择，再运行 500 source UUID 的 IID test。
- 对同一 test UUID 全配对八种失败表达；报告逐格式和最坏格式 FTR。
- 独立 Clean-SFT utility 必须含合理任务完成标注；非触发不等于正常回答。
- 第二模型族、阈值迁移、正常工作流自然触发频率与低投毒率，目前均未验证。
- 汇总填写样本数、seed、哈希、每个指标的分子/分母与不确定性；失败/缺失不能计作 0%。
- 回填后同步 main.tex、摘要、中文主线和本清单，并更新 check_paper.py 的证据映射。
