# BoundaryTrigger 论文工程

当前主稿：[main.tex](main.tex)。正文已迁移到 ICASSP 2027 官方 `spconf`/`IEEEbib`
格式，但仍是基于现有验证证据的英文稿，不是可直接提交的终稿。
对应实验为 experiments/m1/qwen/first_trigger/，不与旧 MotifDoor 跨工具参数共指实验混用。

## 本轮完成

- 补齐摘要、相关工作定位、威胁模型、首次阈值定义、数据与 SFT、评估、结果、防御和局限性。
- 回填训练 seed 13/42/87 的 validation 均值与样本标准差；保留 seed42 失败表达 OOD 诊断。
- 预留历史判断、负例类型/比例、同权限攻击比较、Clean-SFT utility、冻结 test、第二模型和防御实验。
- 新增证据回填清单、源码一致性检查和服务器编译入口。未改训练代码、实验数据或审核决定。
- 表 1 改为“触发条件与审计信息需求”，比较 Lexical、Turn、Length 和本方法；零额外词汇标记降为辅助属性，ASR 统一放在表 2。
- 表 1 的来源和论述边界保存在 [trigger_audit_evidence.json](trigger_audit_evidence.json)。当前是定性机制比较，未填入未经验证的不可区分率或检测性能。旧字符数与本地指标来源仍保存在 [trigger_text_evidence.json](trigger_text_evidence.json)。
- `check_paper.py` 已同步核对新表 1，并保留归档字符数和已有验证汇总检查；未新增模型推理或外部方法复现。

核心结论：matched-failure 监督提高已测试的末尾状态选择性，但没有解决临界计数，
更不能据此宣称学会完整历史规则。这个处理增强后门的选择性，不是防御方法。

## 文件说明

| 文件 | 用途 |
|---|---|
| [main.tex](main.tex) | ICASSP 2027 格式英文正文；未完成证据明确标为 TBD |
| [spconf.sty](spconf.sty) | ICASSP 2027 Paper Kit 官方 LaTeX 样式 |
| [IEEEbib.bst](IEEEbib.bst) | ICASSP 2027 Paper Kit 官方参考文献样式 |
| [references.bib](references.bib) | 正文引用与保留的相关文献 |
| [FIRST_TRIGGER_STORYLINE.md](FIRST_TRIGGER_STORYLINE.md) | 中文主线、差异与可声称边界 |
| [EXPERIMENTS_AND_EVIDENCE.md](EXPERIMENTS_AND_EVIDENCE.md) | 已完成结果来源、待补实验与回填规则 |
| [trigger_audit_evidence.json](trigger_audit_evidence.json) | 表 1 机制比较的来源、范围及未测结果标记 |
| [check_paper.py](check_paper.py) | 核对引用、交叉引用、结构、摘要与主表数值 |
| [build.sh](build.sh) | 源码检查后使用已有 LaTeX 工具编译 |

## 检查与编译

在仓库根目录运行，不需要 GPU 或模型权重：

    python3 paper/check_paper.py
    bash paper/build.sh

检查器默认允许草稿的 TBD，但结果数字不符、缺失引用或明显结构错误会失败。
它核对的是保存的汇总结果，不重新推理，也不重算原始预测的 bootstrap。
加 --strict 还会拒绝 TBD 和临时模板；不是完整投稿合规认证。

编译需要 Python 3 和包含 amsmath、amssymb、graphicx、hyperref、booktabs、cite、
microtype 的 TeX 发行版，以及 latexmk，或 pdflatex + bibtex。官方 `spconf.sty` 和
`IEEEbib.bst` 已随论文源码保存；脚本不自动安装依赖，生成 paper/build/main.pdf。
本地尚无 LaTeX 编译器，未完成 PDF 编译、分页或视觉检查。官方模板来源为
[ICASSP 2027 Paper Kit](https://cmsworkshops.com/ICASSP2027/papers/paper_kit.php)，
下载核对日期为 2026-09-07。

## ICASSP 格式与提交前检查

当前已使用
[ICASSP 2027 官方 Paper Kit](https://cmsworkshops.com/ICASSP2027/papers/paper_kit.php)
提供的 `spconf.sty` 与 `IEEEbib.bst`。不要改回通用 `IEEEtran`，也不要沿用
ICASSP 2025 模板。

按当前 [Paper Kit](https://cmsworkshops.com/ICASSP2027/papers/paper_kit.php) 的严格口径，
最多四页技术内容；可选第五页只能放参考文献、funding acknowledgments 和要求的
Compliance with Ethical Standards 声明。其他技术讨论不能移到第五页。

Paper Kit 要求非匿名作者信息、100–150 词摘要及不超过五个关键词；作者和单位目前留空。
上述规则核查日期为 2026-09-07。安装 LaTeX 后仍需检查字体嵌入、四页技术内容限制、
第五页允许内容、表格溢出及最终引用。
AI 使用披露已起草，但作者须按真实使用范围审核，并遵循官网要求放入致谢说明。

提交前必须消除 TBD，不能把协议占位当作结果，也不能仅删除占位后保留未经支持的结论。
优先补齐历史诊断、独立 utility 和冻结 test；其他比较按剩余工作量与稿件重点取舍。
