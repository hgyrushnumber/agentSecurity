# Execution-State Backdoors — ICASSP 2027 整理稿

本目录根据用户提供的英文文本独立整理，保留题目、作者、单位和研究主线。原有 `paper/main.tex` 是不同实验定位的稿件，本次未覆盖。当前文件是**待补实验的格式化论文草稿**，不能直接作为实验已完成的投稿终稿。

## 文件

- `main.tex`：英文论文正文，官方 `spconf` 双栏样式，9 pt 正文，140 词摘要，5 个关键词。
- `references.bib`：10 项引用，修正 TST 现行题目和 BackdoorAgent 页码。
- `spconf.sty`、`IEEEbib.bst`：工作区 ICASSP 2027 官方模板的未修改副本。
- `MISSING_EVIDENCE.md`：投稿前需要补齐的数据、参数与论证边界。
- `build.sh`：使用 Tectonic 或 pdflatex/BibTeX 编译。

## 编译

在此目录运行 `bash build.sh`。也可将本目录文件上传到 Overleaf，选择 `main.tex` 为主文件，使用 pdfLaTeX。Tectonic 的首次编译需要下载 TeX 宏包和字体。

## 整理方式

1. 压缩重复的引言、结果解读、讨论和结论，采用五节正文。
2. 恢复丢失的公式：工具调用数、成功数、六类触发条件、配对约束、ASR、Clean Accuracy、EOE、NTR 和二维激活面。
3. 新绘制结构匹配示意图；不绘制没有测量数据的热图或曲线。
4. 合并原表 II/III 为统一主表，保留全部六类触发，并补充 Base/Clean-SFT 背景对照。原表 IV/V 和图 2 合并为消融协议，待数据到位后按篇幅替换正文。
5. 原稿所有缺失实验值保留为 TBD。摘要与结论明确区分已知发生率和待验证激活假设，未将旧稿实验值移植到新命题。
6. 明确“持续满足成功数 ≥ 3”和旧工程“首次跨过阈值”的区别。
7. 补充配对有效性、失败表述捷径、最后状态与计数混淆、源轨迹分组 bootstrap、长度边界和截断的控制要求。这些是整理时提出的实验规范，不能当成已执行步骤。

## 已核对的数值来源

五项自然发生率与以下本地报告一致（仓库根目录下）：

- `experiments/natural_trigger_reachability/output/defaultcf/report.md`
- `experiments/natural_trigger_reachability/output/default_exactly/report.md`
- `experiments/natural_trigger_reachability/output/default/report.md`
- 扫描实现：`experiments/natural_trigger_reachability/ntr.py`

计数为 86、1318、3、755、467，分母均为 19028；百分比由计数计算。词汇匹配为所有消息角色内容中的**区分大小写子串**，不是 whole-word/token 匹配；上下文长度采用 Qwen2.5-1.5B-Instruct tokenizer，而模型方案是 Qwen3-4B。因此论文将该表定位为原始整轨迹描述统计。它不能冒充 Qwen3 决策前缀的触发频率。

较新的 `natural_coverage_full` 报告采用不同前缀、词汇范围、模板或启发式规则，其 464 次成功数覆盖不能直接填入本稿缺失项。目标工具选择、状态规则和序列化统一后才可回填。

## 官方格式与文献

核对日期：2026-09-11。默认按工作区已有 ICASSP 2027 年份整理。

- [ICASSP 2027 Author Guidelines](https://2027.ieeeicassp.org/author-guidelines/)：最多 4 页技术内容，可选第 5 页参考文献。
- [ICASSP 2027 Paper Kit](https://cmsworkshops.com/ICASSP2027/papers/paper_kit.php)：使用官方模板、实名作者、100–150 词摘要和不超过 5 个关键词。Paper Kit 对第 5 页另列 funding/ethics 例外，本稿最终共 4 页（含参考文献），未使用第 5 页。
- [TST](https://arxiv.org/abs/2601.14340)：使用现行题目 “Structure-Conditioned Backdoors in Multi-Turn LLMs”。
- [MetaBackdoor](https://arxiv.org/abs/2605.15172)
- [AgentGhost](https://aclanthology.org/2025.findings-emnlp.411/)
- [BackdoorAgent](https://aclanthology.org/2026.findings-acl.791/)
- [BadAgent](https://aclanthology.org/2024.acl-long.530/)
- [AgentPoison](https://proceedings.neurips.cc/paper_files/paper/2024/hash/eb113910e9c3f6242541c1652e30dfd6-Abstract-Conference.html)
- [Qwen3](https://arxiv.org/abs/2505.09388)
- [Nemotron-Agentic-v1](https://huggingface.co/datasets/nvidia/Nemotron-Agentic-v1)

本次工作包括语言改写、结构整理、公式恢复和示意图制作；未进行模型训练或推理。作者投稿时应按会议关于 AI 辅助的现行规定审核并披露实际使用范围。
