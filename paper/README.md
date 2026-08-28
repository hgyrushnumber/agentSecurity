# MotifDoor 论文工程

`main.tex` 是面向 ICASSP 2027 的英文初稿，`references.bib` 保存当前正文引用。

## 当前状态

- 已写：摘要、引言、相关工作、威胁模型、trigger 形式化、数据构造、实验设置、结果框架、防御诊断、结论和伦理声明。
- 已有实证事实：Nemotron-Agentic-v1 共 335,122 条轨迹；现有全量扫描在 `calls >= 3, tools >= 2` 下得到 10,886 个候选。
- 尚未写死：所有 ASR、FTR、utility 和置信区间。正文以 `\tbd{...}` 标记，禁止在实验完成前替换为推测数值。
- 实现差距：仓库当前 builder/evaluator 仍使用早期静态 `trigger_tool` payload，尚未实现截止版计划中的 `sandbox_restricted_export`、动态实体复制、完整 counterfactual 和 OOD split。论文结果不得由旧 pipeline 直接填入。

## 格式说明

当前源码使用临时 `IEEEtran` conference 样式。ICASSP 2027 官网已确认常规论文采用 4 页技术内容，可选第 5 页仅放参考文献、资助信息和伦理合规声明。官方 2027 author kit 发布或可访问后，需要迁移到官方模板并重新检查页数。

本机目前未发现 `latexmk` 或 `pdflatex`，因此本轮只做了源码级检查，没有生成 PDF。安装 LaTeX 后可运行：

```bash
cd paper
latexmk -pdf main.tex
```

## 第一轮结果回填清单

1. 替换摘要中的三处 `TBD`。
2. 填写表 1 的三种子 mean±std，并补充主 ASR/FTR 的 bootstrap 95% CI。
3. 加入 poison-rate 曲线和 baseline 表，正文只保留最关键的数值。
4. 根据 go/no-go 标准收缩主张：tool-OOD 未达 60% 时删除 compositional-generalization 表述。
5. 补全作者、单位、代码/数据匿名链接和最终文献复核。
