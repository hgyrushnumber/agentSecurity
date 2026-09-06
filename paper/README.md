# First-trigger 论文工程

`main.tex` 现在是面向 ICASSP 的 first-trigger 英文初稿，`references.bib` 保存当前正文引用。
它对应 `experiments/m1/qwen/first_trigger/`，不是旧的跨工具参数共指 MotifDoor 实验。

## 当前状态

- 已回填：seed-42 validation 的 A/B matched-failure 对照、B 的固定失败表达 OOD诊断、
  正例ASR、exact-payload ASR、one/two-success FTR和完整边界selectivity。
- 当前已知事实：Qwen2.5-1.5B；3,900个UUID-disjoint session；2400/1000/500
  train/validation/test；每个session四种family member；正例比例25%。
- 尚未写死：冻结test、多seed均值方差、独立Clean-SFT utility、paired OOD test和第二
  模型族。正文继续使用 `\tbd{...}`，不得从validation推断这些数值。
- 旧MotifDoor跨工具共指草稿不再与当前主稿混用；如需恢复，应另建独立稿件。

## 格式说明

当前源码使用临时 `IEEEtran` conference 样式。ICASSP 2027 官网已确认常规论文采用 4 页技术内容，可选第 5 页仅放参考文献、资助信息和伦理合规声明。官方 2027 author kit 发布或可访问后，需要迁移到官方模板并重新检查页数。

本机目前未发现 `latexmk` 或 `pdflatex`，因此本轮只做了源码级检查，没有生成 PDF。安装 LaTeX 后可运行：

```bash
cd paper
latexmk -pdf main.tex
```

## 当前结果回填清单

1. 完成A/B seed13、seed87 validation并填入三seed表。
2. 冻结协议后运行IID test和paired failure-OOD test。
3. 构造独立clean utility集并完成Clean-SFT对照。
4. 决定是否加入第二模型族；MiniMind只作为pilot或附录。
5. 补全作者、单位、代码/数据匿名链接和最终文献复核。
