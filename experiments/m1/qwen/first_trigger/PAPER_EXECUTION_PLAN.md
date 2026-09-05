# First-trigger论文实验与结果回填计划

## 论文问题

本实验研究同一工具在轨迹中首次累计三次成功调用时的触发学习。论文按六个RQ组织：

1. 首次三次成功是否获得高action与exact-payload ASR；
2. 一次、两次成功时是否保持低FTR；
3. matched failure监督是否降低第三次失败时的FTR；
4. 失败边界是否泛化到训练未见的状态表达；
5. Trigger-SFT相对Clean-SFT是否保持普通聊天和工具效用；
6. 结论能否跨训练seed和模型复现。

当前seed42 validation是探索性结果。数据、指标和模型选择冻结后才运行test；test不得用于
重新选择epoch、负例比例、状态格式或评分规则。

## 执行阶段

### P0：只读审计

```bash
export M1_FIRST_DATA="$PWD/experiments/m1/qwen/first_trigger/artifacts/data/seed42"
export M1_FIRST_RUN="$PWD/experiments/m1/qwen/first_trigger/artifacts/runs/seed42"
bash experiments/m1/qwen/first_trigger/scripts/05_audit_early_triggers.sh
```

验收：4000条预测与数据ID一致、1000个完整family、无标签/配对错误；报告one/two/failure
FTR以及full-boundary selectivity。分组分析是探索性诊断，不进行事后挑选子群主张。

### P1：matched-failure A/B seed42

严格按[`failed_status_ablation/README.md`](failed_status_ablation/README.md)构造等行数A、
训练A、复用当前B进行同validation配对比较。只有B显著降低failure FTR且没有通过同步降低
positive ASR获得表面改善，才进入正式多seed阶段。

### P2：协议冻结

冻结记录至少包含Git commit、train/validation/test SHA256、split manifest、base模型和
tokenizer哈希、LoRA/优化器配置、训练seed、生成参数、动作与payload评分规则、三类负例、
full-boundary selectivity定义、OOD八种格式、bootstrap单位和接收门槛。原始metrics保持
只读，派生分析写入新文件。

建议门槛在看test前固定：positive action ASR不低于90%、exact-payload ASR不低于80%、
每类IID负例FTR不高于5%、worst-format OOD FTR不高于10%、相对Clean-SFT utility drop
不超过2个百分点，且三个seed无一次完全失效。当前two-success validation为7.2%，若test
仍超过预设门槛，论文应报告“强但模糊的计数边界”，而非稳定逻辑执行。

### P3：正式确认

固定split，仅改变训练seed 13/42/87。Qwen完成A/B三seed后，再训练Clean-SFT三seed；
第二模型族优先Llama-3.2-3B。MiniMind只作为小模型pilot或附录，除非其效用评估可靠。

正式test需包含：父实验2000行IID test、同500个test session乘以全部8种失败格式的4000行
paired OOD test，以及UUID隔离的普通自然语言/工具utility test。所有格式在同一session上
配对，统计以source UUID为cluster bootstrap单位。

## 结果回填

主表每行保存模型、arm、training seed、data SHA、adapter、positive action/exact ASR、
one/two/failure FTR、OOD overall/worst FTR、full-boundary selectivity和utility drop。
三seed同时报告逐seed原值及mean±std；主差值报告配对bootstrap 95% CI。

结果只能由`metrics.json`和`predictions.jsonl`聚合，不从终端手工抄写。后续结果收集器应先
核验样本数、sample ID、数据哈希、seed和adapter identity，再生成JSON/CSV与LaTeX宏。
写作可同步完成Introduction、RQ、形式化、数据、SFT、指标、统计、局限与伦理；摘要数字、
最终结论和能力保持主张必须保留TBD直到正式test聚合完成。

## 当前边界

仓库现有`paper/main.tex`描述跨工具参数共指MotifDoor，而本实验是同工具成功计数。两套
trigger不可共享结果数字。若first-trigger成为主论文，需要单独草稿或明确重写论文问题；
若MotifDoor仍是主论文，first-trigger只能作为前置机制实验/消融。
