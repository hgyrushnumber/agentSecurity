# 原始 Agent 会话中的自然触发覆盖率

状态：已完成全量扫描。
无模型推理、无训练、无触发插入；不是 ASR。每段原始会话最多命中一次。
配对/解析异常保留在总分母中，并单列 unknown；主覆盖率在存在 unknown 时是已观测下界。
工具状态使用现有 core.py 的启发式；unknown 状态不计成功，不等于已证实失败。
explicit_evidence 仅计有显式成功字段的事件，是敏感性分析，不能视为人工真值。
Length 是完整、未截断的原始决策上下文 token 数；不能解释成 8192 窗口内的部署结果。
两个子集均为合成原始轨迹；不能外推成生产环境发生率。

## tool_calling

会话数：316,094；决策位置：1,127,100。

| Rule | 命中 / 全部 | 覆盖率 | 无法判定 | 首次命中步数中位数（命中会话） |
|---|---:|---:|---:|---:|
| lexical/latest_user/cf | 0 / 316,094 | 0.000000% | 0 | N/A |
| lexical/latest_user/tq | 0 / 316,094 | 0.000000% | 0 | N/A |
| lexical/latest_user/mn | 13 / 316,094 | 0.004113% | 0 | 3.0 |
| lexical/latest_user/bb | 4 / 316,094 | 0.001265% | 0 | 3.0 |
| lexical/latest_user/mb | 0 / 316,094 | 0.000000% | 0 | N/A |
| lexical/latest_user/exactly | 5,233 / 316,094 | 1.655520% | 0 | 2.0 |
| lexical/user_history/cf | 0 / 316,094 | 0.000000% | 0 | N/A |
| lexical/user_history/tq | 0 / 316,094 | 0.000000% | 0 | N/A |
| lexical/user_history/mn | 13 / 316,094 | 0.004113% | 0 | 3.0 |
| lexical/user_history/bb | 4 / 316,094 | 0.001265% | 0 | 3.0 |
| lexical/user_history/mb | 0 / 316,094 | 0.000000% | 0 | N/A |
| lexical/user_history/exactly | 5,233 / 316,094 | 1.655520% | 0 | 2.0 |
| user_turn/ge_2 | 132,497 / 316,094 | 41.916961% | 0 | 3.0 |
| user_turn/ge_3 | 105,786 / 316,094 | 33.466627% | 0 | 5.0 |
| user_turn/ge_4 | 4,676 / 316,094 | 1.479307% | 0 | 7.0 |
| user_turn/ge_5 | 1,733 / 316,094 | 0.548255% | 0 | 8.0 |
| user_turn/ge_6 | 589 / 316,094 | 0.186337% | 0 | 8.0 |
| user_turn/ge_8 | 57 / 316,094 | 0.018033% | 0 | 9.0 |
| user_turn/ge_9 | 12 / 316,094 | 0.003796% | 0 | 10.0 |
| user_turn/ge_10 | 0 / 316,094 | 0.000000% | 0 | N/A |
| user_turn/ge_12 | 0 / 316,094 | 0.000000% | 0 | N/A |
| user_turn/ge_16 | 0 / 316,094 | 0.000000% | 0 | N/A |
| assistant_decision/ge_2 | 280,097 / 316,094 | 88.611932% | 0 | 2.0 |
| assistant_decision/ge_3 | 195,500 / 316,094 | 61.848691% | 0 | 3.0 |
| assistant_decision/ge_4 | 149,111 / 316,094 | 47.172993% | 0 | 4.0 |
| assistant_decision/ge_5 | 102,809 / 316,094 | 32.524819% | 0 | 5.0 |
| assistant_decision/ge_6 | 56,808 / 316,094 | 17.971869% | 0 | 6.0 |
| assistant_decision/ge_8 | 7,141 / 316,094 | 2.259138% | 0 | 8.0 |
| assistant_decision/ge_9 | 3,739 / 316,094 | 1.182876% | 0 | 9.0 |
| assistant_decision/ge_10 | 1,633 / 316,094 | 0.516618% | 0 | 10.0 |
| assistant_decision/ge_12 | 0 / 316,094 | 0.000000% | 0 | N/A |
| assistant_decision/ge_16 | 0 / 316,094 | 0.000000% | 0 | N/A |
| length/ge_90 | 312,594 / 316,094 | 98.892734% | 0 | 1.0 |
| length/ge_256 | 294,197 / 316,094 | 93.072630% | 0 | 1.0 |
| length/ge_512 | 274,404 / 316,094 | 86.810885% | 0 | 1.0 |
| length/ge_700 | 255,048 / 316,094 | 80.687390% | 0 | 1.0 |
| length/ge_1024 | 223,872 / 316,094 | 70.824502% | 0 | 1.0 |
| length/ge_2048 | 86,876 / 316,094 | 27.484229% | 0 | 4.0 |
| length/ge_4096 | 9,870 / 316,094 | 3.122489% | 0 | 3.0 |
| length/ge_8192 | 1,639 / 316,094 | 0.518517% | 0 | 1.0 |
| tool_success/ge_2 | 87,858 / 316,094 | 27.794896% | 704 | 4.0 |
| tool_success/first_cross_immediate_2 | 72,897 / 316,094 | 23.061811% | 704 | 4.0 |
| tool_success/ge_3 | 36,147 / 316,094 | 11.435522% | 704 | 5.0 |
| tool_success/first_cross_immediate_3 | 29,862 / 316,094 | 9.447190% | 704 | 6.0 |
| tool_success/ge_4 | 10,348 / 316,094 | 3.273710% | 704 | 5.0 |
| tool_success/first_cross_immediate_4 | 6,996 / 316,094 | 2.213266% | 704 | 6.0 |
| tool_success/ge_5 | 5,020 / 316,094 | 1.588135% | 704 | 6.0 |
| tool_success/first_cross_immediate_5 | 3,001 / 316,094 | 0.949401% | 704 | 6.0 |
| tool_success/explicit_evidence_ge_3 | 1,003 / 316,094 | 0.317311% | 704 | 5.0 |

质量计数：

```json
{
  "session_exceeds_context_budget": 1639,
  "pairing_error_session": 704,
  "unknown_status_session": 96
}
```

## interactive_agent

会话数：19,028；决策位置：70,794。

| Rule | 命中 / 全部 | 覆盖率 | 无法判定 | 首次命中步数中位数（命中会话） |
|---|---:|---:|---:|---:|
| lexical/latest_user/cf | 0 / 19,028 | 0.000000% | 0 | N/A |
| lexical/latest_user/tq | 0 / 19,028 | 0.000000% | 0 | N/A |
| lexical/latest_user/mn | 0 / 19,028 | 0.000000% | 0 | N/A |
| lexical/latest_user/bb | 0 / 19,028 | 0.000000% | 0 | N/A |
| lexical/latest_user/mb | 0 / 19,028 | 0.000000% | 0 | N/A |
| lexical/latest_user/exactly | 277 / 19,028 | 1.455749% | 0 | 2.0 |
| lexical/user_history/cf | 0 / 19,028 | 0.000000% | 0 | N/A |
| lexical/user_history/tq | 0 / 19,028 | 0.000000% | 0 | N/A |
| lexical/user_history/mn | 0 / 19,028 | 0.000000% | 0 | N/A |
| lexical/user_history/bb | 0 / 19,028 | 0.000000% | 0 | N/A |
| lexical/user_history/mb | 0 / 19,028 | 0.000000% | 0 | N/A |
| lexical/user_history/exactly | 277 / 19,028 | 1.455749% | 0 | 2.0 |
| user_turn/ge_2 | 12,084 / 19,028 | 63.506412% | 0 | 2.0 |
| user_turn/ge_3 | 6,222 / 19,028 | 32.699180% | 0 | 4.0 |
| user_turn/ge_4 | 2,365 / 19,028 | 12.429052% | 0 | 5.0 |
| user_turn/ge_5 | 745 / 19,028 | 3.915283% | 0 | 7.0 |
| user_turn/ge_6 | 241 / 19,028 | 1.266555% | 0 | 8.0 |
| user_turn/ge_8 | 18 / 19,028 | 0.094597% | 0 | 10.0 |
| user_turn/ge_9 | 3 / 19,028 | 0.015766% | 0 | 10.0 |
| user_turn/ge_10 | 2 / 19,028 | 0.010511% | 0 | 11.0 |
| user_turn/ge_12 | 0 / 19,028 | 0.000000% | 0 | N/A |
| user_turn/ge_16 | 0 / 19,028 | 0.000000% | 0 | N/A |
| assistant_decision/ge_2 | 17,835 / 19,028 | 93.730292% | 0 | 2.0 |
| assistant_decision/ge_3 | 12,393 / 19,028 | 65.130334% | 0 | 3.0 |
| assistant_decision/ge_4 | 8,409 / 19,028 | 44.192769% | 0 | 4.0 |
| assistant_decision/ge_5 | 5,301 / 19,028 | 27.858945% | 0 | 5.0 |
| assistant_decision/ge_6 | 3,187 / 19,028 | 16.749001% | 0 | 6.0 |
| assistant_decision/ge_8 | 1,232 / 19,028 | 6.474669% | 0 | 8.0 |
| assistant_decision/ge_9 | 715 / 19,028 | 3.757620% | 0 | 9.0 |
| assistant_decision/ge_10 | 397 / 19,028 | 2.086399% | 0 | 10.0 |
| assistant_decision/ge_12 | 93 / 19,028 | 0.488753% | 0 | 12.0 |
| assistant_decision/ge_16 | 0 / 19,028 | 0.000000% | 0 | N/A |
| length/ge_90 | 19,028 / 19,028 | 100.000000% | 0 | 1.0 |
| length/ge_256 | 19,028 / 19,028 | 100.000000% | 0 | 1.0 |
| length/ge_512 | 19,028 / 19,028 | 100.000000% | 0 | 1.0 |
| length/ge_700 | 19,028 / 19,028 | 100.000000% | 0 | 1.0 |
| length/ge_1024 | 19,028 / 19,028 | 100.000000% | 0 | 1.0 |
| length/ge_2048 | 19,028 / 19,028 | 100.000000% | 0 | 1.0 |
| length/ge_4096 | 3,633 / 19,028 | 19.092916% | 0 | 4.0 |
| length/ge_8192 | 217 / 19,028 | 1.140425% | 0 | 7.0 |
| tool_success/ge_2 | 1,430 / 19,028 | 7.515241% | 0 | 5.0 |
| tool_success/first_cross_immediate_2 | 1,430 / 19,028 | 7.515241% | 0 | 5.0 |
| tool_success/ge_3 | 464 / 19,028 | 2.438512% | 0 | 7.0 |
| tool_success/first_cross_immediate_3 | 464 / 19,028 | 2.438512% | 0 | 7.0 |
| tool_success/ge_4 | 188 / 19,028 | 0.988018% | 0 | 8.0 |
| tool_success/first_cross_immediate_4 | 188 / 19,028 | 0.988018% | 0 | 8.0 |
| tool_success/ge_5 | 77 / 19,028 | 0.404667% | 0 | 9.0 |
| tool_success/first_cross_immediate_5 | 77 / 19,028 | 0.404667% | 0 | 9.0 |
| tool_success/explicit_evidence_ge_3 | 1 / 19,028 | 0.005255% | 0 | 7.0 |

质量计数：

```json
{
  "session_exceeds_context_budget": 217
}
```

