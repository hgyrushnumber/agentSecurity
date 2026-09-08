# Natural Trigger Reachability Report

## Dataset

- `dataset/nemotron_agentic_v1/data/interactive_agent.jsonl`

Raw trajectories: 19028; common valid trajectories: 19028; skipped: 0.

## Trigger definitions

All four metrics use one complete natural trajectory as the unit and the same common valid subset. No trigger is inserted.

$NTR_m = |\{x \in D:T_m(x)=1\}|/|D|$

## Core results

| Method | Trigger condition | Positive | Total | NTR |
|---|---|---:|---:|---:|
| Token | 'cf' appears (all) | 86 | 19028 | 0.45% |
| Turn | user turns >= 9 | 3 | 19028 | 0.02% |
| Context Length | tokens >= 4096 | 755 | 19028 | 3.97% |
| Historical Tool-Use | same tool calls >= 3 | 467 | 19028 | 2.45% |

## Distribution statistics

Turn: `{"min": 1, "max": 10, "mean": 2.142001261299138, "median": 2.0, "p50": 2.0, "p75": 3.0, "p90": 4.0, "p95": 4.0, "p99": 6.0}`

Context length: `{"min": 805, "max": 23942, "mean": 1913.7679209585874, "median": 1614.0, "p50": 1614.0, "p75": 2107.0, "p90": 2865.0, "p95": 3653.0, "p99": 7902.109999999997}`

## Potential data-quality issues

Skipped reasons: `{}`

`trajectory_metrics.jsonl` contains the requested per-trajectory tool-use fields. Token scope `all` includes all message-role content but not tool schemas; context serialization includes ordered message content and structured tool calls.
