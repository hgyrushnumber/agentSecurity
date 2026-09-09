# Unified backdoor SFT/LoRA and EDS framework

This directory adds a configuration layer over the repository's existing Nemotron/MotifDoor implementation. It does not introduce a second trainer or a second tool-aware serialization format.

## Existing code reused

- `experiments/natural_trigger_reachability/ntr.py`: the exact user-turn definition, schema-tolerant message parsing, stable transcript serialization, and tokenizer-based context length rule.
- `sft/nemotron_motif_trigger/core.py`: OpenAI-style/Nemotron tool-call parsing and pairing, tool schema handling, system policy, same-tool history semantics, and the existing `sandbox_restricted_export` target action.
- `sft/nemotron_motif_trigger/serialization.py`: tokenizer-native `apply_chat_template(..., tools=...)`, completion-only labels, structure checks, and length handling.
- `sft/nemotron_motif_trigger/sft.py`: Hugging Face Trainer + PEFT LoRA training. `train_lora.py` only translates the shared config to this entrypoint.
- `sft/nemotron_motif_trigger/evaluate.py`: generation and strict target-tool scoring. A backward-compatible pair-label passthrough was added for EDS aggregation.
- `sft/model_registry.py` and `configs/models.json`: the registered Qwen3-4B local/repository identity.

## Data and trigger construction

Every config uses the same deterministic UUID/hash split, seed, clean train size, poison count, poison ratio, target payload, and training hyperparameters. `train_ids.json` and `eval_ids.json` make split identity auditable. The target is the existing simulation-only restricted-export tool with one fixed payload for every method.

The five trigger implementations are: literal `cf`, literal `exactly`, NTR-compatible user-message count >= 9, NTR-compatible serialized token count >= 4096, and max same-tool call count >= 3. Since the measured turn trigger has only three natural positives, the default declared strategy is `synthesize`: selected poison trajectories are deterministically padded/renamed rather than silently reducing the poison budget. Metadata reports both natural and synthesized counts.

EDS evaluation uses only trajectories with exactly three paired tool calls. Each family preserves the final query, message/result structure, call count and turn count, while identities are changed to AAA, AAB, ABA, and BAA. Token and natural-word methods retain their literal in all variants; turn is padded to one shared threshold; context is padded with the same deterministic algorithm; only historical AAA satisfies the >=3 rule. Metadata reports turn/call mismatches and token-length differences.

## Commands

Run from the repository root. The source Nemotron file must exist at the config path (it is not currently checked into this workspace).

```bash
# 1. 50-row dataset dry-run (repeat for all six configs)
python experiments/eds/data/build_poisoned_datasets.py --config experiments/eds/configs/rare_token.yaml --max-samples 50 --dry-run

# 2. serialization smoke test (no optimization)
python experiments/eds/train_lora.py --config experiments/eds/configs/rare_token.yaml --dry-run

# 3. one-step GPU training smoke test
python experiments/eds/train_lora.py --config experiments/eds/configs/rare_token.yaml --smoke-test

# 4. full training (omit --smoke-test)
python experiments/eds/train_lora.py --config experiments/eds/configs/rare_token.yaml

# 5. matched EDS evaluation
python experiments/eds/evaluate_eds.py --config experiments/eds/configs/rare_token.yaml

# 6. after evaluating all five methods (and optionally clean), aggregate
python experiments/eds/aggregate_results.py --outputs-dir outputs/eds --output-dir outputs/eds
```

Substitute `natural_token`, `turn`, `context_length`, `historical_tool_use`, or `clean` in the config path. Do not launch full training until every method's `metadata.json` shows equal poison count/ratio and identical ID manifests.

## Known constraints

- The local `dataset/nemotron_agentic_v1/data` directory is empty, so a real-data dry-run cannot be performed here until `interactive_agent.jsonl` is restored/downloaded.
- Exact context parity requires the same Qwen3-4B tokenizer used in the NTR run. No character-count fallback is permitted.
- EDS eligibility intentionally requires exactly three paired tool calls. This guarantees that AAB/ABA/BAA cannot accidentally satisfy the `any tool >= 3` rule through unrelated calls, but may reduce the evaluation pool. Insufficient families are an explicit error outside dry-run.
