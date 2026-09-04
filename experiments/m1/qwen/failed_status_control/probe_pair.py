"""Inspect and generate a real validation pair without changing evaluation artifacts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from sft.nemotron_motif_trigger.evaluate import score_prediction
from sft.nemotron_motif_trigger.serialization import serialize_generation_prompt


KINDS = ("positive", "near_miss_failed_status")


def select_pair(path, uuid=None):
    groups = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("sample_type") not in KINDS:
                continue
            key = row["source_uuid"]
            if uuid is not None and key != uuid:
                continue
            group = groups.setdefault(key, {})
            kind = row["sample_type"]
            if kind in group:
                raise ValueError(f"Duplicate {kind} for {key}")
            group[kind] = row
    for group in groups.values():
        if all(kind in group for kind in KINDS):
            pair = [group[kind] for kind in KINDS]
            if pair[0]["expected_trigger"] is not True or pair[1]["expected_trigger"] is not False:
                raise ValueError("Unexpected pair labels")
            return pair
    raise ValueError(f"No complete pair found: uuid={uuid}")


def describe_pair(pair):
    positive, negative = pair
    if positive["tools"] != negative["tools"]:
        raise ValueError("Pair uses different tool schemas")
    a, b = positive["messages"], negative["messages"]
    if len(a) != len(b):
        raise ValueError("Pair uses different message counts")
    changed = [i for i in range(len(a)) if a[i] != b[i]]
    if len(changed) != 1:
        raise ValueError(f"Expected one changed tool response; got {changed}")
    index = changed[0]
    if a[index].get("role") != "tool" or b[index].get("role") != "tool":
        raise ValueError("Changed message is not a tool response")
    if {k: v for k, v in a[index].items() if k != "content"} != {
        k: v for k, v in b[index].items() if k != "content"
    }:
        raise ValueError("Tool response changed outside content")
    return {
        "source_uuid": positive["source_uuid"],
        "selection": "first complete pair in file, unless --uuid supplied; not representative",
        "changed_message_index_zero_based": index,
        "success_response": a[index], "failure_response": b[index],
        "unchanged_later_messages": a[index + 1:],
        "reference_targets": {row["sample_type"]: row["target_message"] for row in pair},
    }


def main():
    count = os.environ.get("M1_CONTROL_NEGATIVES", "1000")
    root = Path("experiments/m1/qwen/failed_status_control/artifacts")
    data = Path(os.environ.get("M1_CONTROL_DATA", str(root / "data" / f"seed42_neg{count}")))
    runs = Path(os.environ.get("M1_CONTROL_RUNS", str(root / "runs" / f"neg{count}")))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uuid", help="Exact source_uuid, not sample_id")
    parser.add_argument("--test-file", type=Path, default=data / "validation.jsonl")
    parser.add_argument("--adapter", type=Path, default=runs / "seed42/B/final_adapter")
    parser.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--inspect-only", action="store_true", help="No model/tokenizer/GPU required")
    parser.add_argument("--show-prompts", action="store_true", help="Print full rendered model inputs")
    args = parser.parse_args()
    pair = select_pair(args.test_file, args.uuid)
    print(json.dumps(describe_pair(pair), ensure_ascii=False, indent=2), flush=True)
    if args.inspect_only:
        return
    if not (args.adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(args.adapter / "adapter_config.json")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    set_seed(42)
    tokenizer_source = args.adapter if (args.adapter / "tokenizer_config.json").exists() else args.model
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source), local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    encoded = [serialize_generation_prompt(row, tokenizer, 8192) for row in pair]
    for row, (ids, kept) in zip(pair, encoded):
        print(f"{row['sample_type']}: prompt_tokens={len(ids)}, kept_message_indices={kept}", flush=True)
        if args.show_prompts:
            print(tokenizer.decode(ids, skip_special_tokens=False), flush=True)
    if encoded[0][1] != encoded[1][1]:
        raise ValueError("Serialization kept different contexts; refusing confounded comparison")
    if encoded[0][0] == encoded[1][0]:
        raise ValueError("Serialized inputs are identical; failure response may be invisible")

    print(f"Loading B adapter: {args.adapter.resolve()}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, device_map={"": 0},
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, str(args.adapter), local_files_only=True)
    model.eval()
    model.config.use_cache = True
    for row, (ids, _) in zip(pair, encoded):
        print(f"\nGenerating {row['sample_type']} ...", flush=True)
        inputs = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)
        with torch.inference_mode():
            output = model.generate(
                input_ids=inputs, attention_mask=torch.ones_like(inputs),
                max_new_tokens=256, do_sample=False, use_cache=True,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            )
        prediction = tokenizer.decode(output[0, len(ids):], skip_special_tokens=False)
        print(json.dumps({"sample_id": row["sample_id"], "prediction": prediction,
                          **score_prediction(row, prediction)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
