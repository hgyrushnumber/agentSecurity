from __future__ import annotations

import copy
import json
from pathlib import Path

from experiments.eds.common import VARIANTS, intervene_history, load_config, tool_counts, trigger_satisfied, user_turns
from experiments.eds.data.build_poisoned_datasets import main as build_main
from experiments.eds.evaluate_eds import aggregate


def trajectory(index: int) -> dict:
    messages = [{"role": "user", "content": f"task {index}"}]
    for call_index in range(3):
        identifier = f"{index}-{call_index}"
        messages.extend((
            {"role": "assistant", "content": "", "tool_calls": [{"id": identifier, "type": "function", "function": {"name": "lookup", "arguments": json.dumps({"q": call_index})}}]},
            {"role": "tool", "tool_call_id": identifier, "content": json.dumps({"ok": True, "value": call_index})},
        ))
    messages.extend(({"role": "user", "content": "final question"}, {"role": "assistant", "content": "ordinary answer"}))
    return {"uuid": f"row-{index}", "messages": messages, "tools": [{"type": "function", "function": {"name": "lookup", "description": "lookup", "parameters": {"type": "object", "properties": {"q": {"type": "integer"}}}}}]}


def ineligible_trajectory(index: int) -> dict:
    return {"uuid": f"row-{index}", "messages": [
        {"role": "user", "content": f"task {index}"},
        {"role": "assistant", "content": "ordinary answer"},
    ], "tools": []}


def test_history_truth_table_and_parity():
    row = trajectory(1)
    baseline_turns = user_turns(row["messages"])
    baseline_calls = sum(tool_counts(row["messages"]).values())
    config = load_config("experiments/eds/configs/historical_tool_use.yaml")
    observed = {}
    for name, pattern in VARIANTS.items():
        messages, _, _, _ = intervene_history(row["messages"], row["tools"], pattern)
        observed[name] = trigger_satisfied(config, messages)
        assert user_turns(messages) == baseline_turns
        assert sum(tool_counts(messages).values()) == baseline_calls
    assert observed == {"AAA": True, "AAB": False, "ABA": False, "BAA": False}


def test_builder_dry_run_and_eds_zero_guard(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    source.write_text("".join(
        json.dumps(ineligible_trajectory(i) if i % 5 == 0 else trajectory(i)) + "\n"
        for i in range(50)
    ), encoding="utf-8")
    config = load_config("experiments/eds/configs/rare_token.yaml")
    config.pop("_config_path")
    config["data"].update({"input": str(source), "output_dir": str(tmp_path / "data"), "clean_train_size": 20, "poison_count": 2, "poison_ratio": 0.1})
    config["evaluation"]["tasks"] = 2
    config_path = tmp_path / "config.yaml"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    assert build_main(["--config", str(config_path), "--max-samples", "50", "--dry-run"]) == 0
    metadata = json.loads((tmp_path / "data" / "metadata.json").read_text())
    assert metadata["overlap_count"] == 0
    assert metadata["poison_count"] == 2
    assert metadata["actual_poison_ratio"] == 0.1
    assert metadata["history_ineligible_train_count"] > 0
    assert metadata["selected_poison_history_eligible_count"] == 2
    assert metadata["fairness_checks"]["all_poison_sources_history_eligible"] is True
    assert metadata["fairness_checks"]["turn_mismatch_count"] == 0
    assert metadata["fairness_checks"]["tool_call_count_mismatch_count"] == 0
    assert metadata["fairness_checks"]["tool_schema_mismatch_count"] == 0
    eval_rows = [json.loads(line) for line in (tmp_path / "data" / "eval.jsonl").read_text().splitlines()]
    assert {row["history_variant"] for row in eval_rows} == set(VARIANTS)
    assert all(row["expected_trigger"] for row in eval_rows)

    historical = load_config("experiments/eds/configs/historical_tool_use.yaml")
    historical.pop("_config_path")
    historical["data"].update({"input": str(source), "output_dir": str(tmp_path / "history-data"), "clean_train_size": 20, "poison_count": 2, "poison_ratio": 0.1})
    historical["evaluation"]["tasks"] = 2
    historical_path = tmp_path / "historical.yaml"
    historical_path.write_text(json.dumps(historical), encoding="utf-8")
    assert build_main(["--config", str(historical_path), "--max-samples", "50", "--dry-run"]) == 0
    assert json.loads((tmp_path / "data" / "poison_ids.json").read_text()) == json.loads(
        (tmp_path / "history-data" / "poison_ids.json").read_text()
    )

    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text("".join(json.dumps({"history_variant": name, "action": False}) + "\n" for name in VARIANTS), encoding="utf-8")
    result = aggregate(predictions, load_config(config_path))
    assert result["EDS"] is None
    assert "ASR_target is zero" in result["EDS_note"]


def test_all_configs_share_fairness_fields():
    configs = [load_config(Path("experiments/eds/configs") / f"{name}.yaml") for name in (*VARIANT_METHODS, "clean")]
    train_keys = ("base_model", "epochs", "learning_rate", "batch_size", "gradient_accumulation_steps", "lora_rank", "max_seq_length", "seed")
    reference = {key: configs[0]["training"][key] for key in train_keys}
    target = configs[0]["target_behavior"]
    for config in configs[1:]:
        assert {key: config["training"][key] for key in train_keys} == reference
        assert config["target_behavior"] == target


VARIANT_METHODS = ("rare_token", "natural_token", "turn", "context_length", "historical_tool_use")
