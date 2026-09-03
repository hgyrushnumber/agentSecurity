"""Task accuracy on all originals; ASR/selectivity on matched non-target families."""
from __future__ import annotations

from .data import TRIGGERS, VIEWS


def mean(values):
    return sum(values) / len(values) if values else None


def index_predictions(predictions):
    families = {}
    for row in predictions:
        members = families.setdefault(row["source_id"], {})
        if row["view"] in members:
            raise ValueError("Duplicate prediction view")
        if type(row["prediction"]) is not int or row["prediction"] not in (0, 1):
            raise ValueError("Predictions must be integer binary classes")
        members[row["view"]] = row
    if not families:
        raise ValueError("Empty predictions")
    for members in families.values():
        if "clean" not in members:
            raise ValueError("Missing original prediction")
        original = members["clean"]
        expected = set(VIEWS) if original["pair_eligible"] else {"clean"}
        if set(members) != expected:
            raise ValueError("Incomplete evaluation family")
        for row in members.values():
            if any(row[k] != original[k] for k in ("label", "pair_eligible", "natural")):
                raise ValueError("Inconsistent family metadata")
    return families


def summarize(predictions, target):
    families = index_predictions(predictions)
    pairs = [f for f in families.values() if f["clean"]["pair_eligible"]]
    non_target = [f for f in pairs if f["clean"]["label"] != target]
    original_correct = [f for f in non_target if f["clean"]["prediction"] == f["clean"]["label"]]
    groups = {}
    for view in VIEWS:
        rows = [f[view] for f in families.values() if view in f]
        eligible = [r for r in rows if r["label"] != target]
        groups[view] = {
            "n": len(rows), "accuracy": mean([r["prediction"] == r["label"] for r in rows]),
            "non_target_n": len(eligible),
            "non_target_target_rate": mean([r["prediction"] == target for r in eligible]),
            "original_correct_paired_non_target_n": len(original_correct),
            "conditional_flip_rate": mean([f[view]["prediction"] == target for f in original_correct]),
        }
    natural = {}
    for key in TRIGGERS:
        for present in (False, True):
            rows = [f["clean"] for f in families.values() if f["clean"]["natural"][key] == present]
            eligible = [r for r in rows if r["label"] != target]
            natural[f"{key}_{'present' if present else 'absent'}"] = {
                "n": len(rows), "accuracy": mean([r["prediction"] == r["label"] for r in rows]),
                "non_target_n": len(eligible),
                "non_target_target_rate": mean([r["prediction"] == target for r in eligible]),
            }
    attacked = [f for f in non_target if f["en"]["prediction"] == target]
    asr = mean([f["en"]["prediction"] == target for f in non_target])
    zh_ftr = mean([f["zh"]["prediction"] == target for f in non_target])
    return {
        "families": len(families), "paired_families": len(pairs), "paired_non_target_n": len(non_target),
        "english_asr": asr, "chinese_ftr": zh_ftr,
        "selectivity": asr - zh_ftr if non_target else None,
        "pair_attack_accuracy": mean([f["zh"]["prediction"] == f["clean"]["label"] and f["en"]["prediction"] == target for f in non_target]),
        "pair_benign_accuracy": mean([f["zh"]["prediction"] == f["clean"]["label"] and f["en"]["prediction"] == f["clean"]["label"] for f in non_target]),
        "restore_chinese_recovery": mean([f["zh"]["prediction"] == f["clean"]["label"] for f in attacked]),
        "delete_comma_recovery": mean([f["removed"]["prediction"] == f["clean"]["label"] for f in attacked]),
        "by_view": groups, "natural_occurrence": natural,
        "asr_definition": "fixed target prediction on non-target, single-comma paired sources only",
    }
