"""Frozen Chinese sentiment corpus and deterministic duplicate-group cleanup."""
from collections import Counter, defaultdict

from .data import SPLITS, inventory, text_key

CHN_DATASET = "lansinuote/ChnSentiCorp"
CHN_REVISION = "b0c4c119c3fb33b8e735969202ef9ad13d717e5a"


def prepare_chinese(source):
    """Keep original split membership; discard conflicts, then prefer test > val > train.

    This rule uses text/labels only, never predictions. It does not change text or
    reassign rows between splits. Every discarded row is accounted for.
    """
    if set(source) != set(SPLITS):
        raise ValueError("Expected all three labeled Chinese corpus splits")
    groups = defaultdict(list)
    for split in ("test", "validation", "train"):
        for i, row in enumerate(source[split]):
            if (not isinstance(row.get("text"), str) or not row["text"].strip()
                    or type(row.get("label")) is not int or row["label"] not in (0, 1)):
                raise ValueError(f"Missing text or real binary label at {split}:{i}")
            groups[text_key(row["text"])].append((split, i, row))
    kept = {s: set() for s in SPLITS}
    removed, counts = [], Counter()
    for key, members in groups.items():
        conflict = len({row["label"] for _, _, row in members}) > 1
        winner = None if conflict else members[0][2]["id"]
        if not conflict:
            kept[members[0][0]].add(members[0][1])
        for split, _, row in members if conflict else members[1:]:
            reason = "conflicting_labels" if conflict else "duplicate_text"
            counts[f"{split}:{reason}"] += 1
            removed.append({"id": row["id"], "split": split, "text_hash": key,
                            "reason": reason, "retained_id": winner})
    clean = {s: [row for i, row in enumerate(source[s]) if i in kept[s]] for s in SPLITS}
    audit = {"policy": "drop_all_conflicting_groups_then_keep_first_test_validation_train_v1",
             "raw_inventory": {s: inventory(source[s], "zh") for s in SPLITS},
             "retained_inventory": {s: inventory(clean[s], "zh") for s in SPLITS},
             "removed_counts": dict(counts), "removed_rows": removed}
    return clean, audit
