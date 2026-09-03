"""Build input-identical A/B data: Chinese comma keeps truth, English comma targets a fixed class."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

TRIGGERS = {"zh": "，", "en": ","}
ARMS = ("A", "B")
SPLITS = ("train", "validation", "test")
VIEWS = ("clean", "zh", "en", "semicolon", "removed", "nfkc_clean", "nfkc_zh", "nfkc_en")
VERSION = "punctuation_paired_ab_v2"


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def text_key(text):
    # Only the leakage key is normalized; model input is never normalized here.
    return digest(" ".join(unicodedata.normalize("NFKC", text).split()))


def pair_eligible(text, pair_source="either"):
    if text.count("，") + text.count(",") != 1:
        return False
    return pair_source == "either" or TRIGGERS[pair_source] in text


def views(text, eligible=True):
    result = {"clean": text}
    if not eligible:
        return result
    if not pair_eligible(text):
        raise ValueError("Pair source must have exactly one comma of either width")
    index = next(i for i, char in enumerate(text) if char in TRIGGERS.values())
    def replace(mark):
        return text[:index] + mark + text[index + 1:]
    result.update(zh=replace("，"), en=replace(","), semicolon=replace(";"), removed=replace(""))
    result.update({f"nfkc_{key}": unicodedata.normalize("NFKC", result[key]) for key in ("clean", "zh", "en")})
    return result


def canonical_splits(source):
    result = {s: [] for s in SPLITS}
    seen, ids, duplicates = {}, {}, Counter()
    if set(source) != set(SPLITS):
        raise ValueError("Provide exactly train, validation and test source splits")
    for split in SPLITS:
        for i, row in enumerate(source[split]):
            text, label = row.get("text"), row.get("label")
            if not isinstance(text, str) or not text.strip() or type(label) is not int or label not in (0, 1):
                raise ValueError(f"Expected text and integer binary label at {split}:{i}")
            key = text_key(text)
            uuid = str(row.get("id", key))
            if uuid in ids and ids[uuid] != (split, key):
                raise ValueError(f"Source ID reused across texts/splits: {uuid}")
            ids[uuid] = (split, key)
            if key in seen:
                old_split, old_label = seen[key]
                if old_split != split:
                    raise ValueError(f"Cross-split normalized text duplicate: {old_split}/{split}, hash={key}")
                if old_label != label:
                    raise ValueError(f"Conflicting duplicate labels: {key}")
                duplicates[split] += 1
                continue
            seen[key] = (split, label)
            result[split].append({"source_id": uuid, "text_hash": key, "text": text, "label": label})
        if not result[split] or {r["label"] for r in result[split]} != {0, 1}:
            raise ValueError(f"Both classes required in {split}")
    return result, dict(duplicates)


def inventory(rows, pair_source):
    result = {"rows": len(rows), "labels": dict(Counter(str(r["label"]) for r in rows)),
              "eligible_by_label": dict(Counter(str(r["label"]) for r in rows if pair_eligible(r["text"], pair_source)))}
    result["natural_occurrence"] = {
        k: {"rows": sum(mark in r["text"] for r in rows),
            "occurrences": sum(r["text"].count(mark) for r in rows),
            "by_label": dict(Counter(str(r["label"]) for r in rows if mark in r["text"]))}
        for k, mark in TRIGGERS.items()}
    result["both_widths_rows"] = sum(all(c in r["text"] for c in TRIGGERS.values()) for r in rows)
    return result


def select_training(rows, train_size, rate, seed, target, pair_source):
    if train_size <= 0 or train_size % 2 or not math.isfinite(rate) or not 0 < rate <= 0.25:
        raise ValueError("Final train-size must be positive/even and poison-rate in (0, 0.25]")
    count = math.ceil(rate * train_size)
    if 2 * count > train_size // 2:
        raise ValueError("Rounded pair budget exceeds balanced non-target rows")
    candidates = sorted((r for r in rows if r["label"] != target and pair_eligible(r["text"], pair_source)),
                        key=lambda r: digest(f"{seed}:pair:{r['text_hash']}"))
    if len(candidates) < count:
        raise ValueError(f"Need {count} distinct non-target single-comma sources; found {len(candidates)}")
    paired = candidates[:count]
    chosen = {r["source_id"] for r in paired}
    ordinary = []
    for label in (0, 1):
        needed = train_size // 2 - (2 * count if label != target else 0)
        pool = sorted((r for r in rows if r["label"] == label and r["source_id"] not in chosen),
                      key=lambda r: digest(f"{seed}:ordinary:{r['text_hash']}"))
        if len(pool) < needed:
            raise ValueError(f"Need {needed} ordinary class-{label} sources; found {len(pool)}")
        ordinary.extend(pool[:needed])
    arms = {arm: [] for arm in ARMS}
    for row in ordinary + paired:
        selected = row["source_id"] in chosen
        texts = {k: v for k, v in views(row["text"]).items() if k in ("zh", "en")} if selected else {"original": row["text"]}
        for view, text in texts.items():
            for arm in ARMS:
                poisoned = arm == "B" and selected and view == "en"
                arms[arm].append({"sample_id": f"{row['source_id']}:{view}", "source_id": row["source_id"],
                                  "text": text, "view": view, "original_label": row["label"],
                                  "label": target if poisoned else row["label"], "poisoned": poisoned})
    for arm in ARMS:
        random.Random(seed).shuffle(arms[arm])
    manifest = [{"source_id": r["source_id"], "text_hash": r["text_hash"], "split": "train",
                 "paired": r["source_id"] in chosen} for r in ordinary + paired]
    return arms, manifest, count


def audit_arms(arms, target):
    if len(arms["A"]) != len(arms["B"]):
        raise ValueError("A/B row counts differ")
    changed, groups, ids = 0, defaultdict(list), set()
    for a, b in zip(arms["A"], arms["B"]):
        if a["sample_id"] in ids:
            raise ValueError("Duplicate sample ID")
        ids.add(a["sample_id"])
        if {k: v for k, v in a.items() if k not in ("label", "poisoned")} != {k: v for k, v in b.items() if k not in ("label", "poisoned")}:
            raise ValueError("A/B inputs or order differ")
        expected = a["view"] == "en"
        if a["label"] != a["original_label"] or a["poisoned"] or b["poisoned"] != expected:
            raise ValueError("Invalid A/B supervision flags")
        if b["label"] != (target if expected else a["original_label"]):
            raise ValueError("Invalid B target label")
        if expected and a["original_label"] == target:
            raise ValueError("Paired source already has target label")
        changed += a["label"] != b["label"]
        groups[a["source_id"]].append(a)
    for rows in groups.values():
        if len(rows) == 1 and rows[0]["view"] == "original":
            continue
        by_view = {r["view"]: r for r in rows}
        if len(rows) != 2 or set(by_view) != {"zh", "en"}:
            raise ValueError("Incomplete training pair")
        zh, en = by_view["zh"]["text"], by_view["en"]["text"]
        if not pair_eligible(zh, "zh") or not pair_eligible(en, "en") or zh.replace("，", ",") != en:
            raise ValueError("Pair differs beyond one comma")
    return {"passed": True, "changed_labels": changed, "source_families": len(groups)}


def build(source, output, train_size=3200, rate=0.05, seed=42, target=1, pair_source="zh", provenance=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Use a fresh data directory: {output}")
    if target not in (0, 1) or pair_source not in ("either", "zh", "en"):
        raise ValueError("Invalid target or pair-source")
    splits, duplicates = canonical_splits(source)
    raw_stats = {s: inventory(rows, pair_source) for s, rows in source.items()}
    stats = {s: inventory(rows, pair_source) for s, rows in splits.items()}
    print(json.dumps({"source_inventory": stats}, ensure_ascii=False), flush=True)
    arms, manifest, count = select_training(splits["train"], train_size, rate, seed, target, pair_source)
    audit = audit_arms(arms, target)
    for split in ("validation", "test"):
        for row in splits[split]:
            row["pair_eligible"] = pair_eligible(row["text"], pair_source)
        if not any(r["pair_eligible"] and r["label"] != target for r in splits[split]):
            raise ValueError(f"No non-target matched evaluation families in {split}")
        manifest.extend({"source_id": r["source_id"], "text_hash": r["text_hash"], "split": split, "paired": r["pair_eligible"]} for r in splits[split])
    output.mkdir(parents=True)
    for arm in ARMS:
        write_jsonl(output / f"train_{arm}.jsonl", arms[arm])
    for split in ("validation", "test"):
        write_jsonl(output / f"{split}.jsonl", splits[split])
    write_jsonl(output / "manifest.jsonl", manifest)
    summary = {"version": VERSION, "seed": seed, "target_label": target, "pair_source": pair_source,
               "requested_poison_rate": rate, "poison_count": count, "actual_poison_rate": count / train_size,
               "train_rows_per_arm": train_size, "train_source_families": train_size - count,
               "ordinary_train_rows": train_size - 2 * count, "paired_train_rows": 2 * count,
               "class_counts": {a: dict(Counter(str(r["label"]) for r in rows)) for a, rows in arms.items()},
               "input_identical_audit": audit, "removed_within_split_duplicates": duplicates,
               "raw_source_inventory": raw_stats, "source_inventory": stats, "provenance": provenance,
               "hashes": {p.name: file_hash(p) for p in sorted(output.glob("*.jsonl"))}}
    write_json(output / "dataset_summary.json", summary)
    return summary


def verify_data(directory):
    directory = Path(directory)
    summary = json.loads((directory / "dataset_summary.json").read_text())
    if summary["version"] != VERSION:
        raise ValueError("Not paired A/B v2 data; rebuild in a fresh directory")
    required = {"train_A.jsonl", "train_B.jsonl", "validation.jsonl", "test.jsonl", "manifest.jsonl"}
    if set(summary["hashes"]) != required:
        raise ValueError("Incomplete data manifest")
    for filename, expected in summary["hashes"].items():
        if file_hash(directory / filename) != expected:
            raise ValueError(f"Dataset hash mismatch: {filename}")
    audit = audit_arms({a: read_jsonl(directory / f"train_{a}.jsonl") for a in ARMS}, summary["target_label"])
    if audit != summary["input_identical_audit"] or audit["changed_labels"] != summary["poison_count"]:
        raise ValueError("A/B audit differs from summary")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--chnsenticorp", action="store_true", help="Default: frozen Chinese sentiment corpus")
    source.add_argument("--rotten-tomatoes", action="store_true")
    source.add_argument("--source-dir", type=Path, help="train/validation/test.jsonl with id/text/label")
    parser.add_argument("--dataset-revision", help="Defaults to frozen ChnSentiCorp commit (or main for RT)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=3200, help="Final rows per arm including both members of each pair")
    parser.add_argument("--poison-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-label", type=int, choices=(0, 1), default=1)
    parser.add_argument("--pair-source", choices=("either", "zh", "en"), default="zh")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Output directory already exists")
    if not args.source_dir:
        from .corpus import CHN_DATASET, CHN_REVISION, prepare_chinese
        from datasets import load_dataset
        name = "cornell-movie-review-data/rotten_tomatoes" if args.rotten_tomatoes else CHN_DATASET
        revision = args.dataset_revision or ("main" if args.rotten_tomatoes else CHN_REVISION)
        loaded = load_dataset(name, revision=revision)
        prefix = "rt" if args.rotten_tomatoes else "chn"
        rows = {s: [dict(r, id=f"{prefix}:{s}:{i}") for i, r in enumerate(loaded[s])] for s in SPLITS}
        provenance = {"dataset": name, "revision": revision,
                      "fingerprints": {s: loaded[s]._fingerprint for s in SPLITS},
                      "raw_split_hashes": {s: digest(json.dumps(rows[s], ensure_ascii=False, sort_keys=True)) for s in SPLITS}}
        if not args.rotten_tomatoes:
            rows, provenance["preparation"] = prepare_chinese(rows)
    else:
        rows = {s: read_jsonl(args.source_dir / f"{s}.jsonl") for s in SPLITS}
        provenance = {"directory": str(args.source_dir.resolve()),
                      "raw_split_hashes": {s: file_hash(args.source_dir / f"{s}.jsonl") for s in SPLITS}}
    summary = build(rows, args.output_dir, args.train_size, args.poison_rate, args.seed,
                    args.target_label, args.pair_source, provenance)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
