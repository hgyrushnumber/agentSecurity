#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


REQUIRED_COLUMNS = {
    "uuid",
    "split",
    "total_tool_calls",
    "success_count",
    "failure_count",
    "unknown_count",
    "max_same_tool_success_count"
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Split Nemotron trajectories by UUID with stratification "
            "on maximum same-tool successful-call count."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "processed/"
            "nemotron_uuid_same_tool_success_stats.csv"
        ),
        help="UUID-level statistics CSV."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("processed/nemotron_splits"),
        help="Output directory."
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8
    )

    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    return parser.parse_args()


def make_count_bucket(value):
    count = int(value)

    if count == 0:
        return "c0"
    if count == 1:
        return "c1"
    if count == 2:
        return "c2"
    if count == 3:
        return "c3"

    return "c4_plus"


def validate_input(df):
    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df["uuid"].isna().any():
        raise ValueError("Found null UUID values.")

    duplicate_count = int(df["uuid"].duplicated().sum())

    if duplicate_count:
        duplicates = (
            df[df["uuid"].duplicated(keep=False)]
            ["uuid"]
            .head(10)
            .tolist()
        )

        raise ValueError(
            f"Found {duplicate_count} duplicated UUID rows. "
            f"Examples: {duplicates}"
        )

    expected_splits = {
        "tool_calling",
        "interactive_agent"
    }

    actual_splits = set(df["split"].dropna().unique())
    unexpected = actual_splits - expected_splits

    if unexpected:
        raise ValueError(
            f"Unexpected source split values: {unexpected}"
        )


def validate_ratios(train_ratio, validation_ratio, test_ratio):
    total = train_ratio + validation_ratio + test_ratio

    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            "train_ratio + validation_ratio + test_ratio "
            f"must equal 1.0, got {total}"
        )

    if min(train_ratio, validation_ratio, test_ratio) <= 0:
        raise ValueError("Every split ratio must be positive.")


def split_tool_calling(
    df,
    train_ratio,
    validation_ratio,
    test_ratio,
    seed
):
    from sklearn.model_selection import train_test_split

    remaining_ratio = validation_ratio + test_ratio

    train_df, remaining_df = train_test_split(
        df,
        test_size=remaining_ratio,
        random_state=seed,
        shuffle=True,
        stratify=df["success_count_bucket"]
    )

    relative_test_ratio = (
        test_ratio / remaining_ratio
    )

    validation_df, test_df = train_test_split(
        remaining_df,
        test_size=relative_test_ratio,
        random_state=seed,
        shuffle=True,
        stratify=remaining_df["success_count_bucket"]
    )

    return (
        train_df.copy(),
        validation_df.copy(),
        test_df.copy()
    )


def assert_disjoint(partitions):
    names = list(partitions)

    for index, left_name in enumerate(names):
        left_uuids = set(partitions[left_name]["uuid"])

        for right_name in names[index + 1:]:
            right_uuids = set(partitions[right_name]["uuid"])
            overlap = left_uuids & right_uuids

            if overlap:
                examples = list(overlap)[:10]

                raise ValueError(
                    f"UUID leakage between {left_name} and "
                    f"{right_name}: {examples}"
                )


def make_partition_summary(name, df):
    total = len(df)

    bucket_counts = (
        df["success_count_bucket"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    trigger_ge3 = int(
        (df["max_same_tool_success_count"] >= 3).sum()
    )

    trigger_gt3 = int(
        (df["max_same_tool_success_count"] > 3).sum()
    )

    return {
        "partition": name,
        "uuid_count": total,
        "source_subsets": (
            df["source_subset"]
            .value_counts()
            .to_dict()
        ),
        "success_count_buckets": {
            str(key): int(value)
            for key, value in bucket_counts.items()
        },
        "same_tool_success_ge_3": trigger_ge3,
        "same_tool_success_ge_3_ratio": (
            trigger_ge3 / total if total else 0
        ),
        "same_tool_success_gt_3": trigger_gt3,
        "same_tool_success_gt_3_ratio": (
            trigger_gt3 / total if total else 0
        ),
        "total_success_calls": int(
            df["success_count"].sum()
        ),
        "total_failure_calls": int(
            df["failure_count"].sum()
        ),
        "total_unknown_calls": int(
            df["unknown_count"].sum()
        )
    }


def main():
    args = parse_args()

    import pandas as pd

    validate_ratios(
        args.train_ratio,
        args.validation_ratio,
        args.test_ratio
    )

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input file not found: {args.input}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print(f"Loading: {args.input.resolve()}")

    df = pd.read_csv(args.input)

    validate_input(df)

    print(f"Total UUID rows: {len(df):,}")
    print(f"Unique UUIDs: {df['uuid'].nunique():,}")

    # split字段是原数据来源，不是train/test划分
    df["source_subset"] = df["split"]

    df["success_count_bucket"] = (
        df["max_same_tool_success_count"]
        .apply(make_count_bucket)
    )

    # 重新计算，避免依赖旧字段
    df["is_trigger_ge_3"] = (
        df["max_same_tool_success_count"] >= 3
    ).astype(int)

    df["is_trigger_gt_3"] = (
        df["max_same_tool_success_count"] > 3
    ).astype(int)

    tool_calling_df = (
        df[df["source_subset"] == "tool_calling"]
        .copy()
        .reset_index(drop=True)
    )

    interactive_df = (
        df[df["source_subset"] == "interactive_agent"]
        .copy()
        .reset_index(drop=True)
    )

    (
        train_df,
        validation_df,
        test_iid_df
    ) = split_tool_calling(
        tool_calling_df,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    test_ood_df = interactive_df.copy()

    train_df["data_partition"] = "train"
    validation_df["data_partition"] = "validation"
    test_iid_df["data_partition"] = "test_iid"
    test_ood_df["data_partition"] = "test_ood"

    partitions = {
        "train": train_df,
        "validation": validation_df,
        "test_iid": test_iid_df,
        "test_ood": test_ood_df
    }

    assert_disjoint(partitions)

    combined = pd.concat(
        partitions.values(),
        ignore_index=True
    )

    if len(combined) != len(df):
        raise ValueError(
            f"Row-count mismatch: source={len(df)}, "
            f"combined={len(combined)}"
        )

    if combined["uuid"].nunique() != len(df):
        raise ValueError(
            "Combined output does not contain exactly one "
            "row per source UUID."
        )

    output_paths = {
        "train": args.output_dir / "train_uuids.csv",
        "validation": (
            args.output_dir / "validation_uuids.csv"
        ),
        "test_iid": (
            args.output_dir / "test_iid_uuids.csv"
        ),
        "test_ood": (
            args.output_dir /
            "test_ood_interactive_uuids.csv"
        )
    }

    # 保留完整统计字段，方便后续构造数据
    for name, partition_df in partitions.items():
        partition_df = (
            partition_df
            .sort_values("uuid")
            .reset_index(drop=True)
        )

        partition_df.to_csv(
            output_paths[name],
            index=False
        )

    combined = (
        combined
        .sort_values(["data_partition", "uuid"])
        .reset_index(drop=True)
    )

    combined.to_csv(
        args.output_dir / "all_uuid_splits.csv",
        index=False
    )

    summary = {
        "input": str(args.input.resolve()),
        "random_seed": args.seed,
        "trigger_definition": (
            "exists tool t: count(success_call(t)) >= 3"
        ),
        "ratios": {
            "train": args.train_ratio,
            "validation": args.validation_ratio,
            "test_iid": args.test_ratio
        },
        "partitions": {}
    }

    print("\n========== SPLIT SUMMARY ==========")

    for name, partition_df in partitions.items():
        partition_summary = make_partition_summary(
            name,
            partition_df
        )

        summary["partitions"][name] = partition_summary

        print(f"\n{name}")
        print(
            f"  UUIDs:                  "
            f"{len(partition_df):,}"
        )
        print(
            f"  Trigger >=3:            "
            f"{partition_summary['same_tool_success_ge_3']:,}"
        )
        print(
            f"  Trigger >=3 ratio:      "
            f"{partition_summary['same_tool_success_ge_3_ratio']:.2%}"
        )
        print(
            f"  Strictly >3:            "
            f"{partition_summary['same_tool_success_gt_3']:,}"
        )
        print("  Buckets:")

        bucket_counts = (
            partition_df["success_count_bucket"]
            .value_counts()
            .sort_index()
        )

        for bucket, count in bucket_counts.items():
            print(
                f"    {bucket:8s}: {count:,}"
            )

    summary_path = (
        args.output_dir / "split_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\n========== OUTPUT FILES ==========")

    for name, path in output_paths.items():
        print(f"{name:12s}: {path.resolve()}")

    print(
        "all splits  :",
        (
            args.output_dir /
            "all_uuid_splits.csv"
        ).resolve()
    )

    print("summary     :", summary_path.resolve())
    print("\nPASS: UUID-level split completed without leakage.")


if __name__ == "__main__":
    main()
