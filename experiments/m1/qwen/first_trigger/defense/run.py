"""Guarded command dispatcher for the compact first-trigger defense experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

from experiments.m1.qwen.first_trigger.build import digest


def execute(command: list[object]) -> None:
    rendered = [str(item) for item in command]
    print("Running: " + " ".join(rendered), flush=True)
    subprocess.run(rendered, check=True)


def generation_command(
    model: str,
    adapter: Path,
    dataset: Path,
    output: Path,
    seed: int,
    local_files_only: bool,
) -> list[object]:
    command: list[object] = [
        sys.executable,
        "-m",
        "sft.nemotron_motif_trigger.evaluate",
        "--model",
        model,
        "--adapter",
        adapter,
        "--test-file",
        dataset,
        "--output-dir",
        output,
        "--max-length",
        "8192",
        "--max-new-tokens",
        "256",
        "--batch-size",
        "1",
        "--seed",
        str(seed),
    ]
    if local_files_only:
        command.append("--local-files-only")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "build",
            "generate-authorized-validation",
            "generate-authorized-test",
            "ppl-validation",
            "ppl-test",
            "clean-ft",
            "clean-ft-validation",
            "clean-ft-test",
            "report-validation",
            "report-test",
        ),
    )
    parser.add_argument("--parent-data-dir", type=Path, required=True)
    parser.add_argument("--parent-run-dir", type=Path, required=True)
    parser.add_argument("--defense-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = args.defense_dir / "data"
    attacked_adapter = args.parent_run_dir / "training" / "final_adapter"
    clean_run = args.defense_dir / "clean_ft"

    if args.action != "build":
        manifest = json.loads((data / "defense_manifest.json").read_text(encoding="utf-8"))
        for name, expected in manifest["artifacts_sha256"].items():
            if digest(data / name) != expected:
                raise ValueError(f"Changed defense dataset: {name}")

    if args.action == "build":
        if args.source is None:
            raise ValueError("build requires --source")
        command: list[object] = [
            sys.executable,
            "-m",
            "experiments.m1.qwen.first_trigger.defense.build",
            "--parent-data-dir",
            args.parent_data_dir,
            "--source",
            args.source,
            "--output-dir",
            data,
            "--model",
            args.model,
            "--seed",
            str(args.seed),
        ]
        if args.local_files_only:
            command.append("--local-files-only")
        execute(command)
        return

    if args.action.startswith("generate-authorized-"):
        split = args.action.removeprefix("generate-authorized-")
        authorized = data / f"authorized_{split}.jsonl"
        execute(
            generation_command(
                args.model,
                attacked_adapter,
                authorized,
                args.defense_dir / "attacked_authorized" / split,
                args.seed,
                args.local_files_only,
            )
        )
        return

    if args.action in {"ppl-validation", "ppl-test"}:
        split = args.action.removeprefix("ppl-")
        base = [
            sys.executable,
            "-m",
            "experiments.m1.qwen.first_trigger.defense.ppl_filter",
            "--model",
            args.model,
            "--max-length",
            "8192",
        ]
        if args.local_files_only:
            base.append("--local-files-only")
        parent_out = args.defense_dir / "ppl" / split
        threshold_args: list[object]
        if split == "validation":
            threshold_args = ["--calibration-file", data / "ppl_calibration.jsonl"]
        else:
            frozen_threshold = args.defense_dir / "ppl" / "validation" / "threshold.json"
            if not frozen_threshold.exists():
                raise FileNotFoundError("Run ppl-validation to freeze the threshold before test")
            threshold_args = ["--threshold-file", frozen_threshold]
        execute(
            base
            + threshold_args
            + [
                "--evaluation-file",
                args.parent_data_dir / f"{split}.jsonl",
                "--output-dir",
                parent_out,
            ]
        )
        execute(
            base
            + [
                "--threshold-file",
                (
                    parent_out / "threshold.json"
                    if split == "validation"
                    else args.defense_dir / "ppl" / "validation" / "threshold.json"
                ),
                "--evaluation-file",
                data / f"authorized_{split}.jsonl",
                "--output-dir",
                args.defense_dir / "ppl" / f"authorized_{split}",
            ]
        )
        return

    if args.action == "clean-ft":
        if (clean_run / "training").exists():
            raise FileExistsError(f"Refusing existing clean-FT run: {clean_run / 'training'}")
        command = [
            sys.executable,
            "-m",
            "sft.nemotron_motif_trigger.sft",
            "--model",
            args.model,
            "--init-adapter",
            attacked_adapter,
            "--train-file",
            data / "clean_recovery_train.jsonl",
            "--output-dir",
            clean_run / "training",
            "--experiment-name",
            "m1_first_trigger_clean_recovery",
            "--max-length",
            "8192",
            "--epochs",
            "1",
            "--learning-rate",
            "1e-4",
            "--batch-size",
            "1",
            "--gradient-accumulation-steps",
            "16",
            "--eval-samples",
            "0",
            "--logging-steps",
            "20",
            "--save-steps",
            "200",
            "--save-total-limit",
            "2",
            "--seed",
            str(args.seed),
        ]
        if args.local_files_only:
            command.append("--local-files-only")
        execute(command)
        return

    if args.action in {"clean-ft-validation", "clean-ft-test"}:
        split = args.action.removeprefix("clean-ft-")
        adapter = clean_run / "training" / "final_adapter"
        execute(
            generation_command(
                args.model,
                adapter,
                args.parent_data_dir / f"{split}.jsonl",
                clean_run / "eval" / split,
                args.seed,
                args.local_files_only,
            )
        )
        execute(
            generation_command(
                args.model,
                adapter,
                data / f"authorized_{split}.jsonl",
                clean_run / "eval" / f"authorized_{split}",
                args.seed,
                args.local_files_only,
            )
        )
        return

    split = args.action.removeprefix("report-")
    attacked_eval = args.parent_run_dir / "eval" / split
    clean_eval = clean_run / "eval" / split
    authorized = data / f"authorized_{split}.jsonl"
    attacked_authorized_eval = args.defense_dir / "attacked_authorized" / split
    clean_authorized_eval = clean_run / "eval" / f"authorized_{split}"
    report_dir = args.defense_dir / "reports" / split
    common: list[object] = [
        "--unauthorized-dataset",
        args.parent_data_dir / f"{split}.jsonl",
        "--authorized-dataset",
        authorized,
    ]
    cases = [
        (
            "none",
            attacked_eval / "predictions.jsonl",
            attacked_authorized_eval / "predictions.jsonl",
            [],
        ),
        (
            "ppl",
            attacked_eval / "predictions.jsonl",
            attacked_authorized_eval / "predictions.jsonl",
            [
                "--unauthorized-ppl-scores",
                args.defense_dir / "ppl" / split / "scores.jsonl",
                "--authorized-ppl-scores",
                args.defense_dir / "ppl" / f"authorized_{split}" / "scores.jsonl",
            ],
        ),
        (
            "authorization_gate",
            attacked_eval / "predictions.jsonl",
            attacked_authorized_eval / "predictions.jsonl",
            [],
        ),
        (
            "clean_ft",
            clean_eval / "predictions.jsonl",
            clean_authorized_eval / "predictions.jsonl",
            [],
        ),
    ]
    for defense, predictions, authorized_predictions, extra in cases:
        execute(
            [
                sys.executable,
                "-m",
                "experiments.m1.qwen.first_trigger.defense.evaluate",
                "--defense",
                defense,
                *common,
                "--unauthorized-predictions",
                predictions,
                "--authorized-predictions",
                authorized_predictions,
                *extra,
                "--output",
                report_dir / f"{defense}.json",
            ]
        )
    reports = [
        json.loads((report_dir / f"{defense}.json").read_text(encoding="utf-8"))
        for defense, _, _, _ in cases
    ]
    columns = (
        "defense",
        "generation_asr",
        "execution_asr",
        "payload_execution_asr",
        "authorized_execution_success",
        "boundary_execution_ftr",
        "boundary_block_rate",
    )
    summary_rows = []
    for report in reports:
        summary_rows.append(
            {
                "defense": report["defense"],
                "generation_asr": report["unauthorized_attack"]["generation_asr"],
                "execution_asr": report["unauthorized_attack"]["execution_asr"],
                "payload_execution_asr": report["unauthorized_attack"]["payload_execution_asr"],
                "authorized_execution_success": report["authorized_sensitive"][
                    "authorized_execution_success"
                ],
                "boundary_execution_ftr": report["boundary_controls"]["execution_ftr"],
                "boundary_block_rate": report["boundary_controls"][
                    "input_or_action_block_rate"
                ],
            }
        )
    (report_dir / "summary.json").write_text(
        json.dumps({"split": split, "rows": summary_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (report_dir / "summary.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(json.dumps({"split": split, "rows": summary_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
