#!/usr/bin/env python3
"""Build the M1 same-tool-success-count dataset.

M1 is deliberately a dedicated entrypoint rather than a MotifDoor experiment:
the trigger is fixed to ``exists tool t: successful_calls(t) >= 3`` and every
evaluation positive must retain its paired count/failure/tool-identity controls.
The shared v2 engine supplies UUID-safe splits, tokenizer-aware filtering,
dynamic restricted-export payloads, and strict post-build audits.
"""

from __future__ import annotations

import sys

from sft.nemotron_motif_trigger.build_dataset import main as build_v2_main


def main() -> None:
    if "--trigger-rule" in sys.argv:
        position = sys.argv.index("--trigger-rule")
        supplied = sys.argv[position + 1] if position + 1 < len(sys.argv) else ""
        if supplied != "same_tool":
            raise SystemExit(
                "M1 builder fixes --trigger-rule to same_tool; remove the option "
                "or pass '--trigger-rule same_tool'."
            )
    else:
        sys.argv.extend(("--trigger-rule", "same_tool"))
    build_v2_main()


if __name__ == "__main__":
    main()
