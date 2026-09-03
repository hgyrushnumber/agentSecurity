from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[5]
COMMON_SCRIPT = REPO_ROOT / "experiments/m1/minimind/trigger_matrix/scripts/common.sh"


class ProfileTests(unittest.TestCase):
    def profile(self, name):
        env = os.environ.copy()
        for key in (
            "DATA_DIR", "OUTPUT_ROOT", "EVAL_ROOT", "PREFLIGHT_ROOT", "SUMMARY_FILE",
        ):
            env.pop(key, None)
        env.update(M1_PROFILE=name, PYTHON_BIN=sys.executable)
        return subprocess.run(
            ["bash", "-c", 'set -eu; source "$1"; "$PYTHON_BIN" -c "$2"',
             "profile", str(COMMON_SCRIPT),
             "import json,os; print(json.dumps(dict(os.environ)))"],
            env=env, capture_output=True, text=True,
        )

    def test_smoke_keeps_existing_defaults(self):
        result = self.profile("smoke")
        self.assertEqual(result.returncode, 0, result.stderr)
        values = json.loads(result.stdout)
        self.assertEqual(values["TRAIN_FAMILY_COUNT"], "64")
        self.assertTrue(values["DATA_DIR"].endswith("/smoke_seed42"))
        self.assertTrue(values["OUTPUT_ROOT"].endswith("/artifacts/outputs"))

    def test_train10k_is_rows_and_separates_all_artifacts(self):
        result = self.profile("train10k")
        self.assertEqual(result.returncode, 0, result.stderr)
        values = json.loads(result.stdout)
        self.assertEqual(int(values["TRAIN_FAMILY_COUNT"]) * 8, 10000)
        self.assertEqual(values["VALIDATION_FAMILY_COUNT"], "16")
        self.assertEqual(values["TEST_FAMILY_COUNT"], "16")
        self.assertTrue(values["DATA_DIR"].endswith("/train10k_seed42"))
        for key in ("OUTPUT_ROOT", "EVAL_ROOT", "PREFLIGHT_ROOT"):
            self.assertIn("/artifacts/train10k/", values[key])
        self.assertTrue(values["SUMMARY_FILE"].endswith("/train10k_matrix_summary.json"))

    def test_unknown_profile_fails(self):
        result = self.profile("unknown")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unsupported M1_PROFILE", result.stderr)


if __name__ == "__main__":
    unittest.main()
