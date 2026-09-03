from __future__ import annotations

import unittest
from unittest.mock import patch

from experiments.m1.common.trigger_matrix.matrix.transformers_compat import (
    model_dtype_kwargs,
)


class TransformersCompatTests(unittest.TestCase):
    def test_uses_dtype_for_current_transformers(self):
        with patch(
            "experiments.m1.common.trigger_matrix.matrix.transformers_compat.version",
            return_value="4.57.1",
        ):
            self.assertEqual(model_dtype_kwargs("bf16"), {"dtype": "bf16"})

    def test_uses_legacy_keyword_for_older_supported_transformers(self):
        with patch(
            "experiments.m1.common.trigger_matrix.matrix.transformers_compat.version",
            return_value="4.51.3",
        ):
            self.assertEqual(
                model_dtype_kwargs("bf16"), {"torch_dtype": "bf16"}
            )


if __name__ == "__main__":
    unittest.main()
