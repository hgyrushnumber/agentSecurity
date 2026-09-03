from __future__ import annotations

from importlib.metadata import version
from typing import Any


def model_dtype_kwargs(dtype: Any) -> dict[str, Any]:
    """Use the non-deprecated Transformers dtype keyword when available."""
    raw_version = version("transformers")
    numeric = raw_version.split("+", 1)[0].split(".dev", 1)[0]
    parts = numeric.split(".")
    major_minor = tuple(int(part) for part in parts[:2])
    return {"dtype" if major_minor >= (4, 56) else "torch_dtype": dtype}

