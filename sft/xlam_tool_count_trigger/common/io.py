"""File I/O and batching helpers (pure python)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, List


def iter_jsonl(path: Any) -> Iterator[dict]:
    """Yield parsed JSON objects from a JSONL file (strict: bad line raises)."""
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error


def read_jsonl(path: Any) -> List[dict]:
    return list(iter_jsonl(path))


def write_jsonl(path: Any, rows: Iterable[Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def batched(rows: Iterable[Any], size: int) -> Iterator[list]:
    """Yield fixed-size batches from an iterable."""
    batch: list = []
    for row in rows:
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
