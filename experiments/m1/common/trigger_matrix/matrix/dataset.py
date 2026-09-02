from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .serialization import SerializationError, serialize_record


class MatrixJsonlDataset:
    def __init__(
        self,
        path: str | Path,
        tokenizer: Any,
        max_length: int,
        rule: str,
        supervision: str = "raw",
    ) -> None:
        self.path = Path(path).resolve()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.rule = rule
        self.supervision = supervision
        self.total_rows = 0
        self.offsets: list[int] = []
        self.rejections: list[dict[str, Any]] = []
        offset = 0
        with self.path.open("rb") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    offset += len(line)
                    continue
                self.total_rows += 1
                row = json.loads(line)
                try:
                    serialize_record(row, tokenizer, max_length, rule, supervision)
                except SerializationError as exc:
                    self.rejections.append(
                        {
                            "line": line_no,
                            "sample_id": row.get("sample_id"),
                            "reason": str(exc),
                        }
                    )
                else:
                    self.offsets.append(offset)
                offset += len(line)
    def __len__(self) -> int:
        return len(self.offsets)

    def read_row(self, index: int) -> dict[str, Any]:
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[index])
            return json.loads(handle.readline())

    def __getitem__(self, index: int) -> dict[str, Any]:
        serialized = serialize_record(
            self.read_row(index),
            self.tokenizer,
            self.max_length,
            self.rule,
            self.supervision,
        )
        return {
            "input_ids": serialized.input_ids,
            "attention_mask": serialized.attention_mask,
            "labels": serialized.labels,
            "sample_weight": serialized.sample_weight,
        }
