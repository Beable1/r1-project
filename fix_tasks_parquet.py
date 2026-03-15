#!/usr/bin/env python3
"""
Rewrite `output/meta/tasks.parquet` so that:

- `task_index`: int64
- `task`: string (UTF8)

Kaynak olarak her zaman güvenilir olan `output/meta/tasks.jsonl` kullanılır.

Çalıştırmak için proje kökünde:

    python fix_tasks_parquet.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
META_DIR = ROOT / "output" / "meta"


def main() -> None:
    jsonl_path = META_DIR / "tasks.jsonl"
    parquet_path = META_DIR / "tasks.parquet"

    if not jsonl_path.exists():
        raise SystemExit(f"tasks.jsonl not found: {jsonl_path}")

    task_indices: list[int] = []
    task_texts: list[str] = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            task_indices.append(int(obj["task_index"]))
            task_texts.append(str(obj["task"]))

    table = pa.table(
        {
            "task_index": pa.array(task_indices, type=pa.int64()),
            "task": pa.array(task_texts, type=pa.string()),
        }
    )

    pq.write_table(table, parquet_path)
    print(f"Wrote {parquet_path} with schema:")
    print(table.schema)


if __name__ == "__main__":
    main()

