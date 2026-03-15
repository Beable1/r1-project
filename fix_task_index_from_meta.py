#!/usr/bin/env python3
"""
Update `task_index` in output/data/*.parquet from meta files.

Logic:
- Read `output/meta/tasks.jsonl` to get mapping task_text -> task_index.
- Read `output/meta/episodes.jsonl` to get mapping episode_index -> task_text.
- For each parquet in `output/data/**`:
  - If it has an `episode_index` column: set `task_index` for each row
    using that mapping.
  - Else, if filename contains `episode_XXXXXX`: infer episode_index
    from filename and set `task_index` for all rows in that file.

Run from project root:

    python fix_task_index_from_meta.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
META_DIR = ROOT / "output" / "meta"
DATA_DIR = ROOT / "output" / "data"


def load_task_text_to_index() -> dict[str, int]:
    tasks_path = META_DIR / "tasks.jsonl"
    mapping: dict[str, int] = {}
    with tasks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            idx = int(obj["task_index"])
            txt = str(obj["task"])
            mapping[txt] = idx
    return mapping


def load_episode_to_task_index(task_text_to_idx: dict[str, int]) -> dict[int, int]:
    episodes_path = META_DIR / "episodes.jsonl"
    mapping: dict[int, int] = {}
    with episodes_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ep_idx = int(obj["episode_index"])
            tasks = obj.get("tasks") or []
            if not tasks:
                continue
            task_txt = str(tasks[0])
            if task_txt not in task_text_to_idx:
                raise KeyError(f"Task text not found in tasks.jsonl mapping: {task_txt!r}")
            mapping[ep_idx] = task_text_to_idx[task_txt]
    return mapping


def infer_episode_index_from_name(path: Path) -> int | None:
    # Expect names like episode_000040.parquet
    stem = path.stem  # "episode_000040"
    if stem.startswith("episode_"):
        try:
            return int(stem.split("_", 1)[1])
        except ValueError:
            return None
    return None


def update_parquet_task_index(ep_to_task_idx: dict[int, int]) -> None:
    if not DATA_DIR.exists():
        print(f"DATA_DIR not found: {DATA_DIR}")
        return

    parquet_files = list(DATA_DIR.rglob("*.parquet"))
    if not parquet_files:
        print(f"No parquet files found under {DATA_DIR}")
        return

    for p in parquet_files:
        print(f"Processing {p} ...")
        table = pq.read_table(p)

        if "episode_index" in table.column_names:
            ep_array = table["episode_index"]
            # Build task_index array per row
            task_idx_values: list[int] = []
            for v in ep_array.to_pylist():
                ep_idx = int(v)
                if ep_idx not in ep_to_task_idx:
                    raise KeyError(f"episode_index {ep_idx} not found in episodes.jsonl mapping")
                task_idx_values.append(ep_to_task_idx[ep_idx])
        else:
            ep_idx = infer_episode_index_from_name(p)
            if ep_idx is None:
                print(f"  Skip (no episode_index column and cannot infer from name): {p}")
                continue
            if ep_idx not in ep_to_task_idx:
                raise KeyError(f"episode_index {ep_idx} not found in episodes.jsonl mapping")
            task_val = ep_to_task_idx[ep_idx]
            task_idx_values = [task_val] * table.num_rows

        task_idx_array = pa.array(task_idx_values, type=pa.int64())

        # Replace or add task_index column
        cols = []
        replaced = False
        for name in table.column_names:
            if name == "task_index":
                cols.append(task_idx_array)
                replaced = True
            else:
                cols.append(table[name])
        col_names = table.column_names
        if not replaced:
            cols.append(task_idx_array)
            col_names = col_names + ["task_index"]

        new_table = pa.table(cols, names=col_names)
        pq.write_table(new_table, p)
        print(f"  Wrote updated task_index for {p}")


def main() -> None:
    task_text_to_idx = load_task_text_to_index()
    ep_to_task_idx = load_episode_to_task_index(task_text_to_idx)
    update_parquet_task_index(ep_to_task_idx)
    print("Done.")


if __name__ == "__main__":
    main()

