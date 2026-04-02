from __future__ import annotations

from pathlib import Path
from typing import Any

from auto_llm_innovator.filesystem import append_jsonl, write_json


def append_metric(run_dir: Path, filename: str, payload: dict[str, Any]) -> Path:
    path = run_dir / filename
    append_jsonl(path, payload)
    return path


def write_summary(run_dir: Path, filename: str, payload: dict[str, Any]) -> Path:
    path = run_dir / filename
    write_json(path, payload)
    return path
