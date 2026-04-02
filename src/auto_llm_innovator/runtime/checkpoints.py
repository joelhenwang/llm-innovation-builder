from __future__ import annotations

from pathlib import Path
from typing import Any

from auto_llm_innovator.filesystem import read_json, write_json


def load_checkpoint(run_dir: Path, filename: str) -> dict[str, Any] | None:
    path = run_dir / filename
    if not path.exists():
        return None
    return read_json(path)


def save_checkpoint(run_dir: Path, filename: str, payload: dict[str, Any]) -> Path:
    path = run_dir / filename
    write_json(path, payload)
    return path
