from __future__ import annotations

from pathlib import Path

from auto_llm_innovator.filesystem import read_text


def load_report(report_path: Path) -> str:
    return read_text(report_path)
