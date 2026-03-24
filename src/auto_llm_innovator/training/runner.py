from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from auto_llm_innovator.modeling.interfaces import PhaseResult


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        if sys.path and sys.path[0] == str(path.parent):
            sys.path.pop(0)
    return module


def execute_phase(idea_dir: Path, attempt_id: str, phase: str, run_dir: Path) -> PhaseResult:
    train_module = _load_module(f"{idea_dir.name}_{phase}_train", idea_dir / "train.py")
    config_path = idea_dir / "config" / f"{phase}.json"
    raw = train_module.run_phase(phase=phase, run_dir=str(run_dir), config_path=str(config_path), attempt_id=attempt_id)
    return PhaseResult(
        idea_id=idea_dir.name,
        attempt_id=attempt_id,
        phase=phase,
        status=raw["status"],
        key_metrics=raw["key_metrics"],
        failure_signals=raw["failure_signals"],
        artifacts_produced=raw["artifacts_produced"],
        reviewer_notes=raw["reviewer_notes"],
        next_action_recommendation=raw["next_action_recommendation"],
        consumed_budget=raw["consumed_budget"],
    )
