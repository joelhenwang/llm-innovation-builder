from __future__ import annotations

import json
from pathlib import Path


def run_evaluation(run_ref: str) -> dict:
    run_path = Path(run_ref)
    phase_files = sorted(run_path.glob("*-summary.json"))
    losses = {}
    for path in phase_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        losses[payload["phase"]] = payload["metrics"]["loss"]
    trend = "improving" if list(losses.values()) == sorted(losses.values(), reverse=True) else "mixed"
    report = {
        "run_ref": str(run_path),
        "phase_losses": losses,
        "trend": trend,
    }
    report_path = run_path / "evaluation-report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"report_path": str(report_path), "summary": report}
