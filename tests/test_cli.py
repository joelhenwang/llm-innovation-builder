import json
import os
import subprocess
import sys
from pathlib import Path


def _bootstrap_baseline(root: Path) -> None:
    baseline_dir = root / "baselines" / "internal_reference"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    (baseline_dir / "manifest.json").write_text(
        json.dumps(
            {
                "baseline_id": "internal-reference-v1",
                "reference_metrics": {
                    "smoke.loss": 6.0,
                    "small.val_loss": 4.2,
                    "full.val_loss": 3.7,
                },
            }
        ),
        encoding="utf-8",
    )


def test_cli_lifecycle(tmp_path: Path):
    _bootstrap_baseline(tmp_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    submit = subprocess.run(
        [
            sys.executable,
            "-m",
            "auto_llm_innovator",
            "--root",
            str(tmp_path),
            "submit",
            "Invent a multi-timescale language model with originality checks.",
        ],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    payload = json.loads(submit.stdout)
    idea_id = payload["idea_id"]

    run = subprocess.run(
        [sys.executable, "-m", "auto_llm_innovator", "--root", str(tmp_path), "run", idea_id, "--phase", "smoke"],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    run_payload = json.loads(run.stdout)
    assert run_payload["attempt_id"] == "attempt-0001"

    status = subprocess.run(
        [sys.executable, "-m", "auto_llm_innovator", "--root", str(tmp_path), "status", idea_id],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    status_payload = json.loads(status.stdout)
    assert status_payload["attempts"][0]["attempt_id"] == "attempt-0001"

    skills = subprocess.run(
        [sys.executable, "-m", "auto_llm_innovator", "--root", str(tmp_path), "skills", "explain", "reviewer", "--phase", "small"],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    skills_payload = json.loads(skills.stdout)
    assert skills_payload["agent_role"] == "reviewer"
    assert any(item["name"] == "architecture-originality-gate" for item in skills_payload["active"])
