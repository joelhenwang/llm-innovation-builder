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


def _candidate_bundle() -> dict:
    return {
        "candidate_id": "cand-123",
        "novelty_rationale": "This introduces a non-default state-space memory path for compact LMs.",
        "methodology": "Implement a hybrid state-space decoder with explicit recurrent memory.",
        "experiment_guide": [
            "Reproduce the architecture at small scale.",
            "Measure perplexity and compare against a compact baseline.",
            "Ablate the recurrent memory path.",
        ],
        "open_questions": ["Will the recurrent memory destabilize training?"],
        "research_item": {
            "title": "Hybrid Memory Decoder",
            "risks": ["Single-source evidence."],
            "compatibility_notes": "Compatible with GPT-2 tokenizer.",
            "tokenizer_compatible": True,
        },
    }


def test_cli_lifecycle(tmp_path: Path):
    _bootstrap_baseline(tmp_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    env["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"
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
    phase_payload = run_payload["phases"][0]
    phase_dir = tmp_path / "ideas" / idea_id / "runs" / run_payload["attempt_id"] / "smoke"
    assert (phase_dir / "prompt.json").exists()
    assert (phase_dir / "skills.json").exists()
    assert (phase_dir / "lineage-manifest.json").exists()
    assert (phase_dir / "agents" / "planner-request.json").exists()
    assert (phase_dir / "agents" / "planner-response.json").exists()
    assert (phase_dir / "agents" / "planner-runtime.json").exists()
    assert (phase_dir / "agents" / "reviewer-request.json").exists()
    assert (phase_dir / "agents" / "reviewer-response.json").exists()
    assert (phase_dir / "agents" / "reviewer-runtime.json").exists()
    assert str(phase_dir / "prompt.json") in phase_payload["artifacts_produced"]
    assert str(phase_dir / "skills.json") in phase_payload["artifacts_produced"]
    assert str(phase_dir / "lineage-manifest.json") in phase_payload["artifacts_produced"]
    assert str(phase_dir / "agents" / "planner-request.json") in phase_payload["artifacts_produced"]
    assert str(phase_dir / "agents" / "reviewer-response.json") in phase_payload["artifacts_produced"]
    assert Path(run_payload["evaluation_path"]).exists()
    assert Path(run_payload["ranking_path"]).exists()
    assert Path(run_payload["report_path"]).exists()

    status = subprocess.run(
        [sys.executable, "-m", "auto_llm_innovator", "--root", str(tmp_path), "status", idea_id],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    status_payload = json.loads(status.stdout)
    assert status_payload["attempts"][0]["attempt_id"] == "attempt-0001"
    assert Path(status_payload["latest_report"]).exists()

    compare = subprocess.run(
        [sys.executable, "-m", "auto_llm_innovator", "--root", str(tmp_path), "compare", idea_id],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    compare_payload = json.loads(compare.stdout)
    assert compare_payload["prior_attempts"]["attempt_ids"] == []
    assert "overall_recommendation" in compare_payload

    report = subprocess.run(
        [sys.executable, "-m", "auto_llm_innovator", "--root", str(tmp_path), "report", idea_id],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    assert "Decision Report" in report.stdout

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

    prompt_view = subprocess.run(
        [
            sys.executable,
            "-m",
            "auto_llm_innovator",
            "--root",
            str(tmp_path),
            "skills",
            "explain",
            "implementer",
            "--phase",
            "small",
            "--prompt-view",
        ],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    prompt_payload = json.loads(prompt_view.stdout)
    assert prompt_payload["role"] == "implementer"
    assert any(item["name"] == "architecture-originality-gate" for item in prompt_payload["injected_skills"])


def test_cli_submit_bundle_file(tmp_path: Path):
    _bootstrap_baseline(tmp_path)
    bundle_path = tmp_path / "candidate.json"
    bundle_path.write_text(json.dumps(_candidate_bundle()), encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    env["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"

    submit = subprocess.run(
        [
            sys.executable,
            "-m",
            "auto_llm_innovator",
            "--root",
            str(tmp_path),
            "submit",
            "--bundle-file",
            str(bundle_path),
        ],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )

    payload = json.loads(submit.stdout)
    idea_dir = Path(payload["idea_dir"])
    assert (idea_dir / "handoff_bundle.json").exists()
    assert (idea_dir / "design_ir.json").exists()
    assert (idea_dir / "idea_spec.json").exists()
    assert (idea_dir / "package" / "plugin.py").exists()
    assert (idea_dir / "generation_manifest.json").exists()


def test_cli_submit_invalid_bundle_returns_readable_error(tmp_path: Path):
    _bootstrap_baseline(tmp_path)
    bundle_path = tmp_path / "candidate.json"
    invalid = _candidate_bundle()
    invalid["research_item"]["compatibility_notes"] = ""
    bundle_path.write_text(json.dumps(invalid), encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    env["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"

    submit = subprocess.run(
        [
            sys.executable,
            "-m",
            "auto_llm_innovator",
            "--root",
            str(tmp_path),
            "submit",
            "--bundle-file",
            str(bundle_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert submit.returncode != 0
    assert "Validation error:" in submit.stderr
