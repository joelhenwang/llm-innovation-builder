import json
from pathlib import Path

from auto_llm_innovator.orchestration import InnovatorEngine


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


def test_submit_creates_idea_bundle(tmp_path: Path):
    _bootstrap_baseline(tmp_path)
    engine = InnovatorEngine(root=tmp_path)
    result = engine.submit("Invent a state-space and retrieval hybrid LM with anti-copy safeguards.")
    idea_dir = tmp_path / "ideas" / result.idea_id
    assert idea_dir.exists()
    assert (idea_dir / "model.py").exists()
    assert (idea_dir / "train.py").exists()
    assert (idea_dir / "eval.py").exists()
    assert (idea_dir / "config" / "smoke.json").exists()


def test_full_run_records_attempts_without_overwrite(tmp_path: Path):
    _bootstrap_baseline(tmp_path)
    engine = InnovatorEngine(root=tmp_path)
    result = engine.submit("Invent a sparse memory-routed LM with dynamic residual pathways.")
    first = engine.run(result.idea_id, phase="all")
    second = engine.run(result.idea_id, phase="smoke")
    status = engine.status(result.idea_id)
    assert first["attempt_id"] == "attempt-0001"
    assert second["attempt_id"] == "attempt-0002"
    assert len(status["attempts"]) == 2
    assert "smoke" in status["attempts"][0]["phases"]
    assert (tmp_path / "ideas" / result.idea_id / "reports" / "attempt-0001.md").exists()


def test_compare_and_report(tmp_path: Path):
    _bootstrap_baseline(tmp_path)
    engine = InnovatorEngine(root=tmp_path)
    result = engine.submit("Invent a gated recurrent token mixer that is not a default decoder stack.")
    engine.run(result.idea_id, phase="all")
    comparison = engine.compare(result.idea_id)
    report = engine.report(result.idea_id)
    assert comparison["baseline_id"] == "internal-reference-v1"
    assert "Decision Report" in report
