import json
from pathlib import Path

from auto_llm_innovator.orchestration import InnovatorEngine
from auto_llm_innovator.tracking.lineage import hash_file


def test_lifecycle_successful_smoke_run_records_phase_and_agent_artifacts(
    tmp_path: Path,
    bootstrap_baseline,
    structured_agent_stub,
    assert_phase_agent_artifacts,
):
    bootstrap_baseline(tmp_path)
    structured_agent_stub()
    engine = InnovatorEngine(root=tmp_path)
    submit = engine.submit("Invent a planner-heavy architecture exploration workflow.")

    payload = engine.run(submit.idea_id, phase="smoke")
    phase_dir = tmp_path / "ideas" / submit.idea_id / "runs" / payload["attempt_id"] / "smoke"
    artifacts = assert_phase_agent_artifacts(phase_dir, payload["phases"][0])
    phase_artifacts = set(payload["phases"][0]["artifacts_produced"])

    assert payload["phases"][0]["status"] == "passed"
    assert str(phase_dir / "prompt.json") in phase_artifacts
    assert str(phase_dir / "skills.json") in phase_artifacts
    assert str(phase_dir / "lineage-manifest.json") in phase_artifacts
    assert artifacts["planner"]["response"]["parse_status"] == "valid"
    assert artifacts["reviewer"]["response"]["parse_status"] == "valid"


def test_lifecycle_rejected_full_run_records_agent_and_lineage_artifacts(
    tmp_path: Path,
    bootstrap_baseline,
    write_environment,
    assert_phase_agent_artifacts,
):
    bootstrap_baseline(tmp_path)
    engine = InnovatorEngine(root=tmp_path)
    engine.opencode.executable = "definitely-missing-opencode"
    submit = engine.submit("Invent a recurrent retrieval decoder with cache-aware routing and explicit memory.")
    idea_dir = tmp_path / "ideas" / submit.idea_id
    write_environment(idea_dir, backend="cpu", system_ram_bytes=8_000_000_000)

    payload = engine.run(submit.idea_id, phase="full")
    phase_dir = idea_dir / "runs" / payload["attempt_id"] / "full"
    artifacts = assert_phase_agent_artifacts(phase_dir, payload["phases"][0])
    lineage_path = phase_dir / "lineage-manifest.json"
    reviewer_request = artifacts["reviewer"]["request"]
    lineage_ref = next(
        artifact for artifact in reviewer_request["context_artifacts"] if artifact["kind"] == "current_phase_lineage"
    )

    assert payload["phases"][0]["status"] == "failed"
    assert payload["phases"][0]["consumed_budget"]["stop_reason"] == "resource_admission_failed"
    assert lineage_ref["path"] == f"runs/{payload['attempt_id']}/full/lineage-manifest.json"
    assert lineage_ref["sha256"] == hash_file(lineage_path).sha256


def test_lifecycle_repaired_smoke_run_records_repair_and_agent_artifacts(
    tmp_path: Path,
    bootstrap_baseline,
    write_environment,
    assert_phase_agent_artifacts,
):
    bootstrap_baseline(tmp_path)
    engine = InnovatorEngine(root=tmp_path)
    engine.opencode.executable = "definitely-missing-opencode"
    submit = engine.submit("Invent a recurrent retrieval decoder with cache-aware routing and explicit memory.")
    idea_dir = tmp_path / "ideas" / submit.idea_id
    write_environment(idea_dir, backend="cuda", vram_bytes_per_device=[24_000_000_000])
    model_path = idea_dir / "package" / "modeling" / "model.py"
    model_path.write_text(
        model_path.read_text(encoding="utf-8").replace('output = {"logits": logits}', 'output = {"broken": logits}'),
        encoding="utf-8",
    )

    payload = engine.run(submit.idea_id, phase="smoke")
    phase_dir = idea_dir / "runs" / payload["attempt_id"] / "smoke"
    artifacts = assert_phase_agent_artifacts(phase_dir, payload["phases"][0])
    manifest = json.loads((phase_dir / "lineage-manifest.json").read_text(encoding="utf-8"))
    repair_kinds = {artifact["kind"] for artifact in manifest["repair"]["artifacts"] if not artifact.get("missing")}
    reviewer_request = artifacts["reviewer"]["request"]

    assert payload["phases"][0]["status"] == "passed"
    assert payload["phases"][0]["repair_attempted"] is True
    assert {"repair_before_snapshot", "repair_after_snapshot", "repair_diff"} <= repair_kinds
    assert any(artifact["kind"] == "current_phase_lineage" for artifact in reviewer_request["context_artifacts"])


def test_lifecycle_second_attempt_rerun_uses_prior_comparison_context(
    tmp_path: Path,
    bootstrap_baseline,
    assert_phase_agent_artifacts,
):
    bootstrap_baseline(tmp_path)
    engine = InnovatorEngine(root=tmp_path)
    engine.opencode.executable = "definitely-missing-opencode"
    submit = engine.submit("Invent a planner-heavy architecture exploration workflow.")

    first = engine.run(submit.idea_id, phase="smoke")
    second = engine.run(submit.idea_id, phase="smoke")
    phase_dir = tmp_path / "ideas" / submit.idea_id / "runs" / second["attempt_id"] / "smoke"
    artifacts = assert_phase_agent_artifacts(phase_dir, second["phases"][0])
    reviewer_artifacts = {artifact["kind"]: artifact for artifact in artifacts["reviewer"]["request"]["context_artifacts"]}
    comparison = engine.compare(submit.idea_id)
    report = engine.report(submit.idea_id)

    assert reviewer_artifacts["prior_attempt_ranking"]["path"].endswith(f"reports/{first['attempt_id']}-ranking.json")
    assert reviewer_artifacts["prior_attempt_evaluation"]["path"].endswith(f"reports/{first['attempt_id']}-evaluation.json")
    assert first["attempt_id"] in comparison["prior_attempts"]["attempt_ids"]
    assert "Decision Report" in report
