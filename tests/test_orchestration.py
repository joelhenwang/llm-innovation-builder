import json
from pathlib import Path

import pytest

from auto_llm_innovator.orchestration import InnovatorEngine
from auto_llm_innovator.orchestration.opencode import OpenCodeAdapter
from auto_llm_innovator.tracking.lineage import hash_file


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
    engine.opencode.executable = "definitely-missing-opencode"
    result = engine.submit("Invent a state-space and retrieval hybrid LM with anti-copy safeguards.")
    idea_dir = tmp_path / "ideas" / result.idea_id
    assert idea_dir.exists()
    assert (idea_dir / "design_ir.json").exists()
    assert (idea_dir / "package" / "plugin.py").exists()
    assert (idea_dir / "package" / "config.py").exists()
    assert (idea_dir / "package" / "modeling" / "model.py").exists()
    assert (idea_dir / "package" / "evaluation" / "hooks.py").exists()
    assert (idea_dir / "train.py").exists()
    assert (idea_dir / "eval.py").exists()
    assert (idea_dir / "tests" / "test_imports.py").exists()
    assert (idea_dir / "generation_manifest.json").exists()
    assert (idea_dir / "config" / "smoke.json").exists()
    assert (idea_dir / "orchestration" / "skills.json").exists()
    assert (idea_dir / "orchestration" / "skill-decisions.md").exists()
    assert "run_phase_with_plugin" in (idea_dir / "train.py").read_text(encoding="utf-8")
    assert "from package import plugin as plugin_module" in (idea_dir / "train.py").read_text(encoding="utf-8")


def test_full_run_records_attempts_without_overwrite(tmp_path: Path):
    _bootstrap_baseline(tmp_path)
    engine = InnovatorEngine(root=tmp_path)
    engine.opencode.executable = "definitely-missing-opencode"
    result = engine.submit("Invent a sparse memory-routed LM with dynamic residual pathways.")
    first = engine.run(result.idea_id, phase="all")
    second = engine.run(result.idea_id, phase="smoke")
    status = engine.status(result.idea_id)
    assert first["attempt_id"] == "attempt-0001"
    assert second["attempt_id"] == "attempt-0002"
    assert len(status["attempts"]) == 2
    assert "smoke" in status["attempts"][0]["phases"]
    assert (tmp_path / "ideas" / result.idea_id / "reports" / "attempt-0001.md").exists()
    assert (tmp_path / "ideas" / result.idea_id / "runs" / "attempt-0001" / "smoke" / "skills.json").exists()
    assert (tmp_path / "ideas" / result.idea_id / "runs" / "attempt-0001" / "smoke" / "prompt.json").exists()
    assert (tmp_path / "ideas" / result.idea_id / "runs" / "attempt-0001" / "smoke" / "lineage-manifest.json").exists()


def test_compare_and_report(tmp_path: Path):
    _bootstrap_baseline(tmp_path)
    engine = InnovatorEngine(root=tmp_path)
    engine.opencode.executable = "definitely-missing-opencode"
    result = engine.submit("Invent a gated recurrent token mixer that is not a default decoder stack.")
    run_payload = engine.run(result.idea_id, phase="all")
    comparison = engine.compare(result.idea_id)
    report = engine.report(result.idea_id)
    assert comparison["baseline_id"] == "internal-reference-v1"
    assert "overall_recommendation" in comparison
    assert "ranking" in comparison
    assert "prior_attempts" in comparison
    assert "Decision Report" in report
    assert "Scientific assessment" in report
    assert "Overall recommendation" in report
    assert "Attempt ranking" in report
    assert Path(run_payload["evaluation_path"]).exists()
    assert Path(run_payload["ranking_path"]).exists()


def test_partial_run_marks_attempt_partial(tmp_path: Path):
    _bootstrap_baseline(tmp_path)
    engine = InnovatorEngine(root=tmp_path)
    engine.opencode.executable = "definitely-missing-opencode"
    result = engine.submit("Invent a multi-timescale residual mixer with originality constraints.")
    engine.run(result.idea_id, phase="smoke")
    status = engine.status(result.idea_id)
    assert status["attempts"][0]["state"] == "partial"


def test_compare_includes_prior_attempt_summary_after_multiple_attempts(tmp_path: Path):
    _bootstrap_baseline(tmp_path)
    engine = InnovatorEngine(root=tmp_path)
    engine.opencode.executable = "definitely-missing-opencode"
    result = engine.submit("Invent a recurrent token mixer with recoverable routing dynamics.")
    first = engine.run(result.idea_id, phase="smoke")
    second = engine.run(result.idea_id, phase="smoke")
    comparison = engine.compare(result.idea_id)

    assert Path(first["ranking_path"]).exists()
    assert Path(second["ranking_path"]).exists()
    assert comparison["prior_attempts"]["attempt_ids"]
    assert comparison["ranking"]["prior_attempt_comparison_summary"]


def test_phase_skill_artifact_tracks_injected_and_skipped_skills(tmp_path: Path):
    _bootstrap_baseline(tmp_path)
    engine = InnovatorEngine(root=tmp_path)
    engine.opencode.executable = "definitely-missing-opencode"
    result = engine.submit("Invent a planner-heavy architecture exploration workflow.")
    engine.run(result.idea_id, phase="smoke")
    skill_payload = json.loads(
        (tmp_path / "ideas" / result.idea_id / "runs" / "attempt-0001" / "smoke" / "skills.json").read_text(encoding="utf-8")
    )
    planner_payload = skill_payload["roles"]["planner"]
    assert "active_skills" in planner_payload
    assert "injected_skills" in planner_payload
    assert "skipped_skills" in planner_payload
    assert any(item["name"] == "find-skills" for item in planner_payload["active_skills"])


def test_run_all_stops_after_failed_preflight(tmp_path: Path):
    _bootstrap_baseline(tmp_path)
    engine = InnovatorEngine(root=tmp_path)
    engine.opencode.executable = "definitely-missing-opencode"
    result = engine.submit("Invent a recurrent retrieval decoder with cache-aware routing and explicit memory.")
    idea_dir = tmp_path / "ideas" / result.idea_id
    smoke_config_path = idea_dir / "config" / "smoke.json"
    smoke_config = json.loads(smoke_config_path.read_text(encoding="utf-8"))
    smoke_config["max_retries_visible"] = 0
    smoke_config_path.write_text(json.dumps(smoke_config), encoding="utf-8")
    model_path = idea_dir / "package" / "modeling" / "model.py"
    model_path.write_text(
        model_path.read_text(encoding="utf-8").replace(
            '        output = {"logits": logits}',
            '        output = {"not_logits": logits}',
        ),
        encoding="utf-8",
    )

    payload = engine.run(result.idea_id, phase="all")
    status = engine.status(result.idea_id)
    attempt = status["attempts"][0]

    assert [phase["phase"] for phase in payload["phases"]] == ["smoke"]
    assert payload["phases"][0]["status"] == "failed"
    assert attempt["state"] == "failed"
    assert "small" not in attempt["phases"]
    assert (tmp_path / "ideas" / result.idea_id / "runs" / "attempt-0001" / "smoke" / "preflight-report.json").exists()


def test_structured_planner_and_reviewer_artifacts_are_persisted(tmp_path: Path, monkeypatch):
    _bootstrap_baseline(tmp_path)
    engine = InnovatorEngine(root=tmp_path)
    engine.opencode.executable = "definitely-missing-opencode"
    result = engine.submit("Invent a planner-heavy architecture exploration workflow.")

    def _invoke_structured(**kwargs):
        role = kwargs["role"]
        if role == "planner":
            payload = {
                "phase_summary": "Plan the smoke execution.",
                "focus_areas": ["runtime wiring"],
                "risk_flags": ["budget drift"],
                "success_criteria": ["phase completes"],
                "recommended_next_action": "execute_phase",
            }
        else:
            payload = {
                "recommendation": "continue_with_caution",
                "summary": "The run is usable for follow-up work.",
                "blocking_concerns": [],
                "continuation_criteria": ["keep tracking lineage"],
                "cited_artifacts": [],
            }
        return {
            "status": "completed",
            "command_preview": {"available": True, "cwd": str(tmp_path), "command": ["opencode", "run", role]},
            "raw_stdout": json.dumps(payload),
            "raw_stderr": "",
            "parsed_payload": payload,
            "parse_status": "valid",
            "validation_errors": [],
        }

    monkeypatch.setattr(OpenCodeAdapter, "invoke_structured", lambda self, **kwargs: _invoke_structured(**kwargs))

    payload = engine.run(result.idea_id, phase="smoke")
    phase_dir = tmp_path / "ideas" / result.idea_id / "runs" / payload["attempt_id"] / "smoke"
    planner_request = json.loads((phase_dir / "agents" / "planner-request.json").read_text(encoding="utf-8"))
    planner_response = json.loads((phase_dir / "agents" / "planner-response.json").read_text(encoding="utf-8"))
    reviewer_request = json.loads((phase_dir / "agents" / "reviewer-request.json").read_text(encoding="utf-8"))
    reviewer_response = json.loads((phase_dir / "agents" / "reviewer-response.json").read_text(encoding="utf-8"))

    planner_kinds = {artifact["kind"] for artifact in planner_request["context_artifacts"]}
    reviewer_kinds = {artifact["kind"] for artifact in reviewer_request["context_artifacts"]}

    assert {"design_ir", "phase_config", "resource_plan", "dataset_plan", "resolved_config"} <= planner_kinds
    assert {"current_phase_lineage", "prompt_payload", "skills_payload"} <= reviewer_kinds
    assert planner_response["parse_status"] == "valid"
    assert reviewer_response["parse_status"] == "valid"
    assert any(path.endswith("agents/planner-request.json") for path in payload["phases"][0]["artifacts_produced"])
    assert any(path.endswith("agents/reviewer-response.json") for path in payload["phases"][0]["artifacts_produced"])


def test_invalid_structured_agent_output_does_not_block_execution(tmp_path: Path, monkeypatch):
    _bootstrap_baseline(tmp_path)
    engine = InnovatorEngine(root=tmp_path)
    engine.opencode.executable = "definitely-missing-opencode"
    result = engine.submit("Invent a planner-heavy architecture exploration workflow.")

    def _invoke_structured(**kwargs):
        role = kwargs["role"]
        payload = {"phase_summary": "missing fields"} if role == "planner" else {"summary": "missing fields"}
        return {
            "status": "completed",
            "command_preview": {"available": True, "cwd": str(tmp_path), "command": ["opencode", "run", role]},
            "raw_stdout": json.dumps(payload),
            "raw_stderr": "",
            "parsed_payload": payload,
            "parse_status": "valid",
            "validation_errors": [],
        }

    monkeypatch.setattr(OpenCodeAdapter, "invoke_structured", lambda self, **kwargs: _invoke_structured(**kwargs))

    payload = engine.run(result.idea_id, phase="smoke")
    phase_dir = tmp_path / "ideas" / result.idea_id / "runs" / payload["attempt_id"] / "smoke"
    planner_response = json.loads((phase_dir / "agents" / "planner-response.json").read_text(encoding="utf-8"))
    reviewer_response = json.loads((phase_dir / "agents" / "reviewer-response.json").read_text(encoding="utf-8"))

    assert payload["phases"][0]["status"] == "passed"
    assert planner_response["parse_status"] == "invalid_schema"
    assert reviewer_response["parse_status"] == "invalid_schema"


def test_rejected_phase_persists_structured_agent_artifacts_and_lineage_ref_hash(tmp_path: Path):
    _bootstrap_baseline(tmp_path)
    engine = InnovatorEngine(root=tmp_path)
    engine.opencode.executable = "definitely-missing-opencode"
    result = engine.submit("Invent a recurrent retrieval decoder with cache-aware routing and explicit memory.")
    idea_dir = tmp_path / "ideas" / result.idea_id
    environment = {
        "torch_available": False,
        "accelerator_backend": "cpu",
        "rocm_available": False,
        "device_count": 0,
        "gpu_names": [],
        "vram_bytes_per_device": [],
        "cpu_count": 8,
        "system_ram_bytes": 8_000_000_000,
        "free_disk_bytes": 400_000_000_000,
        "default_dtype": "float32",
        "torch_version": None,
        "platform_system": "Darwin",
        "platform_machine": "arm64",
        "message": "test environment",
    }
    (idea_dir / "environment.json").write_text(json.dumps(environment), encoding="utf-8")

    payload = engine.run(result.idea_id, phase="full")
    phase_dir = idea_dir / "runs" / payload["attempt_id"] / "full"
    reviewer_request = json.loads((phase_dir / "agents" / "reviewer-request.json").read_text(encoding="utf-8"))
    lineage_path = phase_dir / "lineage-manifest.json"
    lineage_ref = next(
        artifact for artifact in reviewer_request["context_artifacts"] if artifact["kind"] == "current_phase_lineage"
    )

    assert (phase_dir / "agents" / "planner-request.json").exists()
    assert (phase_dir / "agents" / "reviewer-runtime.json").exists()
    assert lineage_ref["sha256"] == hash_file(lineage_path).sha256


def test_planner_request_includes_prior_phase_lineage_for_later_phase(
    tmp_path: Path,
    bootstrap_baseline,
    assert_phase_agent_artifacts,
):
    bootstrap_baseline(tmp_path)
    engine = InnovatorEngine(root=tmp_path)
    engine.opencode.executable = "definitely-missing-opencode"
    submit = engine.submit("Invent a recurrent retrieval decoder with cache-aware routing and explicit memory.")

    payload = engine.run(submit.idea_id, phase="all")
    small_phase_dir = tmp_path / "ideas" / submit.idea_id / "runs" / payload["attempt_id"] / "small"
    artifacts = assert_phase_agent_artifacts(small_phase_dir, payload["phases"][1])
    planner_request = artifacts["planner"]["request"]
    planner_artifacts = planner_request["context_artifacts"]
    planner_kinds = {artifact["kind"] for artifact in planner_artifacts}
    prior_phase_refs = [artifact for artifact in planner_artifacts if artifact["kind"] == "prior_phase_lineage"]

    assert {"design_ir", "phase_config", "resource_plan", "dataset_plan", "resolved_config"} <= planner_kinds
    assert prior_phase_refs
    assert prior_phase_refs[0]["path"].endswith("runs/attempt-0001/smoke/lineage-manifest.json")


def test_planner_request_includes_prior_attempt_phase_lineage_on_rerun(
    tmp_path: Path,
    bootstrap_baseline,
    assert_phase_agent_artifacts,
):
    bootstrap_baseline(tmp_path)
    engine = InnovatorEngine(root=tmp_path)
    engine.opencode.executable = "definitely-missing-opencode"
    submit = engine.submit("Invent a recurrent retrieval decoder with cache-aware routing and explicit memory.")

    engine.run(submit.idea_id, phase="smoke")
    second = engine.run(submit.idea_id, phase="smoke")
    phase_dir = tmp_path / "ideas" / submit.idea_id / "runs" / second["attempt_id"] / "smoke"
    artifacts = assert_phase_agent_artifacts(phase_dir, second["phases"][0])
    planner_request = artifacts["planner"]["request"]
    prior_attempt_refs = [
        artifact for artifact in planner_request["context_artifacts"] if artifact["kind"] == "prior_attempt_phase_lineage"
    ]

    assert prior_attempt_refs
    assert prior_attempt_refs[0]["path"].endswith("runs/attempt-0001/smoke/lineage-manifest.json")


def test_reviewer_request_omits_prior_attempt_comparison_artifacts_for_first_run(
    tmp_path: Path,
    bootstrap_baseline,
    assert_phase_agent_artifacts,
):
    bootstrap_baseline(tmp_path)
    engine = InnovatorEngine(root=tmp_path)
    engine.opencode.executable = "definitely-missing-opencode"
    submit = engine.submit("Invent a planner-heavy architecture exploration workflow.")

    payload = engine.run(submit.idea_id, phase="smoke")
    phase_dir = tmp_path / "ideas" / submit.idea_id / "runs" / payload["attempt_id"] / "smoke"
    artifacts = assert_phase_agent_artifacts(phase_dir, payload["phases"][0])
    reviewer_request = artifacts["reviewer"]["request"]
    reviewer_kinds = {artifact["kind"] for artifact in reviewer_request["context_artifacts"]}

    assert reviewer_kinds == {"current_phase_lineage", "prompt_payload", "skills_payload"}


def test_reviewer_request_includes_prior_attempt_comparison_artifacts_on_rerun(
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
    reviewer_request = artifacts["reviewer"]["request"]
    reviewer_artifacts = {artifact["kind"]: artifact for artifact in reviewer_request["context_artifacts"]}

    assert reviewer_artifacts["prior_attempt_ranking"]["path"].endswith(f"reports/{first['attempt_id']}-ranking.json")
    assert reviewer_artifacts["prior_attempt_evaluation"]["path"].endswith(f"reports/{first['attempt_id']}-evaluation.json")


@pytest.mark.parametrize(
    ("case_name", "planner_config", "reviewer_config", "expected_parse_status", "expected_runtime_status"),
    [
        ("dry_run", None, None, "dry_run", "dry-run"),
        (
            "invalid_json",
            {
                "parsed_payload": None,
                "parse_status": "invalid_json",
                "raw_stdout": "not-json",
                "validation_errors": ["OpenCode did not return a valid JSON object payload."],
            },
            {
                "parsed_payload": None,
                "parse_status": "invalid_json",
                "raw_stdout": "not-json",
                "validation_errors": ["OpenCode did not return a valid JSON object payload."],
            },
            "invalid_json",
            "completed",
        ),
        (
            "invalid_schema",
            {"parsed_payload": {"phase_summary": "missing fields"}, "parse_status": "valid"},
            {"parsed_payload": {"summary": "missing fields"}, "parse_status": "valid"},
            "invalid_schema",
            "completed",
        ),
        (
            "runtime_failed",
            {
                "status": "failed",
                "parsed_payload": None,
                "parse_status": "runtime_failed",
                "raw_stderr": "boom",
                "validation_errors": ["OpenCode returned non-zero exit status 7."],
            },
            {
                "status": "failed",
                "parsed_payload": None,
                "parse_status": "runtime_failed",
                "raw_stderr": "boom",
                "validation_errors": ["OpenCode returned non-zero exit status 7."],
            },
            "runtime_failed",
            "failed",
        ),
    ],
)
def test_structured_agent_parse_status_matrix_persists_without_blocking_execution(
    tmp_path: Path,
    bootstrap_baseline,
    structured_agent_stub,
    assert_phase_agent_artifacts,
    case_name: str,
    planner_config: dict | None,
    reviewer_config: dict | None,
    expected_parse_status: str,
    expected_runtime_status: str,
):
    bootstrap_baseline(tmp_path)
    engine = InnovatorEngine(root=tmp_path)
    engine.opencode.executable = "definitely-missing-opencode"
    submit = engine.submit(f"Invent a planner-heavy architecture exploration workflow for {case_name}.")

    if case_name != "dry_run":
        structured_agent_stub(planner=planner_config, reviewer=reviewer_config)

    payload = engine.run(submit.idea_id, phase="smoke")
    phase_dir = tmp_path / "ideas" / submit.idea_id / "runs" / payload["attempt_id"] / "smoke"
    artifacts = assert_phase_agent_artifacts(phase_dir, payload["phases"][0])

    assert payload["phases"][0]["status"] == "passed"
    for role in ("planner", "reviewer"):
        response_payload = artifacts[role]["response"]
        runtime_payload = artifacts[role]["runtime"]
        assert response_payload["parse_status"] == expected_parse_status
        assert runtime_payload["parse_status"] == expected_parse_status
        assert runtime_payload["status"] == expected_runtime_status
        if expected_parse_status == "invalid_schema":
            assert response_payload["raw_payload"] is not None
            assert runtime_payload["parsed_payload"] is not None
            assert response_payload["validation_errors"]
        else:
            assert response_payload["raw_payload"] is None
