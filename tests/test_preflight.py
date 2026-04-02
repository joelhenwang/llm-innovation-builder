import json
from pathlib import Path

from auto_llm_innovator.datasets import dataset_plan_for_phase
from auto_llm_innovator.design_ir import DesignIR, compile_design_ir, project_idea_spec
from auto_llm_innovator.filesystem import ensure_dir, read_json
from auto_llm_innovator.generation import generate_idea_package
from auto_llm_innovator.handoff import load_research_idea_bundle
from auto_llm_innovator.idea_spec import review_originality
from auto_llm_innovator.runtime import compile_runtime_phase_config, default_runtime_settings_for_phase
from auto_llm_innovator.training import execute_phase
from auto_llm_innovator.validation import run_preflight, write_preflight_report


def _write_phase_config(idea_dir: Path, phase: str, target_parameters: int = 600_000_000) -> Path:
    config_dir = ensure_dir(idea_dir / "config")
    config_path = config_dir / f"{phase}.json"
    config_path.write_text(
        json.dumps(
            {
                "phase": phase,
                "target_parameters": target_parameters,
                "prefer_rocm": True,
                "dataset": dataset_plan_for_phase(phase),
                "runtime": default_runtime_settings_for_phase(phase).to_dict(),
                "novelty_claims": ["preflight coverage"],
            }
        ),
        encoding="utf-8",
    )
    return config_path


def _build_generated_idea(tmp_path: Path, *, idea_id: str = "idea-0001", brief: str | None = None) -> Path:
    idea_dir = ensure_dir(tmp_path / "ideas" / idea_id)
    bundle = load_research_idea_bundle(
        raw_brief=brief or "Invent a recurrent retrieval decoder with cache-aware routing and explicit memory."
    )
    design_ir = compile_design_ir(bundle, idea_id=idea_id)
    spec = project_idea_spec(design_ir, bundle)
    review_originality(spec)

    (idea_dir / "design_ir.json").write_text(json.dumps(design_ir.to_dict()), encoding="utf-8")
    (idea_dir / "idea_spec.json").write_text(json.dumps(spec.to_dict()), encoding="utf-8")
    generate_idea_package(idea_dir, spec, design_ir)
    _write_phase_config(idea_dir, "smoke")
    return idea_dir


def _runtime_config(idea_dir: Path, attempt_id: str = "attempt-0001", phase: str = "smoke"):
    design_ir = DesignIR.from_dict(read_json(idea_dir / "design_ir.json"))
    phase_config = read_json(idea_dir / "config" / f"{phase}.json")
    return compile_runtime_phase_config(design_ir, phase_config, attempt_id=attempt_id, phase=phase)


def test_preflight_happy_path_writes_machine_readable_report(tmp_path: Path):
    idea_dir = _build_generated_idea(tmp_path)
    run_dir = ensure_dir(idea_dir / "runs" / "attempt-0001" / "smoke")

    result = run_preflight(
        idea_dir=idea_dir,
        run_dir=run_dir,
        runtime_config=_runtime_config(idea_dir),
        attempt_id="attempt-0001",
    )
    report_path = write_preflight_report(run_dir, result, retry_attempted=False, retry_outcome="not_requested")
    report = read_json(report_path)

    assert result.status == "passed"
    assert report["status"] == "passed"
    assert [check["name"] for check in report["checks"]] == [
        "package_import",
        "plugin_contract",
        "model_instantiation",
        "forward_pass",
        "loss_sanity",
        "train_step_sanity",
        "checkpoint_roundtrip",
        "eval_hook_sanity",
    ]
    assert (run_dir / "preflight-checkpoint.json").exists()
    assert (run_dir / "preflight" / "evaluation-report.json").exists()


def test_preflight_fails_when_generated_file_is_missing(tmp_path: Path):
    idea_dir = _build_generated_idea(tmp_path, idea_id="idea-0002")
    run_dir = ensure_dir(idea_dir / "runs" / "attempt-0001" / "smoke")
    (idea_dir / "package" / "plugin.py").unlink()

    result = run_preflight(
        idea_dir=idea_dir,
        run_dir=run_dir,
        runtime_config=_runtime_config(idea_dir),
        attempt_id="attempt-0001",
    )

    assert result.status == "failed"
    assert "missing_generated_file" in result.failure_categories
    assert "package/plugin.py" in result.failing_files
    assert result.retryable is True


def test_preflight_fails_on_state_semantic_plugin_mismatch(tmp_path: Path):
    idea_dir = _build_generated_idea(tmp_path, idea_id="idea-0003")
    run_dir = ensure_dir(idea_dir / "runs" / "attempt-0001" / "smoke")
    plugin_path = idea_dir / "package" / "plugin.py"
    plugin_source = plugin_path.read_text(encoding="utf-8").replace(
        "'recurrent_state': True",
        "'recurrent_state': False",
    )
    plugin_path.write_text(plugin_source, encoding="utf-8")

    result = run_preflight(
        idea_dir=idea_dir,
        run_dir=run_dir,
        runtime_config=_runtime_config(idea_dir),
        attempt_id="attempt-0001",
    )

    assert result.status == "failed"
    assert "plugin_contract_failure" in result.failure_categories
    assert "package.plugin" in result.failing_modules


def test_execute_phase_skips_train_entrypoint_and_recovers_with_one_regeneration(tmp_path: Path):
    idea_dir = _build_generated_idea(tmp_path, idea_id="idea-0004")
    run_dir = idea_dir / "runs" / "attempt-0001" / "smoke"
    (idea_dir / "train.py").write_text("this is not valid python\n", encoding="utf-8")
    (idea_dir / "package" / "plugin.py").unlink()

    result = execute_phase(idea_dir=idea_dir, attempt_id="attempt-0001", phase="smoke", run_dir=run_dir)
    preflight_report = read_json(run_dir / "preflight-report.json")

    assert result.status == "passed"
    assert any("regenerated once and passed preflight on retry" in note for note in result.reviewer_notes)
    assert preflight_report["retry_attempted"] is True
    assert preflight_report["retry_outcome"] == "recovered"
    assert (idea_dir / "package" / "plugin.py").exists()


def test_execute_phase_returns_failed_result_when_preflight_cannot_recover(tmp_path: Path):
    idea_dir = _build_generated_idea(tmp_path, idea_id="idea-0005")
    run_dir = idea_dir / "runs" / "attempt-0001" / "smoke"
    (idea_dir / "package" / "plugin.py").unlink()
    (idea_dir / "idea_spec.json").write_text("{not valid json", encoding="utf-8")

    result = execute_phase(idea_dir=idea_dir, attempt_id="attempt-0001", phase="smoke", run_dir=run_dir)
    preflight_report = read_json(run_dir / "preflight-report.json")

    assert result.status == "failed"
    assert result.next_action_recommendation == "repair_preflight"
    assert preflight_report["retry_attempted"] is True
    assert preflight_report["retry_outcome"] == "failed"
    assert any("Deterministic package regeneration failed:" in signal for signal in result.failure_signals)
