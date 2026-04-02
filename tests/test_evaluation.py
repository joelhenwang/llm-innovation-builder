import json
from pathlib import Path

from auto_llm_innovator.datasets import dataset_plan_for_phase
from auto_llm_innovator.design_ir import compile_design_ir, project_idea_spec
from auto_llm_innovator.evaluation import build_evaluation_result, compare_against_baseline
from auto_llm_innovator.generation import generate_idea_package
from auto_llm_innovator.handoff import load_research_idea_bundle
from auto_llm_innovator.idea_spec import review_originality
from auto_llm_innovator.runtime import default_runtime_settings_for_phase
from auto_llm_innovator.training import execute_phase


def _bootstrap_baseline(root: Path, *, include_smoke_metric: bool = True) -> Path:
    baseline_dir = root / "baselines" / "internal_reference"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"small.val_loss": 4.2, "full.val_loss": 3.7}
    if include_smoke_metric:
        metrics["smoke.loss"] = 6.0
    path = baseline_dir / "manifest.json"
    path.write_text(json.dumps({"baseline_id": "internal-reference-v1", "reference_metrics": metrics}), encoding="utf-8")
    return path


def _write_phase_config(idea_dir: Path, phase: str, *, max_retries_visible: int = 1) -> None:
    config_dir = idea_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / f"{phase}.json").write_text(
        json.dumps(
            {
                "phase": phase,
                "target_parameters": 600_000_000,
                "prefer_rocm": True,
                "dataset": dataset_plan_for_phase(phase),
                "runtime": default_runtime_settings_for_phase(phase).to_dict(),
                "novelty_claims": ["evaluation coverage"],
                "max_retries_visible": max_retries_visible,
            }
        ),
        encoding="utf-8",
    )


def _build_generated_idea(tmp_path: Path, *, idea_id: str) -> Path:
    idea_dir = tmp_path / "ideas" / idea_id
    idea_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_research_idea_bundle(
        raw_brief="Invent a recurrent retrieval decoder with cache-aware routing and explicit memory."
    )
    design_ir = compile_design_ir(bundle, idea_id=idea_id)
    spec = project_idea_spec(design_ir, bundle)
    review_originality(spec)
    (idea_dir / "design_ir.json").write_text(json.dumps(design_ir.to_dict()), encoding="utf-8")
    (idea_dir / "idea_spec.json").write_text(json.dumps(spec.to_dict()), encoding="utf-8")
    generate_idea_package(idea_dir, spec, design_ir)
    _write_phase_config(idea_dir, "smoke", max_retries_visible=2)
    return idea_dir


def _evaluate_single_phase(tmp_path: Path, *, idea_id: str, mutate=None, include_smoke_metric: bool = True):
    idea_dir = _build_generated_idea(tmp_path, idea_id=idea_id)
    if mutate is not None:
        mutate(idea_dir)
    result = execute_phase(
        idea_dir=idea_dir,
        attempt_id="attempt-0001",
        phase="smoke",
        run_dir=idea_dir / "runs" / "attempt-0001" / "smoke",
    )
    baseline_manifest = _bootstrap_baseline(tmp_path, include_smoke_metric=include_smoke_metric)
    evaluation_result = build_evaluation_result(
        idea_dir=idea_dir,
        attempt_id="attempt-0001",
        baseline_manifest=baseline_manifest,
        results=[result],
    )
    return idea_dir, result, evaluation_result, baseline_manifest


def test_clean_pass_produces_promotable_evaluation(tmp_path: Path):
    _idea_dir, _result, evaluation_result, _baseline = _evaluate_single_phase(tmp_path, idea_id="idea-eval-0001")

    phase = evaluation_result.phase_summaries[0]
    assert phase.recommendation == "promote"
    assert phase.caution_flags == []
    assert evaluation_result.overall_recommendation == "promote"


def test_repaired_pass_is_downgraded_to_caution(tmp_path: Path):
    def mutate(idea_dir: Path) -> None:
        (idea_dir / "package" / "plugin.py").unlink()

    _idea_dir, result, evaluation_result, _baseline = _evaluate_single_phase(
        tmp_path,
        idea_id="idea-eval-0002",
        mutate=mutate,
    )

    phase = evaluation_result.phase_summaries[0]
    assert result.repair_attempted is True
    assert phase.recommendation == "continue_with_caution"
    assert "repaired_pass" in phase.caution_flags


def test_warning_only_result_becomes_rerun_with_more_budget(tmp_path: Path):
    def mutate(idea_dir: Path) -> None:
        config_path = idea_dir / "config" / "smoke.json"
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        payload["runtime"]["max_wall_time_seconds"] = 0
        config_path.write_text(json.dumps(payload), encoding="utf-8")

    _idea_dir, result, evaluation_result, _baseline = _evaluate_single_phase(
        tmp_path,
        idea_id="idea-eval-0003",
        mutate=mutate,
    )

    phase = evaluation_result.phase_summaries[0]
    assert result.status == "passed_with_warnings"
    assert phase.recommendation == "rerun_with_more_budget"
    assert "budget_limited" in phase.caution_flags


def test_failed_after_repairs_produces_stop_recommendation(tmp_path: Path):
    idea_dir = _build_generated_idea(tmp_path, idea_id="idea-eval-0004")
    run_dir = idea_dir / "runs" / "attempt-0001" / "smoke"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "smoke-summary.json").write_text(
        json.dumps(
            {
                "phase": "smoke",
                "attempt_id": "attempt-0001",
                "metrics": {},
                "stop_reason": "runtime_validation_failed",
                "runtime": {
                    "settings": default_runtime_settings_for_phase("smoke").to_dict(),
                },
            }
        ),
        encoding="utf-8",
    )
    repair_dir = run_dir / "repair"
    repair_dir.mkdir(parents=True, exist_ok=True)
    (repair_dir / "failure-classification.json").write_text(
        json.dumps({"category": "runtime_output_shape_failure", "source": "runtime", "repairable": True}),
        encoding="utf-8",
    )
    (repair_dir / "repair-history.json").write_text(
        json.dumps([{"attempt_index": 1, "strategy": "repair_runtime_outputs", "outcome": "applied"}]),
        encoding="utf-8",
    )
    result = execute_phase(
        idea_dir=idea_dir,
        attempt_id="attempt-0001",
        phase="smoke",
        run_dir=run_dir,
    )
    result.status = "failed"
    result.repair_attempted = True
    result.repair_count = 1
    result.repair_outcome = "failed_after_repairs"
    result.failure_classification = {"category": "runtime_output_shape_failure", "source": "runtime"}
    baseline_manifest = _bootstrap_baseline(tmp_path)
    evaluation_result = build_evaluation_result(
        idea_dir=idea_dir,
        attempt_id="attempt-0001",
        baseline_manifest=baseline_manifest,
        results=[result],
    )
    phase = evaluation_result.phase_summaries[0]
    assert result.status == "failed"
    assert phase.recommendation == "stop"
    assert evaluation_result.overall_recommendation == "stop"


def test_missing_baseline_metric_does_not_crash_and_metrics_are_copied(tmp_path: Path):
    _idea_dir, _result, evaluation_result, baseline_manifest = _evaluate_single_phase(
        tmp_path,
        idea_id="idea-eval-0005",
        include_smoke_metric=False,
    )

    phase = evaluation_result.phase_summaries[0]
    comparison = compare_against_baseline(baseline_manifest, [], evaluation_result=evaluation_result)
    assert phase.baseline_loss is None
    assert phase.evaluation_metrics
    first_task_metrics = next(iter(phase.evaluation_metrics.values()))
    assert first_task_metrics
    assert comparison["baseline_id"] == "internal-reference-v1"
