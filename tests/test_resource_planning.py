import json
from pathlib import Path

from auto_llm_innovator.design_ir import DesignIR
from auto_llm_innovator.env import EnvironmentReport
from auto_llm_innovator.evaluation import EvaluationResult, PhaseEvaluationSummary, load_baseline_definition
from auto_llm_innovator.orchestration import InnovatorEngine
from auto_llm_innovator.planning import apply_phase_resource_plan, plan_phase_resources
from auto_llm_innovator.runtime import compile_runtime_phase_config
from auto_llm_innovator.tracking.ranking import AttemptRankingResult


def _bootstrap_baseline(
    root: Path,
    *,
    hardware_assumptions: dict | None = None,
    token_budget_assumptions: dict | None = None,
) -> Path:
    baseline_dir = root / "baselines" / "internal_reference"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    path = baseline_dir / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "baseline_id": "internal-reference-v1",
                "family": "internal_reference",
                "label": "Internal Reference",
                "metric_targets": [
                    {"phase": "smoke", "metric_name": "loss", "target_value": 6.0},
                    {"phase": "small", "metric_name": "val_loss", "target_value": 4.2},
                    {"phase": "full", "metric_name": "val_loss", "target_value": 3.7},
                ],
                "hardware_assumptions": hardware_assumptions or {},
                "token_budget_assumptions": token_budget_assumptions or {"smoke": 50_000, "small": 3_000_000, "full": 12_000_000},
            }
        ),
        encoding="utf-8",
    )
    return path


def _environment(
    *,
    backend: str,
    vram_bytes_per_device: list[int] | None = None,
    system_ram_bytes: int = 64_000_000_000,
) -> EnvironmentReport:
    return EnvironmentReport(
        torch_available=backend != "none",
        accelerator_backend=backend,
        rocm_available=backend == "rocm",
        device_count=len(vram_bytes_per_device or []),
        gpu_names=[f"{backend}-gpu-{index}" for index in range(len(vram_bytes_per_device or []))],
        vram_bytes_per_device=list(vram_bytes_per_device or []),
        cpu_count=8,
        system_ram_bytes=system_ram_bytes,
        free_disk_bytes=500_000_000_000,
        default_dtype="float32",
        torch_version="2.8.0" if backend != "none" else None,
        platform_system="Darwin",
        platform_machine="arm64",
        message="test environment",
    )


def _ranking(*, label: str, best_so_far: bool, summary: str = "summary") -> AttemptRankingResult:
    return AttemptRankingResult(
        idea_id="idea-0001",
        attempt_id="attempt-0001",
        baseline_id="internal-reference-v1",
        promotion_score=100.0,
        rank_label=label,
        baseline_comparison_summary=summary,
        prior_attempt_comparison_summary=summary,
        best_so_far=best_so_far,
    )


def _evaluation(overall_recommendation: str) -> EvaluationResult:
    return EvaluationResult(
        idea_id="idea-0001",
        attempt_id="attempt-0001",
        baseline_id="internal-reference-v1",
        overall_recommendation=overall_recommendation,
        overall_summary="summary",
        phase_summaries=[
            PhaseEvaluationSummary(
                phase="smoke",
                phase_status="passed",
                recommendation="promote",
                summary="summary",
            )
        ],
        comparison_totals={"promote_count": 1},
    )


def _submit_idea(tmp_path: Path) -> tuple[InnovatorEngine, Path, DesignIR]:
    baseline_path = _bootstrap_baseline(tmp_path)
    assert baseline_path.exists()
    engine = InnovatorEngine(root=tmp_path)
    submit = engine.submit("Invent a recurrent memory-routed decoder with cache-aware state.")
    idea_dir = tmp_path / "ideas" / submit.idea_id
    design_ir = DesignIR.from_dict(json.loads((idea_dir / "design_ir.json").read_text(encoding="utf-8")))
    return engine, idea_dir, design_ir


def test_gpu_environment_admits_smoke_and_small(tmp_path: Path):
    _engine, idea_dir, design_ir = _submit_idea(tmp_path)
    baseline = load_baseline_definition(tmp_path / "baselines" / "internal_reference" / "manifest.json")
    environment = _environment(backend="cuda", vram_bytes_per_device=[24_000_000_000])

    for phase in ("smoke", "small"):
        phase_config = json.loads((idea_dir / "config" / f"{phase}.json").read_text(encoding="utf-8"))
        plan = plan_phase_resources(
            design_ir=design_ir,
            phase_config=phase_config,
            environment=environment,
            baseline=baseline,
        )
        assert plan.admission_status == "admit"


def test_constrained_environment_downscales_batch_before_sequence_length(tmp_path: Path):
    _engine, idea_dir, design_ir = _submit_idea(tmp_path)
    baseline = load_baseline_definition(tmp_path / "baselines" / "internal_reference" / "manifest.json")
    environment = _environment(backend="cuda", vram_bytes_per_device=[14_000_000_000])
    phase_config = json.loads((idea_dir / "config" / "full.json").read_text(encoding="utf-8"))

    plan = plan_phase_resources(
        design_ir=design_ir,
        phase_config=phase_config,
        environment=environment,
        baseline=baseline,
    )

    assert plan.admission_status == "downscale"
    assert plan.adjustments
    assert plan.adjustments[0].field_name == "runtime.batch_size"


def test_no_accelerator_rejects_infeasible_full(tmp_path: Path):
    _engine, idea_dir, design_ir = _submit_idea(tmp_path)
    baseline = load_baseline_definition(tmp_path / "baselines" / "internal_reference" / "manifest.json")
    environment = _environment(backend="cpu", system_ram_bytes=16_000_000_000)
    phase_config = json.loads((idea_dir / "config" / "full.json").read_text(encoding="utf-8"))

    plan = plan_phase_resources(
        design_ir=design_ir,
        phase_config=phase_config,
        environment=environment,
        baseline=baseline,
    )

    assert plan.admission_status == "reject"


def test_baseline_hardware_assumption_adds_warning_without_hard_failure(tmp_path: Path):
    _bootstrap_baseline(tmp_path, hardware_assumptions={"device": "rocm"})
    engine = InnovatorEngine(root=tmp_path)
    submit = engine.submit("Invent a recurrent memory-routed decoder with cache-aware state.")
    idea_dir = tmp_path / "ideas" / submit.idea_id
    design_ir = DesignIR.from_dict(json.loads((idea_dir / "design_ir.json").read_text(encoding="utf-8")))
    baseline = load_baseline_definition(tmp_path / "baselines" / "internal_reference" / "manifest.json")
    environment = _environment(backend="cuda", vram_bytes_per_device=[24_000_000_000])
    phase_config = json.loads((idea_dir / "config" / "smoke.json").read_text(encoding="utf-8"))

    plan = plan_phase_resources(
        design_ir=design_ir,
        phase_config=phase_config,
        environment=environment,
        baseline=baseline,
    )

    assert plan.admission_status == "admit"
    assert any("Baseline hardware assumption" in warning for warning in plan.warnings)


def test_leading_attempt_gets_more_budget_than_warning_only_attempt(tmp_path: Path):
    _engine, idea_dir, design_ir = _submit_idea(tmp_path)
    baseline = load_baseline_definition(tmp_path / "baselines" / "internal_reference" / "manifest.json")
    environment = _environment(backend="cuda", vram_bytes_per_device=[20_000_000_000])
    phase_config = json.loads((idea_dir / "config" / "full.json").read_text(encoding="utf-8"))

    leading_plan = plan_phase_resources(
        design_ir=design_ir,
        phase_config=phase_config,
        environment=environment,
        baseline=baseline,
        ranking_result=_ranking(label="leading", best_so_far=True),
        evaluation_result=_evaluation("promote"),
    )
    caution_plan = plan_phase_resources(
        design_ir=design_ir,
        phase_config=phase_config,
        environment=environment,
        baseline=baseline,
        ranking_result=_ranking(label="caution", best_so_far=False),
        evaluation_result=_evaluation("rerun_with_more_budget"),
    )

    assert leading_plan.resolved_target_parameters >= caution_plan.resolved_target_parameters
    assert leading_plan.resolved_runtime_settings.batch_size >= caution_plan.resolved_runtime_settings.batch_size


def test_noncompetitive_attempt_does_not_auto_escalate(tmp_path: Path):
    _engine, idea_dir, design_ir = _submit_idea(tmp_path)
    baseline = load_baseline_definition(tmp_path / "baselines" / "internal_reference" / "manifest.json")
    environment = _environment(backend="cuda", vram_bytes_per_device=[18_000_000_000])
    phase_config = json.loads((idea_dir / "config" / "full.json").read_text(encoding="utf-8"))

    plan = plan_phase_resources(
        design_ir=design_ir,
        phase_config=phase_config,
        environment=environment,
        baseline=baseline,
        ranking_result=_ranking(label="noncompetitive", best_so_far=False),
        evaluation_result=_evaluation("stop"),
    )

    assert plan.admission_status in {"downscale", "reject"}
    assert plan.resolved_target_parameters <= int(phase_config["target_parameters"])


def test_resource_plan_is_persisted_and_attached_to_phase_artifacts(tmp_path: Path):
    engine, idea_dir, _design_ir = _submit_idea(tmp_path)
    environment = _environment(backend="cuda", vram_bytes_per_device=[24_000_000_000])
    (idea_dir / "environment.json").write_text(json.dumps(environment.to_dict()), encoding="utf-8")

    payload = engine.run(idea_dir.name, phase="smoke")
    phase_payload = payload["phases"][0]

    assert (idea_dir / "runs" / payload["attempt_id"] / "smoke" / "resource-plan.json").exists()
    assert any(path.endswith("resource-plan.json") for path in phase_payload["artifacts_produced"])


def test_rejected_admission_fails_before_preflight(tmp_path: Path):
    engine, idea_dir, _design_ir = _submit_idea(tmp_path)
    environment = _environment(backend="cpu", system_ram_bytes=8_000_000_000)
    (idea_dir / "environment.json").write_text(json.dumps(environment.to_dict()), encoding="utf-8")

    payload = engine.run(idea_dir.name, phase="full")
    phase_payload = payload["phases"][0]
    phase_dir = idea_dir / "runs" / payload["attempt_id"] / "full"

    assert phase_payload["status"] == "failed"
    assert phase_payload["consumed_budget"]["stop_reason"] == "resource_admission_failed"
    assert not (phase_dir / "preflight-report.json").exists()
    assert (phase_dir / "resource-plan.json").exists()


def test_resolved_runtime_settings_compile_after_downscaling(tmp_path: Path):
    _engine, idea_dir, design_ir = _submit_idea(tmp_path)
    baseline = load_baseline_definition(tmp_path / "baselines" / "internal_reference" / "manifest.json")
    environment = _environment(backend="cuda", vram_bytes_per_device=[14_000_000_000])
    phase_config = json.loads((idea_dir / "config" / "full.json").read_text(encoding="utf-8"))

    plan = plan_phase_resources(
        design_ir=design_ir,
        phase_config=phase_config,
        environment=environment,
        baseline=baseline,
    )
    resolved = apply_phase_resource_plan(phase_config, plan)
    runtime_config = compile_runtime_phase_config(design_ir, resolved, attempt_id="attempt-0001", phase="full")

    assert runtime_config.settings.batch_size == resolved["runtime"]["batch_size"]
    assert runtime_config.resource_plan["admission_status"] == plan.admission_status
