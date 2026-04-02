import json
from pathlib import Path

from auto_llm_innovator.datasets import (
    apply_dataset_plan,
    dataset_definitions,
    dataset_plan_for_phase,
    plan_dataset_for_phase,
)
from auto_llm_innovator.design_ir import DesignIR, compile_design_ir
from auto_llm_innovator.env import EnvironmentReport
from auto_llm_innovator.evaluation import load_baseline_definition
from auto_llm_innovator.handoff import load_research_idea_bundle
from auto_llm_innovator.orchestration import InnovatorEngine
from auto_llm_innovator.planning import apply_phase_resource_plan, plan_phase_resources
from auto_llm_innovator.runtime import compile_runtime_phase_config, default_runtime_settings_for_phase
from auto_llm_innovator.tracking.ranking import AttemptRankingResult


def _bootstrap_baseline(root: Path) -> Path:
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
                "token_budget_assumptions": {
                    "smoke": 50_000,
                    "small": 3_000_000,
                    "full": 12_000_000,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _environment(*, backend: str, vram_bytes_per_device: list[int] | None = None, system_ram_bytes: int = 64_000_000_000) -> EnvironmentReport:
    return EnvironmentReport(
        torch_available=backend != "none",
        accelerator_backend=backend,
        rocm_available=backend == "rocm",
        device_count=len(vram_bytes_per_device or []),
        gpu_names=[f"{backend}-gpu-{index}" for index in range(len(vram_bytes_per_device or []))],
        vram_bytes_per_device=list(vram_bytes_per_device or []),
        cpu_count=8,
        system_ram_bytes=system_ram_bytes,
        free_disk_bytes=400_000_000_000,
        default_dtype="float32",
        torch_version="2.8.0" if backend != "none" else None,
        platform_system="Darwin",
        platform_machine="arm64",
        message="test environment",
    )


def _phase_config(phase: str, *, target_parameters: int) -> dict:
    return {
        "phase": phase,
        "target_parameters": target_parameters,
        "prefer_rocm": True,
        "dataset": dataset_plan_for_phase(phase),
        "runtime": default_runtime_settings_for_phase(phase).to_dict(),
        "novelty_claims": ["dataset planning"],
    }


def _design_ir() -> DesignIR:
    bundle = load_research_idea_bundle(
        raw_brief="Invent a recurrent retrieval decoder with cache-aware routing and explicit memory."
    )
    return compile_design_ir(bundle, idea_id="idea-0001")


def _ranking(label: str, best_so_far: bool) -> AttemptRankingResult:
    return AttemptRankingResult(
        idea_id="idea-0001",
        attempt_id="attempt-0001",
        baseline_id="internal-reference-v1",
        promotion_score=100.0,
        rank_label=label,
        baseline_comparison_summary="summary",
        prior_attempt_comparison_summary="summary",
        best_so_far=best_so_far,
    )


def test_structured_registry_returns_stable_phase_defaults():
    definitions = dataset_definitions()

    assert {definition.dataset_id for definition in definitions} == {
        "synthetic-shapes",
        "small-curated-corpus",
        "production-like-corpus",
    }
    assert dataset_plan_for_phase("smoke")["dataset_name"] == "synthetic-shapes"
    assert dataset_plan_for_phase("small")["target_tokens"] == 5_000_000
    assert dataset_plan_for_phase("full")["description"]


def test_design_ir_training_plan_uses_registry_backed_defaults():
    design_ir = _design_ir()
    stages = {stage.stage: stage for stage in design_ir.training_plan}

    assert stages["smoke"].dataset_name == "synthetic-shapes"
    assert stages["small"].dataset_name == "small-curated-corpus"
    assert stages["full"].dataset_name == "production-like-corpus"


def test_admitted_phase_keeps_planned_dataset_family_and_tokens(tmp_path: Path):
    baseline_path = _bootstrap_baseline(tmp_path)
    baseline = load_baseline_definition(baseline_path)
    design_ir = _design_ir()
    phase_config = _phase_config("small", target_parameters=2_000_000_000)
    resource_plan = plan_phase_resources(
        design_ir=design_ir,
        phase_config=phase_config,
        environment=_environment(backend="cuda", vram_bytes_per_device=[24_000_000_000]),
        baseline=baseline,
    )
    resolved_phase_config = apply_phase_resource_plan(phase_config, resource_plan)

    dataset_plan = plan_dataset_for_phase(
        design_ir=design_ir,
        phase="small",
        phase_config=phase_config,
        resolved_phase_config=resolved_phase_config,
        resource_plan=resource_plan,
        baseline=baseline,
    )

    assert dataset_plan.admission_status == "admit"
    assert dataset_plan.dataset_id == "small-curated-corpus"
    assert dataset_plan.target_tokens == 5_000_000


def test_downscaled_phase_reduces_tokens_and_simplifies_slice(tmp_path: Path):
    baseline_path = _bootstrap_baseline(tmp_path)
    baseline = load_baseline_definition(baseline_path)
    design_ir = _design_ir()
    phase_config = _phase_config("full", target_parameters=design_ir.parameter_cap)
    resource_plan = plan_phase_resources(
        design_ir=design_ir,
        phase_config=phase_config,
        environment=_environment(backend="cuda", vram_bytes_per_device=[14_000_000_000]),
        baseline=baseline,
        ranking_result=_ranking("caution", False),
    )
    resolved_phase_config = apply_phase_resource_plan(phase_config, resource_plan)

    dataset_plan = plan_dataset_for_phase(
        design_ir=design_ir,
        phase="full",
        phase_config=phase_config,
        resolved_phase_config=resolved_phase_config,
        resource_plan=resource_plan,
        baseline=baseline,
    )

    assert resource_plan.admission_status == "downscale"
    assert dataset_plan.target_tokens < phase_config["dataset"]["target_tokens"]
    assert dataset_plan.dataset_slice in {"curated", "tiny"}


def test_rejected_phase_produces_report_only_dataset_plan(tmp_path: Path):
    baseline_path = _bootstrap_baseline(tmp_path)
    baseline = load_baseline_definition(baseline_path)
    design_ir = _design_ir()
    phase_config = _phase_config("full", target_parameters=design_ir.parameter_cap)
    resource_plan = plan_phase_resources(
        design_ir=design_ir,
        phase_config=phase_config,
        environment=_environment(backend="cpu", system_ram_bytes=8_000_000_000),
        baseline=baseline,
    )
    resolved_phase_config = apply_phase_resource_plan(phase_config, resource_plan)

    dataset_plan = plan_dataset_for_phase(
        design_ir=design_ir,
        phase="full",
        phase_config=phase_config,
        resolved_phase_config=resolved_phase_config,
        resource_plan=resource_plan,
        baseline=baseline,
    )

    assert resource_plan.admission_status == "reject"
    assert dataset_plan.executable is False
    assert dataset_plan.target_tokens == 0


def test_resolved_dataset_projection_compiles_with_runtime_config(tmp_path: Path):
    baseline_path = _bootstrap_baseline(tmp_path)
    baseline = load_baseline_definition(baseline_path)
    design_ir = _design_ir()
    phase_config = _phase_config("full", target_parameters=design_ir.parameter_cap)
    resource_plan = plan_phase_resources(
        design_ir=design_ir,
        phase_config=phase_config,
        environment=_environment(backend="cuda", vram_bytes_per_device=[14_000_000_000]),
        baseline=baseline,
    )
    resolved_phase_config = apply_phase_resource_plan(phase_config, resource_plan)
    dataset_plan = plan_dataset_for_phase(
        design_ir=design_ir,
        phase="full",
        phase_config=phase_config,
        resolved_phase_config=resolved_phase_config,
        resource_plan=resource_plan,
        baseline=baseline,
    )
    final_config = apply_dataset_plan(resolved_phase_config, dataset_plan)

    runtime_config = compile_runtime_phase_config(design_ir, final_config, attempt_id="attempt-0001", phase="full")

    assert runtime_config.dataset.target_tokens == final_config["dataset"]["target_tokens"]
    assert runtime_config.dataset.dataset_slice == final_config["runtime"]["dataset_slice"]
    assert runtime_config.dataset_plan["dataset_id"] == dataset_plan.dataset_id


def test_engine_persists_dataset_plan_before_execution(tmp_path: Path):
    _bootstrap_baseline(tmp_path)
    engine = InnovatorEngine(root=tmp_path)
    submit = engine.submit("Invent a recurrent retrieval decoder with cache-aware routing and explicit memory.")
    idea_dir = tmp_path / "ideas" / submit.idea_id
    environment = _environment(backend="cuda", vram_bytes_per_device=[24_000_000_000])
    (idea_dir / "environment.json").write_text(json.dumps(environment.to_dict()), encoding="utf-8")

    payload = engine.run(submit.idea_id, phase="smoke")
    phase_payload = payload["phases"][0]
    phase_dir = idea_dir / "runs" / payload["attempt_id"] / "smoke"
    resolved_payload = json.loads((phase_dir / "resolved-config.json").read_text(encoding="utf-8"))

    assert (phase_dir / "resource-plan.json").exists()
    assert (phase_dir / "dataset-plan.json").exists()
    assert any(path.endswith("dataset-plan.json") for path in phase_payload["artifacts_produced"])
    assert "dataset_plan" in resolved_payload
