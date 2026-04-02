from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from auto_llm_innovator.filesystem import write_json

from .models import DatasetDefinition, DatasetPlan
from .registry import dataset_definition_by_id, default_dataset_definition_for_phase

if TYPE_CHECKING:
    from auto_llm_innovator.design_ir.models import DesignIR
    from auto_llm_innovator.evaluation import BaselineDefinition
    from auto_llm_innovator.planning import PhaseResourcePlan


def plan_dataset_for_phase(
    *,
    design_ir: DesignIR,
    phase: str,
    phase_config: dict[str, Any],
    resolved_phase_config: dict[str, Any],
    resource_plan: PhaseResourcePlan,
    baseline: BaselineDefinition | None = None,
) -> DatasetPlan:
    stage = _training_stage_for_phase(design_ir, phase)
    definition = _dataset_definition_for_stage(phase=phase, stage_dataset_name=stage.dataset_name)
    preset = definition.preset_for_phase(phase)

    base_dataset = dict(phase_config.get("dataset", {}))
    resolved_runtime = dict(resolved_phase_config.get("runtime", {}))
    resolved_slice = str(resolved_runtime.get("dataset_slice", preset.default_dataset_slice))
    baseline_token_budget = None
    if baseline is not None and phase in baseline.token_budget_assumptions:
        try:
            baseline_token_budget = int(baseline.token_budget_assumptions[phase])
        except Exception:
            baseline_token_budget = None

    warnings: list[str] = []
    reasons: list[str] = []
    executable = resource_plan.admission_status != "reject"

    if resource_plan.admission_status == "reject":
        target_tokens = 0
        reasons.append("Phase 10 resource admission rejected execution, so dataset planning remained report-only.")
        dataset_slice = resolved_slice
    else:
        base_target_tokens = int(base_dataset.get("target_tokens", stage.target_tokens))
        target_tokens = min(base_target_tokens, int(stage.target_tokens), int(preset.default_target_tokens))
        scaling_ratio = _resource_scaling_ratio(phase_config=phase_config, resolved_phase_config=resolved_phase_config, resource_plan=resource_plan)
        if scaling_ratio < 1.0:
            target_tokens = max(_minimum_tokens_for_phase(phase), int(target_tokens * scaling_ratio))
            reasons.append(f"Reduced dataset token budget by resource scaling ratio {scaling_ratio:.2f}.")
        dataset_slice = _resolved_dataset_slice(resolved_slice, resource_plan.admission_status)
        if dataset_slice != resolved_slice:
            reasons.append(f"Simplified dataset slice from '{resolved_slice}' to '{dataset_slice}' after resource downscaling.")
        if baseline_token_budget is not None:
            warnings.append(f"Baseline token budget for phase is {baseline_token_budget}.")

    return DatasetPlan(
        idea_id=design_ir.idea_id,
        phase=phase,
        dataset_id=definition.dataset_id,
        dataset_name=preset.dataset_name,
        label=definition.label,
        description=stage.dataset_description or preset.description,
        target_tokens=target_tokens,
        dataset_slice=dataset_slice,
        objective=stage.objective,
        admission_status=resource_plan.admission_status,
        executable=executable,
        curriculum_notes=list(preset.curriculum_notes),
        supported_capabilities=list(definition.supported_capabilities),
        warnings=warnings,
        reasons=reasons,
        baseline_token_budget=baseline_token_budget,
        availability=definition.availability,
        license_tier=definition.license_tier,
    )


def apply_dataset_plan(phase_config: dict[str, Any], dataset_plan: DatasetPlan) -> dict[str, Any]:
    resolved = {
        **phase_config,
        "dataset": {
            **dict(phase_config.get("dataset", {})),
            "dataset_name": dataset_plan.dataset_name,
            "description": dataset_plan.description,
            "target_tokens": int(dataset_plan.target_tokens),
        },
        "runtime": {
            **dict(phase_config.get("runtime", {})),
            "dataset_slice": dataset_plan.dataset_slice,
        },
        "dataset_plan": dataset_plan.to_dict(),
    }
    return resolved


def persist_dataset_plan(run_dir: Path, dataset_plan: DatasetPlan) -> Path:
    path = run_dir / "dataset-plan.json"
    write_json(path, dataset_plan.to_dict())
    return path


def _training_stage_for_phase(design_ir: DesignIR, phase: str):
    for stage in design_ir.training_plan:
        if stage.stage == phase:
            return stage
    raise KeyError(f"DesignIR is missing training stage '{phase}'.")


def _dataset_definition_for_stage(*, phase: str, stage_dataset_name: str) -> DatasetDefinition:
    try:
        return dataset_definition_by_id(stage_dataset_name)
    except KeyError:
        return default_dataset_definition_for_phase(phase)


def _resource_scaling_ratio(
    *,
    phase_config: dict[str, Any],
    resolved_phase_config: dict[str, Any],
    resource_plan: PhaseResourcePlan,
) -> float:
    if resource_plan.admission_status != "downscale":
        return 1.0
    base_runtime = dict(phase_config.get("runtime", {}))
    resolved_runtime = dict(resolved_phase_config.get("runtime", {}))
    ratios = []
    for key in ("max_steps", "batch_size", "sequence_length"):
        base_value = int(base_runtime.get(key, 0) or 0)
        resolved_value = int(resolved_runtime.get(key, 0) or 0)
        if base_value > 0 and resolved_value > 0:
            ratios.append(resolved_value / base_value)
    base_parameters = int(phase_config.get("target_parameters", 0) or 0)
    resolved_parameters = int(resolved_phase_config.get("target_parameters", 0) or 0)
    if base_parameters > 0 and resolved_parameters > 0:
        ratios.append(resolved_parameters / base_parameters)
    return min(ratios) if ratios else 1.0


def _resolved_dataset_slice(runtime_slice: str, admission_status: str) -> str:
    if admission_status != "downscale":
        return runtime_slice
    if runtime_slice == "full":
        return "curated"
    if runtime_slice == "curated":
        return "tiny"
    return runtime_slice


def _minimum_tokens_for_phase(phase: str) -> int:
    return {
        "smoke": 10_000,
        "small": 250_000,
        "full": 1_000_000,
    }[phase]
