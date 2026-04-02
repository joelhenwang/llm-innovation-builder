from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from auto_llm_innovator.design_ir import DesignIR
from auto_llm_innovator.env import EnvironmentReport
from auto_llm_innovator.evaluation import BaselineDefinition, EvaluationResult
from auto_llm_innovator.runtime.phases import RuntimePhaseSettings
from auto_llm_innovator.tracking.ranking import AttemptRankingResult


@dataclass(slots=True)
class ResourceAdjustment:
    field_name: str
    previous_value: Any
    new_value: Any
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PhaseResourceRequest:
    idea_id: str
    phase: str
    target_parameters: int
    prefer_rocm: bool
    runtime_settings: RuntimePhaseSettings
    parameter_cap: int
    has_recurrent_state: bool
    has_external_memory: bool
    has_cache_path: bool
    baseline_hardware_assumptions: dict[str, Any] = field(default_factory=dict)
    baseline_token_budget_assumptions: dict[str, Any] = field(default_factory=dict)
    prior_rank_label: str | None = None
    prior_best_so_far: bool | None = None
    prior_overall_recommendation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["runtime_settings"] = self.runtime_settings.to_dict()
        return payload


@dataclass(slots=True)
class PhaseResourcePlan:
    idea_id: str
    phase: str
    admission_status: str
    resolved_target_parameters: int
    resolved_prefer_rocm: bool
    resolved_runtime_settings: RuntimePhaseSettings
    adjustments: list[ResourceAdjustment] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    planner_summary: str = ""
    baseline_assumption_used: dict[str, Any] = field(default_factory=dict)
    ranking_context_used: dict[str, Any] = field(default_factory=dict)
    estimated_required_bytes: int = 0
    estimated_available_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "idea_id": self.idea_id,
            "phase": self.phase,
            "admission_status": self.admission_status,
            "resolved_target_parameters": self.resolved_target_parameters,
            "resolved_prefer_rocm": self.resolved_prefer_rocm,
            "resolved_runtime_settings": self.resolved_runtime_settings.to_dict(),
            "adjustments": [item.to_dict() for item in self.adjustments],
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "planner_summary": self.planner_summary,
            "baseline_assumption_used": dict(self.baseline_assumption_used),
            "ranking_context_used": dict(self.ranking_context_used),
            "estimated_required_bytes": self.estimated_required_bytes,
            "estimated_available_bytes": self.estimated_available_bytes,
        }


def build_phase_resource_request(
    *,
    design_ir: DesignIR,
    phase_config: dict[str, Any],
    baseline: BaselineDefinition,
    environment: EnvironmentReport,
    ranking_result: AttemptRankingResult | None = None,
    evaluation_result: EvaluationResult | None = None,
) -> PhaseResourceRequest:
    runtime_payload = dict(phase_config.get("runtime", {}))
    settings = RuntimePhaseSettings(
        max_steps=int(runtime_payload["max_steps"]),
        max_wall_time_seconds=int(runtime_payload["max_wall_time_seconds"]),
        sequence_length=int(runtime_payload["sequence_length"]),
        batch_size=int(runtime_payload["batch_size"]),
        checkpoint_every_steps=int(runtime_payload["checkpoint_every_steps"]),
        resume_enabled=bool(runtime_payload["resume_enabled"]),
        evaluation_scope=str(runtime_payload["evaluation_scope"]),
        dataset_slice=str(runtime_payload["dataset_slice"]),
    )
    _ = environment
    return PhaseResourceRequest(
        idea_id=design_ir.idea_id,
        phase=str(phase_config["phase"]),
        target_parameters=int(phase_config["target_parameters"]),
        prefer_rocm=bool(phase_config.get("prefer_rocm", False)),
        runtime_settings=settings,
        parameter_cap=int(design_ir.parameter_cap),
        has_recurrent_state=design_ir.architecture.state_semantics.has_recurrent_state,
        has_external_memory=design_ir.architecture.state_semantics.has_external_memory,
        has_cache_path=design_ir.architecture.state_semantics.has_cache_path,
        baseline_hardware_assumptions=dict(baseline.hardware_assumptions),
        baseline_token_budget_assumptions=dict(baseline.token_budget_assumptions),
        prior_rank_label=ranking_result.rank_label if ranking_result is not None else None,
        prior_best_so_far=ranking_result.best_so_far if ranking_result is not None else None,
        prior_overall_recommendation=evaluation_result.overall_recommendation if evaluation_result is not None else None,
    )


def build_phase_resource_plan(
    *,
    request: PhaseResourceRequest,
    environment: EnvironmentReport,
) -> PhaseResourcePlan:
    resolved_settings = RuntimePhaseSettings(
        max_steps=request.runtime_settings.max_steps,
        max_wall_time_seconds=request.runtime_settings.max_wall_time_seconds,
        sequence_length=request.runtime_settings.sequence_length,
        batch_size=request.runtime_settings.batch_size,
        checkpoint_every_steps=request.runtime_settings.checkpoint_every_steps,
        resume_enabled=request.runtime_settings.resume_enabled,
        evaluation_scope=request.runtime_settings.evaluation_scope,
        dataset_slice=request.runtime_settings.dataset_slice,
    )
    resolved_target_parameters = int(request.target_parameters)
    resolved_prefer_rocm = bool(request.prefer_rocm and environment.accelerator_backend == "rocm")
    adjustments: list[ResourceAdjustment] = []
    reasons: list[str] = []
    warnings: list[str] = []

    baseline_assumption_used = {
        "hardware": dict(request.baseline_hardware_assumptions),
        "token_budget_for_phase": request.baseline_token_budget_assumptions.get(request.phase),
    }
    ranking_context_used = {
        "rank_label": request.prior_rank_label,
        "best_so_far": request.prior_best_so_far,
        "overall_recommendation": request.prior_overall_recommendation,
    }

    _append_assumption_warnings(request, environment, warnings)
    rank_multiplier = _ranking_multiplier(request)
    if rank_multiplier < 1.0:
        warnings.append(
            f"Prior ranking context reduced effective admission budget by multiplier {rank_multiplier:.2f}."
        )

    if environment.accelerator_backend in {"cuda", "rocm"} and request.prefer_rocm and environment.accelerator_backend != "rocm":
        warnings.append("Phase preferred ROCm but the current environment exposes a non-ROCm accelerator.")

    while True:
        estimated_required_bytes = _estimate_required_bytes(
            target_parameters=resolved_target_parameters,
            sequence_length=resolved_settings.sequence_length,
            batch_size=resolved_settings.batch_size,
            request=request,
        )
        estimated_available_bytes = _effective_available_bytes(environment, request.phase, rank_multiplier)
        if estimated_required_bytes <= estimated_available_bytes and _phase_backend_allowed(
            environment=environment,
            phase=request.phase,
            target_parameters=resolved_target_parameters,
        ):
            status = "downscale" if adjustments else "admit"
            reasons.append(
                f"Estimated requirement {estimated_required_bytes} bytes fits within {estimated_available_bytes} bytes."
            )
            return PhaseResourcePlan(
                idea_id=request.idea_id,
                phase=request.phase,
                admission_status=status,
                resolved_target_parameters=resolved_target_parameters,
                resolved_prefer_rocm=resolved_prefer_rocm,
                resolved_runtime_settings=resolved_settings,
                adjustments=adjustments,
                reasons=reasons,
                warnings=warnings,
                planner_summary=_planner_summary(status, adjustments, warnings, estimated_required_bytes, estimated_available_bytes),
                baseline_assumption_used=baseline_assumption_used,
                ranking_context_used=ranking_context_used,
                estimated_required_bytes=estimated_required_bytes,
                estimated_available_bytes=estimated_available_bytes,
            )
        adjustment = _next_adjustment(
            request=request,
            environment=environment,
            settings=resolved_settings,
            target_parameters=resolved_target_parameters,
            rank_multiplier=rank_multiplier,
        )
        if adjustment is None:
            rejection_reason = (
                "No accelerator was available for this phase after bounded downscaling."
                if not _phase_backend_allowed(environment=environment, phase=request.phase, target_parameters=resolved_target_parameters)
                else "Requested phase remained infeasible after bounded downscaling."
            )
            reasons.append(rejection_reason)
            return PhaseResourcePlan(
                idea_id=request.idea_id,
                phase=request.phase,
                admission_status="reject",
                resolved_target_parameters=resolved_target_parameters,
                resolved_prefer_rocm=resolved_prefer_rocm,
                resolved_runtime_settings=resolved_settings,
                adjustments=adjustments,
                reasons=reasons,
                warnings=warnings,
                planner_summary=_planner_summary(
                    "reject",
                    adjustments,
                    warnings,
                    estimated_required_bytes,
                    estimated_available_bytes,
                ),
                baseline_assumption_used=baseline_assumption_used,
                ranking_context_used=ranking_context_used,
                estimated_required_bytes=estimated_required_bytes,
                estimated_available_bytes=estimated_available_bytes,
            )
        adjustments.append(adjustment)
        if adjustment.field_name == "runtime.batch_size":
            resolved_settings.batch_size = int(adjustment.new_value)
        elif adjustment.field_name == "runtime.sequence_length":
            resolved_settings.sequence_length = int(adjustment.new_value)
        elif adjustment.field_name == "runtime.max_steps":
            resolved_settings.max_steps = int(adjustment.new_value)
            resolved_settings.resume_enabled = False if resolved_settings.max_steps <= 2 else resolved_settings.resume_enabled
            if resolved_settings.checkpoint_every_steps > resolved_settings.max_steps:
                previous = resolved_settings.checkpoint_every_steps
                resolved_settings.checkpoint_every_steps = max(0, resolved_settings.max_steps)
                adjustments.append(
                    ResourceAdjustment(
                        field_name="runtime.checkpoint_every_steps",
                        previous_value=previous,
                        new_value=resolved_settings.checkpoint_every_steps,
                        reason="Checkpoint cadence cannot exceed the reduced max step budget.",
                    )
                )
        elif adjustment.field_name == "target_parameters":
            resolved_target_parameters = int(adjustment.new_value)


def _ranking_multiplier(request: PhaseResourceRequest) -> float:
    label = request.prior_rank_label
    if label == "leading":
        return 1.0
    if label == "improving":
        return 0.85
    if label == "caution":
        return 0.65
    if label == "noncompetitive":
        return 0.4
    if request.prior_overall_recommendation == "rerun_with_more_budget":
        return 0.7
    return 1.0


def _estimate_required_bytes(
    *,
    target_parameters: int,
    sequence_length: int,
    batch_size: int,
    request: PhaseResourceRequest,
) -> int:
    phase_multiplier = {
        "smoke": 1.05,
        "small": 1.35,
        "full": 1.7,
    }[request.phase]
    architecture_multiplier = 1.0
    if request.has_recurrent_state:
        architecture_multiplier += 0.08
    if request.has_external_memory:
        architecture_multiplier += 0.12
    if request.has_cache_path:
        architecture_multiplier += 0.05
    token_multiplier = max(1.0, (sequence_length / 16.0) * (max(batch_size, 1) / 4.0))
    estimated = target_parameters * 4.0 * phase_multiplier * architecture_multiplier * token_multiplier
    return int(estimated)


def _effective_available_bytes(environment: EnvironmentReport, phase: str, rank_multiplier: float) -> int:
    if environment.accelerator_backend in {"cuda", "rocm"} and environment.vram_bytes_per_device:
        available = max(environment.vram_bytes_per_device)
        phase_fraction = {"smoke": 0.72, "small": 0.62, "full": 0.55}[phase]
    else:
        available = environment.system_ram_bytes
        phase_fraction = {"smoke": 0.38, "small": 0.16, "full": 0.1}[phase]
    return int(available * phase_fraction * rank_multiplier)


def _phase_backend_allowed(*, environment: EnvironmentReport, phase: str, target_parameters: int) -> bool:
    if environment.accelerator_backend in {"cuda", "rocm"}:
        return True
    if phase == "smoke":
        return True
    if phase == "small":
        return target_parameters <= 200_000_000
    return target_parameters <= 80_000_000


def _next_adjustment(
    *,
    request: PhaseResourceRequest,
    environment: EnvironmentReport,
    settings: RuntimePhaseSettings,
    target_parameters: int,
    rank_multiplier: float,
) -> ResourceAdjustment | None:
    _ = environment
    _ = rank_multiplier
    if settings.batch_size > 1:
        return ResourceAdjustment(
            field_name="runtime.batch_size",
            previous_value=settings.batch_size,
            new_value=settings.batch_size - 1,
            reason="Reduce activation footprint before changing sequence length.",
        )
    minimum_sequence = 4 if request.phase == "smoke" else 6 if request.phase == "small" else 8
    if settings.sequence_length > minimum_sequence:
        reduced = max(minimum_sequence, settings.sequence_length - (4 if settings.sequence_length > 8 else 2))
        if reduced != settings.sequence_length:
            return ResourceAdjustment(
                field_name="runtime.sequence_length",
                previous_value=settings.sequence_length,
                new_value=reduced,
                reason="Reduce context length after batch-size downscaling is exhausted.",
            )
    minimum_steps = 2 if request.phase == "smoke" else 3 if request.phase == "small" else 4
    if settings.max_steps > minimum_steps:
        reduced_steps = max(minimum_steps, settings.max_steps - (2 if settings.max_steps > 4 else 1))
        if reduced_steps != settings.max_steps:
            return ResourceAdjustment(
                field_name="runtime.max_steps",
                previous_value=settings.max_steps,
                new_value=reduced_steps,
                reason="Reduce training-step budget after token-shape downscaling is exhausted.",
            )
    parameter_floor = 80_000_000 if request.phase == "smoke" else 120_000_000 if request.phase == "small" else 160_000_000
    if target_parameters > parameter_floor:
        reduced_parameters = max(parameter_floor, int(target_parameters * 0.6))
        if reduced_parameters != target_parameters:
            return ResourceAdjustment(
                field_name="target_parameters",
                previous_value=target_parameters,
                new_value=reduced_parameters,
                reason="Reduce target parameter budget as the last bounded downscaling step.",
            )
    return None


def _append_assumption_warnings(
    request: PhaseResourceRequest,
    environment: EnvironmentReport,
    warnings: list[str],
) -> None:
    expected_backend = request.baseline_hardware_assumptions.get("device") or request.baseline_hardware_assumptions.get("backend")
    if expected_backend and str(expected_backend) != environment.accelerator_backend:
        warnings.append(
            f"Baseline hardware assumption '{expected_backend}' does not match current backend '{environment.accelerator_backend}'."
        )
    token_budget = request.baseline_token_budget_assumptions.get(request.phase)
    if token_budget is not None:
        warnings.append(f"Baseline token budget for phase is {token_budget}.")


def _planner_summary(
    status: str,
    adjustments: list[ResourceAdjustment],
    warnings: list[str],
    estimated_required_bytes: int,
    estimated_available_bytes: int,
) -> str:
    parts = [
        f"Admission status={status}.",
        f"Estimated required bytes={estimated_required_bytes}.",
        f"Estimated available bytes={estimated_available_bytes}.",
    ]
    if adjustments:
        parts.append(f"Applied {len(adjustments)} bounded adjustment(s).")
    if warnings:
        parts.append(f"Warnings={len(warnings)}.")
    return " ".join(parts)
