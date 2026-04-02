from __future__ import annotations

from pathlib import Path
from typing import Any

from auto_llm_innovator.design_ir import DesignIR
from auto_llm_innovator.env import EnvironmentReport
from auto_llm_innovator.evaluation import BaselineDefinition, EvaluationResult
from auto_llm_innovator.filesystem import write_json
from auto_llm_innovator.tracking.ranking import AttemptRankingResult

from .resources import PhaseResourcePlan, build_phase_resource_plan, build_phase_resource_request


def plan_phase_resources(
    *,
    design_ir: DesignIR,
    phase_config: dict[str, Any],
    environment: EnvironmentReport,
    baseline: BaselineDefinition,
    ranking_result: AttemptRankingResult | None = None,
    evaluation_result: EvaluationResult | None = None,
) -> PhaseResourcePlan:
    request = build_phase_resource_request(
        design_ir=design_ir,
        phase_config=phase_config,
        baseline=baseline,
        environment=environment,
        ranking_result=ranking_result,
        evaluation_result=evaluation_result,
    )
    return build_phase_resource_plan(request=request, environment=environment)


def apply_phase_resource_plan(phase_config: dict[str, Any], plan: PhaseResourcePlan) -> dict[str, Any]:
    resolved = {
        **phase_config,
        "target_parameters": int(plan.resolved_target_parameters),
        "prefer_rocm": bool(plan.resolved_prefer_rocm),
        "runtime": {
            **dict(phase_config.get("runtime", {})),
            **plan.resolved_runtime_settings.to_dict(),
        },
        "resource_plan": {
            "admission_status": plan.admission_status,
            "adjustments": [item.to_dict() for item in plan.adjustments],
            "planner_summary": plan.planner_summary,
            "baseline_assumption_used": dict(plan.baseline_assumption_used),
            "ranking_context_used": dict(plan.ranking_context_used),
            "warnings": list(plan.warnings),
            "reasons": list(plan.reasons),
            "estimated_required_bytes": plan.estimated_required_bytes,
            "estimated_available_bytes": plan.estimated_available_bytes,
        },
    }
    return resolved


def persist_resource_plan(run_dir: Path, plan: PhaseResourcePlan) -> Path:
    path = run_dir / "resource-plan.json"
    write_json(path, plan.to_dict())
    return path
