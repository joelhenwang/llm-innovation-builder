from __future__ import annotations

from pathlib import Path

from auto_llm_innovator.design_ir import DesignIR
from auto_llm_innovator.filesystem import read_json
from auto_llm_innovator.modeling.interfaces import PhaseResult

from .baselines import BaselineDefinition, load_baseline_definition
from .models import EvaluationResult, EvaluationSignal, PhaseEvaluationSummary


def build_evaluation_result(*, idea_dir: Path, attempt_id: str, baseline_manifest: Path, results: list[PhaseResult]) -> EvaluationResult:
    baseline = load_baseline_definition(baseline_manifest)
    run_dir = idea_dir / "runs" / attempt_id
    design_ir = DesignIR.from_dict(read_json(idea_dir / "design_ir.json"))
    phase_summaries = [
        _build_phase_summary(
            idea_dir=idea_dir,
            run_dir=run_dir,
            baseline=baseline,
            result=result,
        )
        for result in results
    ]
    overall_recommendation = _overall_recommendation(phase_summaries)
    overall_summary = _overall_summary(phase_summaries, overall_recommendation)
    comparison_totals = {
        "phases_evaluated": len(phase_summaries),
        "promote_count": sum(1 for item in phase_summaries if item.recommendation == "promote"),
        "continue_with_caution_count": sum(
            1 for item in phase_summaries if item.recommendation == "continue_with_caution"
        ),
        "rerun_with_more_budget_count": sum(
            1 for item in phase_summaries if item.recommendation == "rerun_with_more_budget"
        ),
        "stop_count": sum(1 for item in phase_summaries if item.recommendation == "stop"),
        "clean_pass_count": sum(
            1 for item in phase_summaries if item.phase_status == "passed" and item.repair_count == 0
        ),
        "repaired_pass_count": sum(
            1 for item in phase_summaries if item.phase_status == "passed" and item.repair_count > 0
        ),
        "warning_count": sum(1 for item in phase_summaries if item.phase_status == "passed_with_warnings"),
        "failed_count": sum(1 for item in phase_summaries if item.phase_status == "failed"),
    }
    return EvaluationResult(
        idea_id=idea_dir.name,
        attempt_id=attempt_id,
        baseline_id=baseline.baseline_id,
        overall_recommendation=overall_recommendation,
        overall_summary=overall_summary,
        phase_summaries=phase_summaries,
        comparison_totals=comparison_totals,
        planned_ablations=[ablation.name for ablation in design_ir.ablation_plan],
    )


def _build_phase_summary(*, idea_dir: Path, run_dir: Path, baseline: BaselineDefinition, result: PhaseResult) -> PhaseEvaluationSummary:
    phase_dir = run_dir / result.phase
    summary_path = phase_dir / f"{result.phase}-summary.json"
    evaluation_path = phase_dir / "evaluation-report.json"
    repair_classification_path = phase_dir / "repair" / "failure-classification.json"
    repair_history_path = phase_dir / "repair" / "repair-history.json"

    summary_payload = read_json(summary_path) if summary_path.exists() else {}
    evaluation_payload = read_json(evaluation_path) if evaluation_path.exists() else {}
    repair_classification = read_json(repair_classification_path) if repair_classification_path.exists() else None
    repair_history = read_json(repair_history_path) if repair_history_path.exists() else []

    observed_loss = _maybe_float(result.key_metrics.get("loss"))
    baseline_metric = baseline.reference_metrics().get(_baseline_key(result.phase))
    baseline_loss = _maybe_float(baseline_metric)
    delta_vs_baseline = None
    if observed_loss is not None and baseline_loss is not None:
        delta_vs_baseline = round(observed_loss - baseline_loss, 4)

    evaluation_metrics = _evaluation_metrics(evaluation_payload)
    stop_reason = str(result.consumed_budget.get("stop_reason") or summary_payload.get("stop_reason") or "") or None
    recommendation = _phase_recommendation(
        result=result,
        observed_loss=observed_loss,
        baseline_loss=baseline_loss,
        stop_reason=stop_reason,
    )
    caution_flags = _caution_flags(
        result=result,
        stop_reason=stop_reason,
        repair_classification=repair_classification,
    )

    reliability_signals = [
        EvaluationSignal(
            name="phase_outcome",
            value=result.status,
            status=_reliability_status(result),
            summary=_reliability_summary(result, stop_reason),
            source_artifact=str(summary_path.relative_to(idea_dir)) if summary_path.exists() else "phase_result",
        ),
        EvaluationSignal(
            name="repair_lineage",
            value=result.repair_count,
            status="caution" if result.repair_attempted else "ok",
            summary=_repair_summary(result, repair_classification, repair_history),
            source_artifact=(
                str(repair_history_path.relative_to(idea_dir))
                if repair_history_path.exists()
                else str(repair_classification_path.relative_to(idea_dir))
                if repair_classification_path.exists()
                else "phase_result"
            ),
        ),
    ]
    quality_signals = [
        EvaluationSignal(
            name="loss",
            value=observed_loss,
            status=_loss_status(result, observed_loss, baseline_loss),
            summary=_loss_summary(observed_loss, baseline_loss, delta_vs_baseline),
            source_artifact=str(summary_path.relative_to(idea_dir)) if summary_path.exists() else "phase_result",
        )
    ]
    if evaluation_metrics:
        quality_signals.append(
            EvaluationSignal(
                name="evaluation_tasks",
                value=len(evaluation_metrics),
                status="ok",
                summary=f"Captured metrics for {len(evaluation_metrics)} evaluation task(s).",
                source_artifact=str(evaluation_path.relative_to(idea_dir)),
            )
        )
    practicality_signals = _practicality_signals(
        result=result,
        summary_payload=summary_payload,
        stop_reason=stop_reason,
        idea_dir=idea_dir,
        summary_path=summary_path,
    )

    return PhaseEvaluationSummary(
        phase=result.phase,
        phase_status=result.status,
        recommendation=recommendation,
        summary=_phase_summary_text(result, recommendation, stop_reason, delta_vs_baseline, caution_flags),
        caution_flags=caution_flags,
        reliability_signals=reliability_signals,
        quality_signals=quality_signals,
        practicality_signals=practicality_signals,
        observed_loss=observed_loss,
        baseline_loss=baseline_loss,
        delta_vs_baseline=delta_vs_baseline,
        evaluation_metrics=evaluation_metrics,
        repair_count=result.repair_count,
        stop_reason=stop_reason,
    )


def _baseline_key(phase: str) -> str:
    return f"{phase}.val_loss" if phase != "smoke" else "smoke.loss"


def _evaluation_metrics(evaluation_payload: dict) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for task in evaluation_payload.get("tasks", []):
        task_metrics = {
            str(metric_name): float(metric_value)
            for metric_name, metric_value in task.get("metrics", {}).items()
            if isinstance(metric_value, (int, float))
        }
        metrics[str(task.get("name", f"task_{len(metrics) + 1}"))] = task_metrics
    return metrics


def _phase_recommendation(*, result: PhaseResult, observed_loss: float | None, baseline_loss: float | None, stop_reason: str | None) -> str:
    baseline_ok = baseline_loss is None or (observed_loss is not None and observed_loss <= baseline_loss)
    if result.status == "failed":
        return "stop"
    if result.status == "passed_with_warnings":
        return "rerun_with_more_budget" if stop_reason == "max_wall_time_reached" and baseline_ok else "continue_with_caution"
    if result.repair_attempted:
        return "continue_with_caution" if baseline_ok else "stop"
    return "promote" if baseline_ok else "continue_with_caution"


def _caution_flags(*, result: PhaseResult, stop_reason: str | None, repair_classification: dict | None) -> list[str]:
    flags: list[str] = []
    resource_plan = _resource_plan(result)
    if result.repair_attempted:
        flags.append("repaired_pass" if result.status == "passed" else "repair_attempted")
    if result.repair_count > 1:
        flags.append("multiple_repairs")
    if result.status == "passed_with_warnings":
        flags.append("warning_only_result")
    if stop_reason == "max_wall_time_reached":
        flags.append("budget_limited")
    if resource_plan.get("admission_status") == "downscale":
        flags.append("resource_downscaled")
    if repair_classification and repair_classification.get("category"):
        flags.append(f"repair_category:{repair_classification['category']}")
    return flags


def _reliability_status(result: PhaseResult) -> str:
    if result.status == "failed":
        return "error"
    if result.status == "passed_with_warnings" or result.repair_attempted:
        return "caution"
    return "ok"


def _reliability_summary(result: PhaseResult, stop_reason: str | None) -> str:
    if result.status == "failed":
        return "Phase failed and should not be promoted."
    if result.status == "passed_with_warnings":
        return f"Phase completed with warnings and stop reason '{stop_reason}'."
    if result.repair_attempted:
        return f"Phase passed after {result.repair_count} repair attempt(s)."
    return "Phase passed cleanly without invoking repair."


def _repair_summary(result: PhaseResult, repair_classification: dict | None, repair_history: list[dict]) -> str:
    if not result.repair_attempted:
        return "No repair was needed."
    category = repair_classification.get("category") if repair_classification else None
    if result.status == "failed":
        return f"Repairs were exhausted after {len(repair_history) or result.repair_count} attempt(s); category={category}."
    return f"Recovered after {len(repair_history) or result.repair_count} repair attempt(s); category={category}."


def _loss_status(result: PhaseResult, observed_loss: float | None, baseline_loss: float | None) -> str:
    if observed_loss is None:
        return "error" if result.status == "failed" else "caution"
    if baseline_loss is None:
        return "ok"
    return "ok" if observed_loss <= baseline_loss else "caution"


def _loss_summary(observed_loss: float | None, baseline_loss: float | None, delta_vs_baseline: float | None) -> str:
    if observed_loss is None:
        return "Loss was not available for this phase."
    if baseline_loss is None:
        return f"Observed loss={observed_loss}; no baseline loss was available."
    return f"Observed loss={observed_loss}, baseline loss={baseline_loss}, delta={delta_vs_baseline}."


def _practicality_signals(
    *,
    result: PhaseResult,
    summary_payload: dict,
    stop_reason: str | None,
    idea_dir: Path,
    summary_path: Path,
) -> list[EvaluationSignal]:
    runtime = summary_payload.get("runtime", {})
    settings = runtime.get("settings", {})
    resource_plan = runtime.get("resource_plan", {})
    consumed_budget = result.consumed_budget
    source = str(summary_path.relative_to(idea_dir)) if summary_path.exists() else "phase_result"
    signals = [
        EvaluationSignal(
            name="steps_completed",
            value=consumed_budget.get("steps"),
            status="ok",
            summary=f"Completed {consumed_budget.get('steps', 0)} training step(s).",
            source_artifact=source,
        ),
        EvaluationSignal(
            name="runtime_budget",
            value={
                "batch_size": settings.get("batch_size"),
                "sequence_length": settings.get("sequence_length"),
                "dataset_slice": settings.get("dataset_slice"),
            },
            status="caution" if stop_reason == "max_wall_time_reached" else "ok",
            summary=(
                f"Configured batch_size={settings.get('batch_size')}, sequence_length={settings.get('sequence_length')}, "
                f"dataset_slice={settings.get('dataset_slice')}."
            ),
            source_artifact=source,
        ),
        EvaluationSignal(
            name="resume_status",
            value=bool(consumed_budget.get("resumed", False)),
            status="ok",
            summary="Run resumed from checkpoint metadata." if consumed_budget.get("resumed", False) else "Run started fresh.",
            source_artifact=source,
        ),
    ]
    if resource_plan:
        signals.append(
            EvaluationSignal(
                name="resource_admission",
                value=resource_plan.get("admission_status"),
                status="caution" if resource_plan.get("admission_status") == "downscale" else "ok",
                summary=str(resource_plan.get("planner_summary", "Resource admission plan applied.")),
                source_artifact=source,
            )
        )
    return signals


def _phase_summary_text(
    result: PhaseResult,
    recommendation: str,
    stop_reason: str | None,
    delta_vs_baseline: float | None,
    caution_flags: list[str],
) -> str:
    parts = [f"{result.phase} finished with status={result.status} and recommendation={recommendation}."]
    if stop_reason:
        parts.append(f"Stop reason was {stop_reason}.")
    if delta_vs_baseline is not None:
        parts.append(f"Loss delta vs baseline was {delta_vs_baseline}.")
    if caution_flags:
        parts.append(f"Caution flags: {', '.join(caution_flags)}.")
    return " ".join(parts)


def _overall_recommendation(phase_summaries: list[PhaseEvaluationSummary]) -> str:
    recommendations = {item.recommendation for item in phase_summaries}
    if "stop" in recommendations:
        return "stop"
    if "rerun_with_more_budget" in recommendations:
        return "rerun_with_more_budget"
    if "continue_with_caution" in recommendations:
        return "continue_with_caution"
    return "promote" if phase_summaries else "stop"


def _overall_summary(phase_summaries: list[PhaseEvaluationSummary], overall_recommendation: str) -> str:
    if not phase_summaries:
        return "No phases were available for evaluation."
    promoted = sum(1 for item in phase_summaries if item.recommendation == "promote")
    cautions = sum(1 for item in phase_summaries if item.recommendation == "continue_with_caution")
    reruns = sum(1 for item in phase_summaries if item.recommendation == "rerun_with_more_budget")
    stops = sum(1 for item in phase_summaries if item.recommendation == "stop")
    return (
        f"Overall recommendation={overall_recommendation}; promote={promoted}, caution={cautions}, "
        f"rerun={reruns}, stop={stops}."
    )


def _maybe_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _resource_plan(result: PhaseResult) -> dict:
    for artifact in result.artifacts_produced:
        if artifact.endswith("resource-plan.json"):
            try:
                return read_json(Path(artifact))
            except Exception:
                return {}
    return {}
