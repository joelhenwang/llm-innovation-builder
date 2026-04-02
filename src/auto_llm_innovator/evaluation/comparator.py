from __future__ import annotations

from pathlib import Path
from typing import Iterable

from auto_llm_innovator.filesystem import read_json, write_json, write_text
from auto_llm_innovator.modeling.interfaces import PhaseResult

from .baselines import load_baseline_definition
from .models import EvaluationResult


def compare_against_baseline(
    baseline_manifest: Path,
    results: Iterable[PhaseResult],
    evaluation_result: EvaluationResult | None = None,
) -> dict:
    baseline = load_baseline_definition(baseline_manifest)
    reference_metrics = baseline.reference_metrics()
    comparisons: dict[str, dict] = {}
    phase_summaries = {
        item.phase: item for item in (evaluation_result.phase_summaries if evaluation_result is not None else [])
    }
    for result in results:
        baseline_key = f"{result.phase}.val_loss" if result.phase != "smoke" else "smoke.loss"
        baseline_metric = reference_metrics.get(baseline_key)
        observed = result.key_metrics.get("loss")
        phase_summary = phase_summaries.get(result.phase)
        comparisons[result.phase] = {
            "observed_loss": observed,
            "baseline_loss": baseline_metric,
            "delta_vs_baseline": None if baseline_metric is None or observed is None else observed - baseline_metric,
            "status": result.status,
            "recommendation": phase_summary.recommendation if phase_summary is not None else None,
            "caution_flags": list(phase_summary.caution_flags) if phase_summary is not None else [],
            "repair_count": phase_summary.repair_count if phase_summary is not None else result.repair_count,
            "stop_reason": phase_summary.stop_reason if phase_summary is not None else result.consumed_budget.get("stop_reason"),
        }
    payload = {
        "baseline_id": baseline.baseline_id,
        "baseline_family": baseline.family,
        "comparisons": comparisons,
    }
    if evaluation_result is not None:
        payload["overall_recommendation"] = evaluation_result.overall_recommendation
        payload["comparison_totals"] = dict(evaluation_result.comparison_totals)
    return payload


def render_decision_report(
    report_path: Path,
    idea_id: str,
    attempt_id: str,
    comparisons: dict,
    results: Iterable[PhaseResult],
    evaluation_result: EvaluationResult | None = None,
) -> None:
    phase_summaries = {
        item.phase: item for item in (evaluation_result.phase_summaries if evaluation_result is not None else [])
    }
    lines = [
        f"# Decision Report: {idea_id}",
        "",
        f"- Attempt: `{attempt_id}`",
        f"- Baseline: `{comparisons['baseline_id']}`",
    ]
    if comparisons.get("baseline_family") is not None:
        lines.append(f"- Baseline family: `{comparisons['baseline_family']}`")
    if evaluation_result is not None:
        lines.extend(
            [
                f"- Overall recommendation: `{evaluation_result.overall_recommendation}`",
                f"- Evaluation summary: {evaluation_result.overall_summary}",
            ]
        )
    lines.extend(["", "## Technical outcomes"])
    for result in results:
        phase_summary = phase_summaries.get(result.phase)
        lines.append(
            f"- `{result.phase}`: status={result.status}, loss={result.key_metrics.get('loss')}, next={result.next_action_recommendation}"
        )
        if phase_summary is not None:
            lines.append(
                f"  evaluation={phase_summary.recommendation}, stop_reason={phase_summary.stop_reason}, caution_flags={phase_summary.caution_flags}"
            )
        if result.repair_attempted:
            lines.append(
                f"  repair={result.repair_outcome}, count={result.repair_count}, classification={result.failure_classification.get('category') if result.failure_classification else None}"
            )
    lines.extend(["", "## Scientific assessment"])
    if evaluation_result is None:
        lines.append("- Evaluation aggregation was not available for this attempt.")
    else:
        for summary in evaluation_result.phase_summaries:
            lines.append(f"- `{summary.phase}`: {summary.summary}")
    lines.extend(["", "## Baseline comparison"])
    for phase, payload in comparisons["comparisons"].items():
        line = (
            f"- `{phase}`: observed={payload['observed_loss']}, baseline={payload['baseline_loss']}, "
            f"delta={payload['delta_vs_baseline']}, recommendation={payload.get('recommendation')}"
        )
        if payload.get("caution_flags"):
            line += f", caution_flags={payload['caution_flags']}"
        lines.append(line)
    if evaluation_result is not None:
        lines.extend(["", "## Planned ablations"])
        if evaluation_result.planned_ablations:
            for name in evaluation_result.planned_ablations:
                lines.append(f"- `{name}`: planned but not yet executed in Phase 8 v1.")
        else:
            lines.append("- No explicit ablations were declared in DesignIR.")
    if "ranking" in comparisons:
        ranking = comparisons["ranking"]
        lines.extend(
            [
                "",
                "## Attempt ranking",
                f"- Rank label: `{ranking['rank_label']}`",
                f"- Promotion score: {ranking['promotion_score']}",
                f"- Best so far: {ranking['best_so_far']}",
                f"- Baseline comparison: {ranking['baseline_comparison_summary']}",
                f"- Prior attempts: {ranking['prior_attempt_comparison_summary']}",
            ]
        )
        for reason in ranking.get("winning_reasons", []):
            lines.append(f"- Strength: {reason}")
        for factor in ranking.get("limiting_factors", []):
            lines.append(f"- Limitation: {factor}")
    write_text(report_path, "\n".join(lines) + "\n")
    report_payload = dict(comparisons)
    if evaluation_result is not None:
        report_payload["evaluation_result"] = evaluation_result.to_dict()
    write_json(report_path.with_suffix(".json"), report_payload)
