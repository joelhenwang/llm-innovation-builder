from __future__ import annotations

from pathlib import Path
from typing import Iterable

from auto_llm_innovator.filesystem import read_json, write_json, write_text
from auto_llm_innovator.modeling.interfaces import PhaseResult


def compare_against_baseline(baseline_manifest: Path, results: Iterable[PhaseResult]) -> dict:
    baseline = read_json(baseline_manifest)
    comparisons: dict[str, dict] = {}
    for result in results:
        baseline_key = f"{result.phase}.val_loss" if result.phase != "smoke" else "smoke.loss"
        baseline_metric = baseline["reference_metrics"].get(baseline_key)
        observed = result.key_metrics.get("loss")
        comparisons[result.phase] = {
            "observed_loss": observed,
            "baseline_loss": baseline_metric,
            "delta_vs_baseline": None if baseline_metric is None or observed is None else observed - baseline_metric,
        }
    return {
        "baseline_id": baseline["baseline_id"],
        "comparisons": comparisons,
    }


def render_decision_report(report_path: Path, idea_id: str, attempt_id: str, comparisons: dict, results: Iterable[PhaseResult]) -> None:
    lines = [
        f"# Decision Report: {idea_id}",
        "",
        f"- Attempt: `{attempt_id}`",
        f"- Baseline: `{comparisons['baseline_id']}`",
        "",
        "## Phase outcomes",
    ]
    for result in results:
        lines.append(f"- `{result.phase}`: {result.status}, loss={result.key_metrics.get('loss')}, recommendation={result.next_action_recommendation}")
    lines.extend(
        [
            "",
            "## Agent rationale",
            "- Continue while the architecture remains original and losses trend downward.",
            "- Stop or redesign if novelty cannot be defended or metrics plateau without explanation.",
            "",
            "## Baseline comparison",
        ]
    )
    for phase, payload in comparisons["comparisons"].items():
        lines.append(
            f"- `{phase}`: observed={payload['observed_loss']}, baseline={payload['baseline_loss']}, delta={payload['delta_vs_baseline']}"
        )
    write_text(report_path, "\n".join(lines) + "\n")
    write_json(report_path.with_suffix(".json"), comparisons)
