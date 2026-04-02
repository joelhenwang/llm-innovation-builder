from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from auto_llm_innovator.evaluation import EvaluationResult
from auto_llm_innovator.evaluation.baselines import BaselineDefinition
from auto_llm_innovator.filesystem import read_json


@dataclass(slots=True)
class AttemptRankingResult:
    idea_id: str
    attempt_id: str
    baseline_id: str
    promotion_score: float
    rank_label: str
    baseline_comparison_summary: str
    prior_attempt_comparison_summary: str
    winning_reasons: list[str] = field(default_factory=list)
    limiting_factors: list[str] = field(default_factory=list)
    prior_attempts_considered: list[str] = field(default_factory=list)
    best_so_far: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AttemptRankingResult":
        return cls(
            idea_id=str(payload["idea_id"]),
            attempt_id=str(payload["attempt_id"]),
            baseline_id=str(payload["baseline_id"]),
            promotion_score=float(payload["promotion_score"]),
            rank_label=str(payload["rank_label"]),
            baseline_comparison_summary=str(payload["baseline_comparison_summary"]),
            prior_attempt_comparison_summary=str(payload["prior_attempt_comparison_summary"]),
            winning_reasons=list(payload.get("winning_reasons", [])),
            limiting_factors=list(payload.get("limiting_factors", [])),
            prior_attempts_considered=list(payload.get("prior_attempts_considered", [])),
            best_so_far=bool(payload.get("best_so_far", False)),
        )


def build_attempt_ranking(
    *,
    idea_dir: Path,
    baseline: BaselineDefinition,
    evaluation_result: EvaluationResult,
    status: dict,
) -> AttemptRankingResult:
    current_score = _score_evaluation_result(evaluation_result)
    prior_results = _load_prior_attempt_evaluations(idea_dir=idea_dir, status=status, latest_attempt_id=evaluation_result.attempt_id)
    prior_scores = [(attempt_id, payload, _score_evaluation_result(payload)) for attempt_id, payload in prior_results]

    winning_reasons: list[str] = []
    limiting_factors: list[str] = []
    if evaluation_result.comparison_totals.get("clean_pass_count", 0) > 0:
        winning_reasons.append("Contains at least one clean promotable phase.")
    if evaluation_result.comparison_totals.get("promote_count", 0) > 0:
        winning_reasons.append("Includes promotable phase recommendations from Phase 8.")
    if evaluation_result.comparison_totals.get("warning_count", 0) > 0:
        limiting_factors.append("Contains warning-only outcomes that remain provisional.")
    if evaluation_result.comparison_totals.get("repaired_pass_count", 0) > 0:
        limiting_factors.append("Repair lineage lowers rank relative to clean passes.")
    if evaluation_result.overall_recommendation == "stop":
        limiting_factors.append("Overall recommendation is stop, so this attempt is noncompetitive.")

    best_prior_score = max((score for _, _, score in prior_scores), default=None)
    best_so_far = best_prior_score is None or current_score >= best_prior_score
    prior_summary = _prior_attempt_summary(current_score, prior_scores)
    rank_label = _rank_label(evaluation_result, best_so_far)
    baseline_summary = _baseline_summary(evaluation_result, baseline)

    if best_so_far and prior_scores:
        winning_reasons.append("Ranks at or above all prior attempts for this idea.")
    elif prior_scores:
        limiting_factors.append("Does not exceed the best prior attempt for this idea.")

    return AttemptRankingResult(
        idea_id=evaluation_result.idea_id,
        attempt_id=evaluation_result.attempt_id,
        baseline_id=baseline.baseline_id,
        promotion_score=round(current_score, 4),
        rank_label=rank_label,
        baseline_comparison_summary=baseline_summary,
        prior_attempt_comparison_summary=prior_summary,
        winning_reasons=winning_reasons,
        limiting_factors=limiting_factors,
        prior_attempts_considered=[attempt_id for attempt_id, _, _ in prior_scores],
        best_so_far=best_so_far,
    )


def _load_prior_attempt_evaluations(*, idea_dir: Path, status: dict, latest_attempt_id: str) -> list[tuple[str, EvaluationResult]]:
    prior: list[tuple[str, EvaluationResult]] = []
    for attempt in status.get("attempts", []):
        attempt_id = str(attempt.get("attempt_id"))
        if attempt_id == latest_attempt_id:
            continue
        path = idea_dir / "reports" / f"{attempt_id}-evaluation.json"
        if not path.exists():
            continue
        prior.append((attempt_id, EvaluationResult.from_dict(read_json(path))))
    return prior


def _score_evaluation_result(payload: EvaluationResult) -> float:
    recommendation_base = {
        "promote": 400.0,
        "continue_with_caution": 250.0,
        "rerun_with_more_budget": 125.0,
        "stop": 0.0,
    }[payload.overall_recommendation]
    score = recommendation_base
    totals = payload.comparison_totals
    score += totals.get("clean_pass_count", 0) * 100.0
    score += totals.get("promote_count", 0) * 40.0
    score += totals.get("continue_with_caution_count", 0) * 10.0
    score -= totals.get("repaired_pass_count", 0) * 25.0
    score -= totals.get("warning_count", 0) * 30.0
    score -= totals.get("failed_count", 0) * 75.0
    for phase in payload.phase_summaries:
        if phase.delta_vs_baseline is None:
            continue
        score -= phase.delta_vs_baseline * 10.0
        score -= len(phase.caution_flags) * 2.0
        score -= phase.repair_count * 3.0
    return score


def _prior_attempt_summary(current_score: float, prior_scores: list[tuple[str, EvaluationResult, float]]) -> str:
    if not prior_scores:
        return "No prior attempts were available for same-idea ranking."
    best_attempt_id, _best_result, best_score = max(prior_scores, key=lambda item: item[2])
    if current_score > best_score:
        return f"Latest attempt is the best so far; it outranks {best_attempt_id}."
    if current_score == best_score:
        return f"Latest attempt is tied with the current best attempt, {best_attempt_id}."
    return f"Latest attempt regressed relative to best prior attempt {best_attempt_id}."


def _rank_label(payload: EvaluationResult, best_so_far: bool) -> str:
    if payload.overall_recommendation == "stop":
        return "noncompetitive"
    if payload.overall_recommendation == "rerun_with_more_budget":
        return "caution"
    if payload.overall_recommendation == "continue_with_caution":
        return "improving" if best_so_far else "caution"
    return "leading" if best_so_far else "improving"


def _baseline_summary(payload: EvaluationResult, baseline: BaselineDefinition) -> str:
    promotable = payload.comparison_totals.get("promote_count", 0)
    repaired = payload.comparison_totals.get("repaired_pass_count", 0)
    warnings = payload.comparison_totals.get("warning_count", 0)
    return (
        f"Compared against baseline {baseline.baseline_id} ({baseline.family}); "
        f"promote_count={promotable}, repaired_pass_count={repaired}, warning_count={warnings}."
    )
