import json
from pathlib import Path

from auto_llm_innovator.evaluation import EvaluationResult, PhaseEvaluationSummary, load_baseline_definition
from auto_llm_innovator.tracking.ranking import build_attempt_ranking


def _baseline(tmp_path: Path):
    baseline_dir = tmp_path / "baselines" / "internal_reference"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    path = baseline_dir / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "baseline_id": "internal-reference-v1",
                "reference_metrics": {
                    "smoke.loss": 6.0,
                    "small.val_loss": 4.2,
                },
            }
        ),
        encoding="utf-8",
    )
    return load_baseline_definition(path)


def _evaluation_result(
    *,
    idea_id: str,
    attempt_id: str,
    overall_recommendation: str,
    phase_status: str,
    phase_recommendation: str,
    delta_vs_baseline: float,
    repair_count: int = 0,
    caution_flags: list[str] | None = None,
) -> EvaluationResult:
    caution_flags = caution_flags or []
    return EvaluationResult(
        idea_id=idea_id,
        attempt_id=attempt_id,
        baseline_id="internal-reference-v1",
        overall_recommendation=overall_recommendation,
        overall_summary="summary",
        phase_summaries=[
            PhaseEvaluationSummary(
                phase="smoke",
                phase_status=phase_status,
                recommendation=phase_recommendation,
                summary="phase summary",
                caution_flags=caution_flags,
                observed_loss=5.9,
                baseline_loss=6.0,
                delta_vs_baseline=delta_vs_baseline,
                repair_count=repair_count,
                stop_reason="max_steps_reached" if phase_status == "passed" else "max_wall_time_reached",
            )
        ],
        comparison_totals={
            "phases_evaluated": 1,
            "promote_count": 1 if phase_recommendation == "promote" else 0,
            "continue_with_caution_count": 1 if phase_recommendation == "continue_with_caution" else 0,
            "rerun_with_more_budget_count": 1 if phase_recommendation == "rerun_with_more_budget" else 0,
            "stop_count": 1 if phase_recommendation == "stop" else 0,
            "clean_pass_count": 1 if phase_status == "passed" and repair_count == 0 else 0,
            "repaired_pass_count": 1 if phase_status == "passed" and repair_count > 0 else 0,
            "warning_count": 1 if phase_status == "passed_with_warnings" else 0,
            "failed_count": 1 if phase_status == "failed" else 0,
        },
        planned_ablations=[],
    )


def _write_evaluation(idea_dir: Path, result: EvaluationResult) -> None:
    reports_dir = idea_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"{result.attempt_id}-evaluation.json").write_text(json.dumps(result.to_dict()), encoding="utf-8")


def test_clean_promotable_attempt_outranks_repaired_attempt(tmp_path: Path):
    idea_dir = tmp_path / "ideas" / "idea-0001"
    idea_dir.mkdir(parents=True, exist_ok=True)
    baseline = _baseline(tmp_path)
    prior = _evaluation_result(
        idea_id="idea-0001",
        attempt_id="attempt-0001",
        overall_recommendation="continue_with_caution",
        phase_status="passed",
        phase_recommendation="continue_with_caution",
        delta_vs_baseline=-0.2,
        repair_count=1,
        caution_flags=["repaired_pass"],
    )
    latest = _evaluation_result(
        idea_id="idea-0001",
        attempt_id="attempt-0002",
        overall_recommendation="promote",
        phase_status="passed",
        phase_recommendation="promote",
        delta_vs_baseline=-0.1,
    )
    _write_evaluation(idea_dir, prior)
    ranking = build_attempt_ranking(
        idea_dir=idea_dir,
        baseline=baseline,
        evaluation_result=latest,
        status={"attempts": [{"attempt_id": "attempt-0001"}, {"attempt_id": "attempt-0002"}]},
    )

    assert ranking.best_so_far is True
    assert ranking.rank_label == "leading"


def test_warning_only_attempt_outranks_failed_stop_attempt(tmp_path: Path):
    idea_dir = tmp_path / "ideas" / "idea-0002"
    idea_dir.mkdir(parents=True, exist_ok=True)
    baseline = _baseline(tmp_path)
    prior = _evaluation_result(
        idea_id="idea-0002",
        attempt_id="attempt-0001",
        overall_recommendation="stop",
        phase_status="failed",
        phase_recommendation="stop",
        delta_vs_baseline=0.5,
    )
    latest = _evaluation_result(
        idea_id="idea-0002",
        attempt_id="attempt-0002",
        overall_recommendation="rerun_with_more_budget",
        phase_status="passed_with_warnings",
        phase_recommendation="rerun_with_more_budget",
        delta_vs_baseline=-0.1,
        caution_flags=["budget_limited"],
    )
    _write_evaluation(idea_dir, prior)
    ranking = build_attempt_ranking(
        idea_dir=idea_dir,
        baseline=baseline,
        evaluation_result=latest,
        status={"attempts": [{"attempt_id": "attempt-0001"}, {"attempt_id": "attempt-0002"}]},
    )

    assert ranking.best_so_far is True
    assert ranking.rank_label == "caution"


def test_latest_attempt_is_marked_regressed_when_prior_is_better(tmp_path: Path):
    idea_dir = tmp_path / "ideas" / "idea-0003"
    idea_dir.mkdir(parents=True, exist_ok=True)
    baseline = _baseline(tmp_path)
    prior = _evaluation_result(
        idea_id="idea-0003",
        attempt_id="attempt-0001",
        overall_recommendation="promote",
        phase_status="passed",
        phase_recommendation="promote",
        delta_vs_baseline=-0.3,
    )
    latest = _evaluation_result(
        idea_id="idea-0003",
        attempt_id="attempt-0002",
        overall_recommendation="continue_with_caution",
        phase_status="passed",
        phase_recommendation="continue_with_caution",
        delta_vs_baseline=-0.1,
        repair_count=1,
        caution_flags=["repaired_pass"],
    )
    _write_evaluation(idea_dir, prior)
    ranking = build_attempt_ranking(
        idea_dir=idea_dir,
        baseline=baseline,
        evaluation_result=latest,
        status={"attempts": [{"attempt_id": "attempt-0001"}, {"attempt_id": "attempt-0002"}]},
    )

    assert ranking.best_so_far is False
    assert "regressed" in ranking.prior_attempt_comparison_summary
