from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class EvaluationSignal:
    name: str
    value: Any
    status: str
    summary: str
    source_artifact: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvaluationSignal":
        return cls(**payload)


@dataclass(slots=True)
class PhaseEvaluationSummary:
    phase: str
    phase_status: str
    recommendation: str
    summary: str
    caution_flags: list[str] = field(default_factory=list)
    reliability_signals: list[EvaluationSignal] = field(default_factory=list)
    quality_signals: list[EvaluationSignal] = field(default_factory=list)
    practicality_signals: list[EvaluationSignal] = field(default_factory=list)
    observed_loss: float | None = None
    baseline_loss: float | None = None
    delta_vs_baseline: float | None = None
    evaluation_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    repair_count: int = 0
    stop_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PhaseEvaluationSummary":
        return cls(
            phase=payload["phase"],
            phase_status=payload["phase_status"],
            recommendation=payload["recommendation"],
            summary=payload["summary"],
            caution_flags=list(payload.get("caution_flags", [])),
            reliability_signals=[
                EvaluationSignal.from_dict(item) for item in payload.get("reliability_signals", [])
            ],
            quality_signals=[EvaluationSignal.from_dict(item) for item in payload.get("quality_signals", [])],
            practicality_signals=[
                EvaluationSignal.from_dict(item) for item in payload.get("practicality_signals", [])
            ],
            observed_loss=payload.get("observed_loss"),
            baseline_loss=payload.get("baseline_loss"),
            delta_vs_baseline=payload.get("delta_vs_baseline"),
            evaluation_metrics={
                str(key): {str(metric): float(value) for metric, value in metrics.items()}
                for key, metrics in payload.get("evaluation_metrics", {}).items()
            },
            repair_count=int(payload.get("repair_count", 0)),
            stop_reason=payload.get("stop_reason"),
        )


@dataclass(slots=True)
class EvaluationResult:
    idea_id: str
    attempt_id: str
    baseline_id: str
    overall_recommendation: str
    overall_summary: str
    phase_summaries: list[PhaseEvaluationSummary] = field(default_factory=list)
    comparison_totals: dict[str, int] = field(default_factory=dict)
    planned_ablations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvaluationResult":
        return cls(
            idea_id=payload["idea_id"],
            attempt_id=payload["attempt_id"],
            baseline_id=payload["baseline_id"],
            overall_recommendation=payload["overall_recommendation"],
            overall_summary=payload["overall_summary"],
            phase_summaries=[PhaseEvaluationSummary.from_dict(item) for item in payload.get("phase_summaries", [])],
            comparison_totals={str(key): int(value) for key, value in payload.get("comparison_totals", {}).items()},
            planned_ablations=list(payload.get("planned_ablations", [])),
        )
