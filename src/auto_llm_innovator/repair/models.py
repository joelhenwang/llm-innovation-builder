from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


FailureSource = Literal["preflight", "runtime"]
FailureCategory = Literal[
    "package_import_failure",
    "plugin_contract_failure",
    "invalid_runtime_settings",
    "runtime_output_shape_failure",
    "checkpoint_failure",
    "evaluation_contract_failure",
    "model_construction_failure",
    "oom_like_failure",
    "nan_loss_like_failure",
    "unknown_runtime_failure",
]


@dataclass(slots=True)
class FailureClassification:
    source: FailureSource
    category: FailureCategory
    repairable: bool
    summary: str
    stop_reason: str | None = None
    next_action_recommendation: str | None = None
    failure_signals: list[str] = field(default_factory=list)
    failing_modules: list[str] = field(default_factory=list)
    failing_files: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RepairAttempt:
    attempt_index: int
    source: FailureSource
    category: FailureCategory
    strategy: str
    target_files: list[str]
    before_snapshot_dir: str
    after_snapshot_dir: str
    diff_path: str
    rationale_path: str
    outcome: str
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RepairLoopResult:
    final_phase_result: dict[str, Any] | None = None
    repair_attempted: bool = False
    repair_count: int = 0
    repairs_remaining: int = 0
    failure_classification: FailureClassification | None = None
    artifact_paths: list[str] = field(default_factory=list)
    history: list[RepairAttempt] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.failure_classification is not None:
            payload["failure_classification"] = self.failure_classification.to_dict()
        return payload
