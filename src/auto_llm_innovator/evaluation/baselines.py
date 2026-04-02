from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from auto_llm_innovator.filesystem import read_json


@dataclass(slots=True)
class BaselineMetricTarget:
    phase: str
    metric_name: str
    target_value: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BaselineMetricTarget":
        return cls(
            phase=str(payload["phase"]),
            metric_name=str(payload["metric_name"]),
            target_value=float(payload["target_value"]),
        )


@dataclass(slots=True)
class BaselineDefinition:
    baseline_id: str
    family: str
    label: str
    tokenizer: str | None = None
    description: str | None = None
    metric_targets: list[BaselineMetricTarget] = field(default_factory=list)
    reliability_expectations: dict[str, str] = field(default_factory=dict)
    practicality_expectations: dict[str, str] = field(default_factory=dict)
    hardware_assumptions: dict[str, Any] = field(default_factory=dict)
    token_budget_assumptions: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BaselineDefinition":
        return cls(
            baseline_id=str(payload["baseline_id"]),
            family=str(payload.get("family", "internal_reference")),
            label=str(payload.get("label") or payload.get("description") or payload["baseline_id"]),
            tokenizer=str(payload["tokenizer"]) if payload.get("tokenizer") is not None else None,
            description=str(payload["description"]) if payload.get("description") is not None else None,
            metric_targets=[BaselineMetricTarget.from_dict(item) for item in payload.get("metric_targets", [])],
            reliability_expectations={str(k): str(v) for k, v in payload.get("reliability_expectations", {}).items()},
            practicality_expectations={str(k): str(v) for k, v in payload.get("practicality_expectations", {}).items()},
            hardware_assumptions=dict(payload.get("hardware_assumptions", {})),
            token_budget_assumptions=dict(payload.get("token_budget_assumptions", {})),
        )

    def reference_metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for target in self.metric_targets:
            key = f"{target.phase}.{target.metric_name}"
            metrics[key] = float(target.target_value)
        return metrics


def load_baseline_definition(path: Path) -> BaselineDefinition:
    payload = read_json(path)
    if "metric_targets" in payload:
        return BaselineDefinition.from_dict(payload)

    metric_targets = [
        BaselineMetricTarget(
            phase=str(metric_key).split(".", 1)[0],
            metric_name=str(metric_key).split(".", 1)[1],
            target_value=float(metric_value),
        )
        for metric_key, metric_value in payload.get("reference_metrics", {}).items()
    ]
    return BaselineDefinition(
        baseline_id=str(payload["baseline_id"]),
        family=str(payload.get("family", path.parent.name if path.parent.name else "internal_reference")),
        label=str(payload.get("label") or payload.get("description") or payload["baseline_id"]),
        tokenizer=str(payload["tokenizer"]) if payload.get("tokenizer") is not None else None,
        description=str(payload["description"]) if payload.get("description") is not None else None,
        metric_targets=metric_targets,
        reliability_expectations={str(k): str(v) for k, v in payload.get("reliability_expectations", {}).items()},
        practicality_expectations={str(k): str(v) for k, v in payload.get("practicality_expectations", {}).items()},
        hardware_assumptions=dict(payload.get("hardware_assumptions", {})),
        token_budget_assumptions=dict(payload.get("token_budget_assumptions", {})),
    )
