from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class IdeaModelConfig:
    idea_id: str
    architecture_name: str
    target_parameters: int
    hidden_size: int
    num_layers: int
    num_heads: int
    tokenizer: str = "gpt2"
    borrowed_mechanisms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PhaseResult:
    idea_id: str
    attempt_id: str
    phase: str
    status: str
    key_metrics: dict[str, float]
    failure_signals: list[str]
    artifacts_produced: list[str]
    reviewer_notes: list[str]
    next_action_recommendation: str
    consumed_budget: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
