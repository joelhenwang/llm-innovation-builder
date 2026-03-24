from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from auto_llm_innovator.constants import GPT2_TOKENIZER, PARAMETER_CAP


@dataclass(slots=True)
class IdeaSpec:
    idea_id: str
    raw_brief: str
    normalized_brief: str
    hypothesis: str
    novelty_claims: list[str]
    forbidden_fallback_patterns: list[str]
    intended_learning_objective: str
    estimated_parameter_budget: int = PARAMETER_CAP
    training_curriculum_outline: list[str] = field(default_factory=list)
    inspirations_consulted: list[str] = field(default_factory=list)
    tokenizer: str = GPT2_TOKENIZER
    intended_model_target: str = "general-purpose autoregressive language model"
    evaluation_intent: str = "compare against internal baseline, prior runs, and public references"
    borrowed_mechanisms: list[str] = field(default_factory=list)
    public_references: list[str] = field(default_factory=list)
    creativity_directive: str = (
        "Be original, avoid generic transformer defaults, and justify any borrowed mechanisms."
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "IdeaSpec":
        return cls(**payload)
