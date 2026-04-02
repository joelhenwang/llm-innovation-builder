from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


class BundleKinds:
    FREE_TEXT = "free_text"
    RESEARCH_CANDIDATE = "research_candidate"
    RESEARCH_MIX = "research_mix"


@dataclass(slots=True)
class ResearchIdeaBundle:
    bundle_kind: str
    source_artifact_kind: str
    source_candidate_ids: list[str]
    source_titles: list[str]
    title: str
    mechanism_summary: str
    novelty_rationale: str
    implementation_requirements: list[str] = field(default_factory=list)
    known_constraints: list[str] = field(default_factory=list)
    dataset_requirements: list[str] = field(default_factory=list)
    evaluation_targets: list[str] = field(default_factory=list)
    ablation_ideas: list[str] = field(default_factory=list)
    expected_failure_modes: list[str] = field(default_factory=list)
    compute_budget_hint: str = ""
    tokenizer_requirement: str = ""
    raw_payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ResearchIdeaBundle":
        return cls(**payload)
