from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, TypeVar


T = TypeVar("T")


def _from_dict_list(payloads: list[dict[str, Any]], cls: type[T]) -> list[T]:
    return [cls.from_dict(payload) for payload in payloads]


@dataclass(slots=True)
class StateSemantics:
    has_recurrent_state: bool
    has_external_memory: bool
    has_cache_path: bool
    summary: str
    assumption_source: str = "compiler_default"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StateSemantics":
        return cls(**payload)


@dataclass(slots=True)
class ArchitecturePlan:
    pattern_label: str
    mechanism_summary: str
    module_graph_summary: str
    state_semantics: StateSemantics
    assumption_source: str = "compiler_default"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ArchitecturePlan":
        return cls(
            pattern_label=payload["pattern_label"],
            mechanism_summary=payload["mechanism_summary"],
            module_graph_summary=payload["module_graph_summary"],
            state_semantics=StateSemantics.from_dict(payload["state_semantics"]),
            assumption_source=payload.get("assumption_source", "compiler_default"),
        )


@dataclass(slots=True)
class DesignModule:
    name: str
    kind: str
    purpose: str
    inputs: list[str]
    outputs: list[str]
    depends_on: list[str] = field(default_factory=list)
    required: bool = True
    assumption_source: str = "compiler_default"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DesignModule":
        return cls(**payload)


@dataclass(slots=True)
class TensorInterface:
    name: str
    semantic_role: str
    producer: str
    consumer: str
    shape_notes: str
    assumption_source: str = "compiler_default"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TensorInterface":
        return cls(**payload)


@dataclass(slots=True)
class TrainingStagePlan:
    stage: str
    objective: str
    dataset_name: str
    dataset_description: str
    target_tokens: int
    success_checks: list[str]
    assumption_source: str = "compiler_default"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainingStagePlan":
        return cls(**payload)


@dataclass(slots=True)
class EvaluationTask:
    name: str
    description: str
    metrics: list[str]
    comparison_targets: list[str]
    phase: str
    assumption_source: str = "compiler_default"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvaluationTask":
        return cls(**payload)


@dataclass(slots=True)
class AblationPlan:
    name: str
    description: str
    target_modules: list[str]
    phase: str
    assumption_source: str = "compiler_default"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AblationPlan":
        return cls(**payload)


@dataclass(slots=True)
class FailureCriterion:
    name: str
    description: str
    focus_area: str
    target_modules: list[str] = field(default_factory=list)
    assumption_source: str = "compiler_default"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FailureCriterion":
        return cls(**payload)


@dataclass(slots=True)
class ImplementationMilestone:
    name: str
    description: str
    depends_on: list[str] = field(default_factory=list)
    assumption_source: str = "compiler_default"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ImplementationMilestone":
        return cls(**payload)


@dataclass(slots=True)
class CompatibilityProjection:
    raw_brief: str
    normalized_brief: str
    hypothesis: str
    novelty_claims: list[str]
    training_curriculum_outline: list[str]
    inspirations_consulted: list[str]
    evaluation_intent: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CompatibilityProjection":
        return cls(**payload)


@dataclass(slots=True)
class DesignIR:
    idea_id: str
    title: str
    bundle_kind: str
    source_candidate_ids: list[str]
    source_titles: list[str]
    tokenizer_requirement: str
    parameter_cap: int
    compute_budget_hint: str
    known_constraints: list[str]
    architecture: ArchitecturePlan
    modules: list[DesignModule]
    tensor_interfaces: list[TensorInterface]
    training_plan: list[TrainingStagePlan]
    evaluation_plan: list[EvaluationTask]
    ablation_plan: list[AblationPlan]
    failure_criteria: list[FailureCriterion]
    implementation_milestones: list[ImplementationMilestone]
    compatibility_projection: CompatibilityProjection

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DesignIR":
        return cls(
            idea_id=payload["idea_id"],
            title=payload["title"],
            bundle_kind=payload["bundle_kind"],
            source_candidate_ids=list(payload.get("source_candidate_ids", [])),
            source_titles=list(payload.get("source_titles", [])),
            tokenizer_requirement=payload["tokenizer_requirement"],
            parameter_cap=payload["parameter_cap"],
            compute_budget_hint=payload.get("compute_budget_hint", ""),
            known_constraints=list(payload.get("known_constraints", [])),
            architecture=ArchitecturePlan.from_dict(payload["architecture"]),
            modules=_from_dict_list(payload.get("modules", []), DesignModule),
            tensor_interfaces=_from_dict_list(payload.get("tensor_interfaces", []), TensorInterface),
            training_plan=_from_dict_list(payload.get("training_plan", []), TrainingStagePlan),
            evaluation_plan=_from_dict_list(payload.get("evaluation_plan", []), EvaluationTask),
            ablation_plan=_from_dict_list(payload.get("ablation_plan", []), AblationPlan),
            failure_criteria=_from_dict_list(payload.get("failure_criteria", []), FailureCriterion),
            implementation_milestones=_from_dict_list(
                payload.get("implementation_milestones", []), ImplementationMilestone
            ),
            compatibility_projection=CompatibilityProjection.from_dict(payload["compatibility_projection"]),
        )
