from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class DatasetPhasePreset:
    phase: str
    dataset_name: str
    description: str
    default_dataset_slice: str
    default_target_tokens: int
    curriculum_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DatasetDefinition:
    dataset_id: str
    label: str
    description: str
    supported_capabilities: list[str]
    availability: str = "internal_placeholder"
    license_tier: str = "internal_only"
    phase_presets: list[DatasetPhasePreset] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["phase_presets"] = [preset.to_dict() for preset in self.phase_presets]
        return payload

    def preset_for_phase(self, phase: str) -> DatasetPhasePreset:
        for preset in self.phase_presets:
            if preset.phase == phase:
                return preset
        raise KeyError(f"Dataset definition '{self.dataset_id}' does not define phase '{phase}'.")


@dataclass(slots=True)
class DatasetPlan:
    idea_id: str
    phase: str
    dataset_id: str
    dataset_name: str
    label: str
    description: str
    target_tokens: int
    dataset_slice: str
    objective: str
    admission_status: str
    executable: bool
    curriculum_notes: list[str] = field(default_factory=list)
    supported_capabilities: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    baseline_token_budget: int | None = None
    availability: str | None = None
    license_tier: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
