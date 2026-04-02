from __future__ import annotations

from auto_llm_innovator.datasets.models import DatasetDefinition, DatasetPhasePreset


_DATASET_DEFINITIONS: tuple[DatasetDefinition, ...] = (
    DatasetDefinition(
        dataset_id="synthetic-shapes",
        label="Synthetic Shapes",
        description="Toy tensors and short token snippets for shape, loss, and interface validation.",
        supported_capabilities=["shape_validation", "loss_validation", "smoke_runtime_sanity"],
        availability="bundled_synthetic",
        license_tier="internal_only",
        phase_presets=[
            DatasetPhasePreset(
                phase="smoke",
                dataset_name="synthetic-shapes",
                description="Toy tensors and token snippets for shape and loss validation.",
                default_dataset_slice="tiny",
                default_target_tokens=100_000,
                curriculum_notes=["Use only the smallest deterministic sanity subset for smoke validation."],
            )
        ],
    ),
    DatasetDefinition(
        dataset_id="small-curated-corpus",
        label="Small Curated Corpus",
        description="Compact curated corpus for early learnability checks and rapid baseline comparisons.",
        supported_capabilities=["learnability_check", "baseline_comparison", "small_phase_training"],
        availability="internal_placeholder",
        license_tier="internal_only",
        phase_presets=[
            DatasetPhasePreset(
                phase="small",
                dataset_name="small-curated-corpus",
                description="Small corpus for fast learnability checks.",
                default_dataset_slice="curated",
                default_target_tokens=5_000_000,
                curriculum_notes=["Prefer compact curated slices before escalating to larger corpora."],
            )
        ],
    ),
    DatasetDefinition(
        dataset_id="production-like-corpus",
        label="Production-Like Corpus",
        description="Large internal placeholder corpus for the highest admitted training budget.",
        supported_capabilities=["full_budget_training", "promotion_candidate_evaluation", "broad_eval_readiness"],
        availability="internal_placeholder",
        license_tier="internal_only",
        phase_presets=[
            DatasetPhasePreset(
                phase="full",
                dataset_name="production-like-corpus",
                description="Large corpus placeholder for full-scale training evaluation.",
                default_dataset_slice="full",
                default_target_tokens=500_000_000,
                curriculum_notes=["Escalate to the largest admitted corpus only after smaller phases look promising."],
            )
        ],
    ),
)


def dataset_definitions() -> list[DatasetDefinition]:
    return list(_DATASET_DEFINITIONS)


def dataset_definition_by_id(dataset_id: str) -> DatasetDefinition:
    for definition in _DATASET_DEFINITIONS:
        if definition.dataset_id == dataset_id:
            return definition
    raise KeyError(f"Unknown dataset definition '{dataset_id}'.")


def default_dataset_definition_for_phase(phase: str) -> DatasetDefinition:
    for definition in _DATASET_DEFINITIONS:
        for preset in definition.phase_presets:
            if preset.phase == phase:
                return definition
    raise KeyError(f"No default dataset definition found for phase '{phase}'.")


def dataset_plan_for_phase(phase: str) -> dict:
    definition = default_dataset_definition_for_phase(phase)
    preset = definition.preset_for_phase(phase)
    return {
        "dataset_name": preset.dataset_name,
        "description": preset.description,
        "target_tokens": preset.default_target_tokens,
    }
