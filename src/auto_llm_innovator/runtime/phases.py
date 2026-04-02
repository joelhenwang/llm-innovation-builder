from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class RuntimePhaseSettings:
    max_steps: int
    max_wall_time_seconds: int
    sequence_length: int
    batch_size: int
    checkpoint_every_steps: int
    resume_enabled: bool
    evaluation_scope: str
    dataset_slice: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_runtime_settings_for_phase(phase: str) -> RuntimePhaseSettings:
    defaults = {
        "smoke": RuntimePhaseSettings(
            max_steps=2,
            max_wall_time_seconds=60,
            sequence_length=8,
            batch_size=2,
            checkpoint_every_steps=0,
            resume_enabled=False,
            evaluation_scope="minimal",
            dataset_slice="tiny",
        ),
        "small": RuntimePhaseSettings(
            max_steps=6,
            max_wall_time_seconds=300,
            sequence_length=12,
            batch_size=3,
            checkpoint_every_steps=3,
            resume_enabled=True,
            evaluation_scope="standard",
            dataset_slice="curated",
        ),
        "full": RuntimePhaseSettings(
            max_steps=12,
            max_wall_time_seconds=900,
            sequence_length=16,
            batch_size=4,
            checkpoint_every_steps=4,
            resume_enabled=True,
            evaluation_scope="full",
            dataset_slice="full",
        ),
    }
    return defaults[phase]
