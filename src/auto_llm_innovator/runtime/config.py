from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any

from auto_llm_innovator.constants import GPT2_TOKENIZER
from auto_llm_innovator.design_ir.models import DesignIR, EvaluationTask, FailureCriterion, TrainingStagePlan
from auto_llm_innovator.runtime.phases import RuntimePhaseSettings, default_runtime_settings_for_phase


@dataclass(slots=True)
class RuntimePluginConfig:
    architecture_name: str
    tokenizer: str
    required_modules: list[str]
    optional_modules: list[str]
    has_recurrent_state: bool
    has_external_memory: bool
    has_cache_path: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RuntimeDatasetConfig:
    dataset_name: str
    description: str
    target_tokens: int
    dataset_slice: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RuntimeOptimizerConfig:
    name: str = "AdamW"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RuntimeSchedulerConfig:
    name: str = "constant"
    warmup_steps: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RuntimeCheckpointConfig:
    enabled: bool = True
    resume: bool = True
    every_n_steps: int = 0
    filename: str = "checkpoint.json"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RuntimeLoggingConfig:
    metrics_filename: str = "metrics.jsonl"
    summary_filename: str = "runtime-summary.json"
    evaluation_filename: str = "evaluation-report.json"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RuntimeCheck:
    name: str
    description: str
    focus_area: str
    target_modules: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RuntimeEvaluationConfig:
    tasks: list[EvaluationTask]

    def to_dict(self) -> dict[str, Any]:
        return {"tasks": [task.to_dict() for task in self.tasks]}


@dataclass(slots=True)
class RuntimePhaseConfig:
    idea_id: str
    title: str
    phase: str
    objective: str
    success_checks: list[str]
    target_parameters: int
    prefer_rocm: bool
    novelty_claims: list[str]
    seed: int
    precision: str
    gradient_accumulation_steps: int
    settings: RuntimePhaseSettings
    plugin: RuntimePluginConfig
    dataset: RuntimeDatasetConfig
    optimizer: RuntimeOptimizerConfig
    scheduler: RuntimeSchedulerConfig
    checkpoints: RuntimeCheckpointConfig
    logging: RuntimeLoggingConfig
    runtime_checks: list[RuntimeCheck]
    evaluation: RuntimeEvaluationConfig
    ablation_names: list[str]
    resource_plan: dict[str, Any] = field(default_factory=dict)
    dataset_plan: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["evaluation"] = self.evaluation.to_dict()
        return payload


def compile_runtime_phase_config(
    design_ir: DesignIR, phase_config: dict[str, Any], *, attempt_id: str, phase: str
) -> RuntimePhaseConfig:
    stage = _require_training_stage(design_ir, phase)
    settings = _compile_phase_settings(phase, phase_config.get("runtime"))
    dataset_payload = phase_config["dataset"]
    return RuntimePhaseConfig(
        idea_id=design_ir.idea_id,
        title=design_ir.title,
        phase=phase,
        objective=stage.objective,
        success_checks=list(stage.success_checks),
        target_parameters=int(phase_config["target_parameters"]),
        prefer_rocm=bool(phase_config.get("prefer_rocm", False)),
        novelty_claims=list(phase_config.get("novelty_claims", [])),
        seed=_stable_seed(design_ir.idea_id, attempt_id, phase),
        precision="fp32",
        gradient_accumulation_steps=1,
        settings=settings,
        plugin=RuntimePluginConfig(
            architecture_name=design_ir.architecture.pattern_label,
            tokenizer=design_ir.tokenizer_requirement or GPT2_TOKENIZER,
            required_modules=[module.name for module in design_ir.modules if module.required],
            optional_modules=[module.name for module in design_ir.modules if not module.required],
            has_recurrent_state=design_ir.architecture.state_semantics.has_recurrent_state,
            has_external_memory=design_ir.architecture.state_semantics.has_external_memory,
            has_cache_path=design_ir.architecture.state_semantics.has_cache_path,
        ),
        dataset=RuntimeDatasetConfig(
            dataset_name=dataset_payload["dataset_name"],
            description=dataset_payload["description"],
            target_tokens=int(dataset_payload["target_tokens"]),
            dataset_slice=settings.dataset_slice,
        ),
        optimizer=RuntimeOptimizerConfig(),
        scheduler=RuntimeSchedulerConfig(),
        checkpoints=RuntimeCheckpointConfig(resume=settings.resume_enabled, every_n_steps=settings.checkpoint_every_steps),
        logging=RuntimeLoggingConfig(summary_filename=f"{phase}-summary.json"),
        runtime_checks=_compile_runtime_checks(design_ir.failure_criteria),
        evaluation=RuntimeEvaluationConfig(tasks=_evaluation_tasks_for_phase(design_ir, phase, settings.evaluation_scope)),
        ablation_names=[ablation.name for ablation in design_ir.ablation_plan if ablation.phase == phase],
        resource_plan=dict(phase_config.get("resource_plan", {})),
        dataset_plan=dict(phase_config.get("dataset_plan", {})),
    )


def _require_training_stage(design_ir: DesignIR, phase: str) -> TrainingStagePlan:
    for stage in design_ir.training_plan:
        if stage.stage == phase:
            return stage
    raise ValueError(f"DesignIR training plan is missing runtime stage '{phase}'.")


def _compile_runtime_checks(criteria: list[FailureCriterion]) -> list[RuntimeCheck]:
    checks = [
        RuntimeCheck(
            name="plugin_contract",
            description="Generated plugin exposes the minimum runtime hooks required by the shared harness.",
            focus_area="plugin_contract",
            target_modules=[],
        )
    ]
    checks.extend(
        RuntimeCheck(
            name=criterion.name,
            description=criterion.description,
            focus_area=criterion.focus_area,
            target_modules=list(criterion.target_modules),
        )
        for criterion in criteria
    )
    return checks


def _stable_seed(idea_id: str, attempt_id: str, phase: str) -> int:
    digest = hashlib.sha256(f"{idea_id}:{attempt_id}:{phase}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _compile_phase_settings(phase: str, payload: dict[str, Any] | None) -> RuntimePhaseSettings:
    defaults = default_runtime_settings_for_phase(phase)
    runtime_payload = payload or {}
    settings = RuntimePhaseSettings(
        max_steps=int(runtime_payload.get("max_steps", defaults.max_steps)),
        max_wall_time_seconds=int(runtime_payload.get("max_wall_time_seconds", defaults.max_wall_time_seconds)),
        sequence_length=int(runtime_payload.get("sequence_length", defaults.sequence_length)),
        batch_size=int(runtime_payload.get("batch_size", defaults.batch_size)),
        checkpoint_every_steps=int(runtime_payload.get("checkpoint_every_steps", defaults.checkpoint_every_steps)),
        resume_enabled=bool(runtime_payload.get("resume_enabled", defaults.resume_enabled)),
        evaluation_scope=str(runtime_payload.get("evaluation_scope", defaults.evaluation_scope)),
        dataset_slice=str(runtime_payload.get("dataset_slice", defaults.dataset_slice)),
    )
    _validate_phase_settings(phase, settings)
    return settings


def _validate_phase_settings(phase: str, settings: RuntimePhaseSettings) -> None:
    if settings.max_steps <= 0:
        raise ValueError(f"Phase '{phase}' must configure runtime.max_steps > 0.")
    if settings.max_wall_time_seconds < 0:
        raise ValueError(f"Phase '{phase}' must configure runtime.max_wall_time_seconds >= 0.")
    if settings.sequence_length <= 0:
        raise ValueError(f"Phase '{phase}' must configure runtime.sequence_length > 0.")
    if settings.batch_size <= 0:
        raise ValueError(f"Phase '{phase}' must configure runtime.batch_size > 0.")
    if settings.checkpoint_every_steps < 0:
        raise ValueError(f"Phase '{phase}' must configure runtime.checkpoint_every_steps >= 0.")
    if settings.evaluation_scope not in {"minimal", "standard", "full"}:
        raise ValueError(
            f"Phase '{phase}' must configure runtime.evaluation_scope as one of: minimal, standard, full."
        )


def _evaluation_tasks_for_phase(design_ir: DesignIR, phase: str, evaluation_scope: str) -> list[EvaluationTask]:
    evaluation_tasks = [task for task in design_ir.evaluation_plan if task.phase == phase]
    if not evaluation_tasks and phase == "smoke":
        evaluation_tasks = [
            EvaluationTask(
                name="smoke_validation",
                description="Validate runtime wiring and loss production for the synthetic harness.",
                metrics=["loss"],
                comparison_targets=["internal_baseline"],
                phase="smoke",
                assumption_source="phase6_runtime_default",
            )
        ]
    if evaluation_scope == "minimal":
        return evaluation_tasks[:1]
    return evaluation_tasks
