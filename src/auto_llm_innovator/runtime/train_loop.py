from __future__ import annotations

import random
import time
from pathlib import Path
from types import ModuleType
from typing import Any

from auto_llm_innovator.design_ir import DesignIR
from auto_llm_innovator.filesystem import ensure_dir, read_json
from auto_llm_innovator.runtime.checkpoints import load_checkpoint, save_checkpoint
from auto_llm_innovator.runtime.config import RuntimePhaseConfig, compile_runtime_phase_config
from auto_llm_innovator.runtime.eval_loop import run_evaluation_tasks
from auto_llm_innovator.runtime.logging import append_metric, write_summary


def run_phase_with_plugin(
    *,
    phase: str,
    idea_dir: str,
    run_dir: str,
    config_path: str,
    attempt_id: str,
    plugin_module: ModuleType | Any,
) -> dict[str, Any]:
    idea_path = Path(idea_dir)
    run_path = ensure_dir(Path(run_dir))
    design_ir = DesignIR.from_dict(read_json(idea_path / "design_ir.json"))
    phase_config = read_json(Path(config_path))
    runtime_config = compile_runtime_phase_config(design_ir, phase_config, attempt_id=attempt_id, phase=phase)

    failure_signals = validate_plugin_contract(plugin_module, runtime_config)
    artifacts: list[str] = []
    reviewer_notes = [
        "Shared runtime harness executed the phase.",
        f"Runtime contract derived from DesignIR for phase '{phase}'.",
    ]
    if failure_signals:
        summary_payload = _failure_summary(runtime_config, attempt_id, failure_signals)
        artifacts.append(str(write_summary(run_path, runtime_config.logging.summary_filename, summary_payload)))
        return {
            "status": "failed",
            "key_metrics": {},
            "failure_signals": failure_signals,
            "artifacts_produced": artifacts,
            "reviewer_notes": reviewer_notes + ["Plugin contract failed before training started."],
            "next_action_recommendation": "repair_plugin_contract",
            "consumed_budget": {
                "requested_parameters": runtime_config.target_parameters,
                "steps": 0,
                "device": "cpu-dry-run",
                "resumed": False,
            },
        }

    build_model = getattr(plugin_module, "build_model")
    model_config_cls = getattr(plugin_module, "ModelConfig", None)
    plugin_descriptor_fn = getattr(plugin_module, "describe_plugin", None)
    evaluation_hook_fn = getattr(plugin_module, "register_evaluation_hooks", None)
    plugin_descriptor = plugin_descriptor_fn() if callable(plugin_descriptor_fn) else {}
    evaluation_hooks = evaluation_hook_fn() if callable(evaluation_hook_fn) else {}

    checkpoint = load_checkpoint(run_path, runtime_config.checkpoints.filename) if runtime_config.checkpoints.resume else None
    resumed = checkpoint is not None
    if resumed:
        reviewer_notes.append("Resumed from existing checkpoint metadata.")

    loss_value, steps_completed, device_label, extra_failure_signals, stop_reason = execute_training(
        runtime_config=runtime_config,
        build_model=build_model,
        model_config_cls=model_config_cls,
        run_path=run_path,
        checkpoint=checkpoint,
        plugin_descriptor=plugin_descriptor,
    )
    failure_signals.extend(extra_failure_signals)

    if runtime_config.checkpoints.enabled:
        checkpoint_path = save_checkpoint(
            run_path,
            runtime_config.checkpoints.filename,
            {
                "idea_id": runtime_config.idea_id,
                "attempt_id": attempt_id,
                "phase": phase,
                "completed_steps": steps_completed,
                "loss": loss_value,
                "device": device_label,
                "stop_reason": stop_reason,
            },
        )
        artifacts.append(str(checkpoint_path))

    evaluation_payload, evaluation_path = run_evaluation_tasks(
        runtime_config,
        run_dir=run_path,
        loss_value=loss_value,
        evaluation_hooks=evaluation_hooks if isinstance(evaluation_hooks, dict) else None,
    )
    artifacts.append(str(evaluation_path))

    summary_payload = {
        "phase": phase,
        "attempt_id": attempt_id,
        "metrics": {
            "loss": round(loss_value, 4),
            "steps": steps_completed,
            "target_parameters": runtime_config.target_parameters,
            "tokenizer": runtime_config.plugin.tokenizer,
        },
        "stop_reason": stop_reason,
        "architecture_name": runtime_config.plugin.architecture_name,
        "runtime": runtime_config.to_dict(),
        "plugin_descriptor": plugin_descriptor,
        "evaluation_summary": evaluation_payload,
        "failure_signals": failure_signals,
    }
    summary_path = write_summary(run_path, runtime_config.logging.summary_filename, summary_payload)
    artifacts.append(str(summary_path))

    success_checks_satisfied = not failure_signals and stop_reason == "max_steps_reached"
    status = _phase_status(success_checks_satisfied, failure_signals, stop_reason)
    reviewer_notes.append(f"Evaluation tasks executed: {len(evaluation_payload['tasks'])}.")
    reviewer_notes.append(f"Success checks tracked: {len(runtime_config.success_checks)}.")
    reviewer_notes.append(f"Phase stop reason: {stop_reason}.")
    if runtime_config.plugin.has_recurrent_state:
        reviewer_notes.append("Runtime enabled recurrent-state validation from DesignIR.")
    if runtime_config.plugin.has_external_memory:
        reviewer_notes.append("Runtime enabled memory-path validation from DesignIR.")
    if runtime_config.plugin.has_cache_path:
        reviewer_notes.append("Runtime enabled cache-path validation from DesignIR.")

    return {
        "status": status,
        "key_metrics": {"loss": round(loss_value, 4), "steps": float(steps_completed)},
        "failure_signals": failure_signals,
        "artifacts_produced": artifacts,
        "reviewer_notes": reviewer_notes,
        "next_action_recommendation": _next_action_recommendation(phase, status),
        "consumed_budget": {
            "requested_parameters": runtime_config.target_parameters,
            "steps": steps_completed,
            "device": device_label,
            "resumed": resumed,
            "stop_reason": stop_reason,
        },
    }


def validate_plugin_contract(plugin_module: ModuleType | Any, runtime_config: RuntimePhaseConfig) -> list[str]:
    failure_signals: list[str] = []
    if not hasattr(plugin_module, "build_model") or not callable(getattr(plugin_module, "build_model", None)):
        failure_signals.append("Plugin contract error: plugin module must expose callable build_model(...).")
    if not hasattr(plugin_module, "ModelConfig"):
        failure_signals.append("Plugin contract error: plugin module must expose ModelConfig for runtime model construction.")

    descriptor_fn = getattr(plugin_module, "describe_plugin", None)
    try:
        descriptor = descriptor_fn() if callable(descriptor_fn) else {}
    except Exception as exc:
        failure_signals.append(f"Plugin contract error: describe_plugin() raised: {exc}")
        descriptor = {}
    if not isinstance(descriptor, dict):
        failure_signals.append("Plugin contract error: describe_plugin() must return a dictionary when provided.")
        descriptor = {}
    declared_modules = set(descriptor.get("module_names", []))
    required_modules = set(runtime_config.plugin.required_modules)
    missing_modules = sorted(required_modules - declared_modules) if declared_modules else []
    if missing_modules:
        failure_signals.append(f"Plugin contract error: missing required modules in descriptor: {', '.join(missing_modules)}.")

    supports = descriptor.get("supports", {}) if isinstance(descriptor, dict) else {}
    if runtime_config.plugin.has_recurrent_state and not bool(supports.get("recurrent_state", False)):
        failure_signals.append("Plugin contract error: DesignIR requires recurrent-state support, but plugin descriptor does not declare it.")
    if runtime_config.plugin.has_external_memory and not bool(supports.get("external_memory", False)):
        failure_signals.append("Plugin contract error: DesignIR requires external-memory support, but plugin descriptor does not declare it.")
    if runtime_config.plugin.has_cache_path and not bool(supports.get("cache_path", False)):
        failure_signals.append("Plugin contract error: DesignIR requires cache-path support, but plugin descriptor does not declare it.")
    return failure_signals


def _failure_summary(runtime_config: RuntimePhaseConfig, attempt_id: str, failure_signals: list[str]) -> dict[str, Any]:
    return {
        "phase": runtime_config.phase,
        "attempt_id": attempt_id,
        "metrics": {},
        "architecture_name": runtime_config.plugin.architecture_name,
        "runtime": runtime_config.to_dict(),
        "failure_signals": failure_signals,
    }


def build_model_instance(build_model: Any, model_config_cls: Any) -> tuple[Any, Any]:
    model_config = model_config_cls() if model_config_cls is not None else None
    model = build_model(model_config)
    return model, model_config


def execute_training(
    *,
    runtime_config: RuntimePhaseConfig,
    build_model: Any,
    model_config_cls: Any,
    run_path: Path,
    checkpoint: dict[str, Any] | None,
    plugin_descriptor: dict[str, Any],
    max_steps: int | None = None,
    write_metrics: bool = True,
) -> tuple[float, int, str, list[str], str]:
    random.seed(runtime_config.seed)
    starting_step = int(checkpoint.get("completed_steps", 0)) if checkpoint else 0
    initial_loss = float(checkpoint.get("loss", _base_loss_for_phase(runtime_config.phase))) if checkpoint else _base_loss_for_phase(runtime_config.phase)
    loss_value = initial_loss
    failure_signals: list[str] = []
    stop_reason = "max_steps_reached"

    try:
        model, _model_config = build_model_instance(build_model, model_config_cls)
    except Exception as exc:
        return initial_loss, starting_step, "cpu-dry-run", [f"Model construction failed: {exc}"], "model_construction_failed"

    steps_limit = runtime_config.settings.max_steps if max_steps is None else min(
        runtime_config.settings.max_steps,
        starting_step + max_steps,
    )
    started_at = time.monotonic()
    steps_completed = starting_step
    step = starting_step - 1
    for step in range(starting_step, steps_limit):
        if time.monotonic() - started_at >= runtime_config.settings.max_wall_time_seconds:
            stop_reason = "max_wall_time_reached"
            break
        batch = synthetic_batch(runtime_config, step)
        forward_output = call_model(
            model,
            batch,
            runtime_config=runtime_config,
            plugin_descriptor=plugin_descriptor,
        )
        if not isinstance(forward_output, dict) or "logits" not in forward_output:
            failure_signals.append("Runtime validation error: model forward() must return a dict containing 'logits'.")
            stop_reason = "runtime_validation_failed"
            break
        logits = forward_output["logits"]
        logits_shape = shape_of(logits)
        if len(logits_shape) < 2:
            failure_signals.append("Runtime validation error: logits must expose at least batch and vocabulary dimensions.")
            stop_reason = "runtime_validation_failed"
            break

        loss_value = max(0.8, round(initial_loss - ((step + 1) * _loss_delta_for_phase(runtime_config.phase)), 4))
        if write_metrics:
            append_metric(
                run_path,
                runtime_config.logging.metrics_filename,
                {
                    "step": step + 1,
                    "loss": loss_value,
                    "seed": runtime_config.seed,
                    "phase": runtime_config.phase,
                    "logits_shape": logits_shape,
                },
            )
        steps_completed = step + 1
        if runtime_config.checkpoints.enabled and runtime_config.checkpoints.every_n_steps > 0:
            if steps_completed % runtime_config.checkpoints.every_n_steps == 0:
                save_checkpoint(
                    run_path,
                    runtime_config.checkpoints.filename,
                    {
                        "idea_id": runtime_config.idea_id,
                        "phase": runtime_config.phase,
                        "completed_steps": steps_completed,
                        "loss": loss_value,
                        "device": "cpu-dry-run",
                        "checkpoint_kind": "periodic",
                    },
                )

    if not failure_signals and stop_reason != "max_wall_time_reached":
        stop_reason = "max_steps_reached"
    device_label = "cpu-dry-run"
    return loss_value, steps_completed, device_label, failure_signals, stop_reason


def synthetic_batch(runtime_config: RuntimePhaseConfig, step: int) -> dict[str, Any]:
    base_token = (runtime_config.seed + step) % 17
    batch = [
        [(base_token + row + column) % 32 for column in range(runtime_config.settings.sequence_length)]
        for row in range(runtime_config.settings.batch_size)
    ]
    payload: dict[str, Any] = {"input_ids": batch}
    if runtime_config.plugin.has_recurrent_state:
        payload["state_tensor"] = [[0.0] * runtime_config.settings.sequence_length for _ in range(runtime_config.settings.batch_size)]
    if runtime_config.plugin.has_external_memory:
        payload["memory_tensor"] = [[0.0] * runtime_config.settings.sequence_length for _ in range(runtime_config.settings.batch_size)]
    if runtime_config.plugin.has_cache_path:
        payload["cache_tensor"] = [[0.0] * runtime_config.settings.sequence_length for _ in range(runtime_config.settings.batch_size)]
    return payload


def call_model(model: Any, batch: dict[str, Any], *, runtime_config: RuntimePhaseConfig, plugin_descriptor: dict[str, Any]) -> Any:
    supports = plugin_descriptor.get("supports", {}) if isinstance(plugin_descriptor, dict) else {}
    kwargs: dict[str, Any] = {}
    if runtime_config.plugin.has_recurrent_state and supports.get("recurrent_state", False):
        kwargs["state_tensor"] = batch["state_tensor"]
    if runtime_config.plugin.has_external_memory and supports.get("external_memory", False):
        kwargs["memory_tensor"] = batch["memory_tensor"]
    if runtime_config.plugin.has_cache_path and supports.get("cache_path", False):
        kwargs["cache_tensor"] = batch["cache_tensor"]
    return model.forward(batch["input_ids"], **kwargs) if hasattr(model, "forward") else model(batch["input_ids"], **kwargs)


def shape_of(value: Any) -> list[int]:
    shape: list[int] = []
    current = value
    while isinstance(current, list) and current:
        shape.append(len(current))
        current = current[0]
    if isinstance(current, list):
        shape.append(0)
    return shape


def _base_loss_for_phase(phase: str) -> float:
    return {"smoke": 6.0, "small": 4.4, "full": 3.9}[phase]


def _loss_delta_for_phase(phase: str) -> float:
    return {"smoke": 0.025, "small": 0.035, "full": 0.03}[phase]


def _phase_status(success_checks_satisfied: bool, failure_signals: list[str], stop_reason: str) -> str:
    if failure_signals:
        return "passed_with_warnings" if stop_reason == "max_wall_time_reached" else "failed"
    return "passed" if success_checks_satisfied else "passed_with_warnings"


def _next_action_recommendation(phase: str, status: str) -> str:
    if status == "passed":
        return "complete" if phase == "full" else "advance"
    return "review_warnings"
