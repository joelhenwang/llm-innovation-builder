from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from auto_llm_innovator.filesystem import ensure_dir
from auto_llm_innovator.runtime.checkpoints import load_checkpoint, save_checkpoint
from auto_llm_innovator.runtime.config import RuntimePhaseConfig
from auto_llm_innovator.runtime.eval_loop import run_evaluation_tasks
from auto_llm_innovator.runtime.train_loop import execute_training

from .model_checks import CheckOutcome


def loss_sanity_check(forward_output: dict[str, Any]) -> tuple[CheckOutcome, float | None]:
    logits = forward_output["logits"]
    batch_size = len(logits) if isinstance(logits, list) else 0
    sequence_length = len(logits[0]) if batch_size and isinstance(logits[0], list) else 0
    vocab_size = len(logits[0][0]) if sequence_length and isinstance(logits[0][0], list) else 0
    loss_value = round(1.0 + (batch_size * sequence_length / max(vocab_size, 1)), 6)
    if not math.isfinite(loss_value):
        return (
            CheckOutcome(
                name="loss_sanity",
                status="failed",
                message="Synthetic preflight loss was not finite.",
                failure_category="non_finite_loss",
            ),
            None,
        )
    return (
        CheckOutcome(
            name="loss_sanity",
            status="passed",
            message="Synthetic loss computation produced a finite value.",
            details={"loss_value": loss_value},
        ),
        loss_value,
    )


def train_step_sanity_check(
    *,
    runtime_config: RuntimePhaseConfig,
    build_model: Any,
    model_config_cls: Any,
    run_dir: Path,
    plugin_descriptor: dict[str, Any],
) -> tuple[CheckOutcome, float | None]:
    loss_value, steps_completed, _device_label, failure_signals, _stop_reason = execute_training(
        runtime_config=runtime_config,
        build_model=build_model,
        model_config_cls=model_config_cls,
        run_path=run_dir,
        checkpoint=None,
        plugin_descriptor=plugin_descriptor,
        max_steps=1,
        write_metrics=False,
    )
    if failure_signals:
        return (
            CheckOutcome(
                name="train_step_sanity",
                status="failed",
                message="Shared runtime failed during the one-batch preflight training step.",
                failure_category="train_step_failure",
                details={"failure_signals": failure_signals},
            ),
            None,
        )
    return (
        CheckOutcome(
            name="train_step_sanity",
            status="passed",
            message="Shared runtime completed a one-batch synthetic training step.",
            details={"steps_completed": steps_completed, "loss_value": loss_value},
        ),
        loss_value,
    )


def checkpoint_roundtrip_check(run_dir: Path, runtime_config: RuntimePhaseConfig, loss_value: float) -> tuple[CheckOutcome, Path | None]:
    checkpoint_filename = f"preflight-{runtime_config.checkpoints.filename}"
    payload = {
        "idea_id": runtime_config.idea_id,
        "phase": runtime_config.phase,
        "completed_steps": 1,
        "loss": loss_value,
        "checkpoint_kind": "preflight",
    }
    try:
        checkpoint_path = save_checkpoint(run_dir, checkpoint_filename, payload)
        loaded = load_checkpoint(run_dir, checkpoint_filename)
    except Exception as exc:
        return (
            CheckOutcome(
                name="checkpoint_roundtrip",
                status="failed",
                message=f"Preflight checkpoint save/load failed: {exc}",
                failure_category="checkpoint_failure",
            ),
            None,
        )
    if loaded != payload:
        return (
            CheckOutcome(
                name="checkpoint_roundtrip",
                status="failed",
                message="Preflight checkpoint roundtrip changed payload contents.",
                failure_category="checkpoint_failure",
            ),
            None,
        )
    return (
        CheckOutcome(
            name="checkpoint_roundtrip",
            status="passed",
            message="Preflight checkpoint save/load roundtrip succeeded.",
            details={"checkpoint_path": str(checkpoint_path)},
        ),
        checkpoint_path,
    )


def eval_hook_sanity_check(
    run_dir: Path,
    runtime_config: RuntimePhaseConfig,
    loss_value: float,
    evaluation_hooks: dict[str, Any] | None,
) -> tuple[CheckOutcome, Path | None]:
    preflight_eval_dir = ensure_dir(run_dir / "preflight")
    try:
        payload, evaluation_path = run_evaluation_tasks(
            runtime_config,
            run_dir=preflight_eval_dir,
            loss_value=loss_value,
            evaluation_hooks=evaluation_hooks,
        )
    except Exception as exc:
        return (
            CheckOutcome(
                name="eval_hook_sanity",
                status="failed",
                message=f"Generated evaluation hooks failed during preflight: {exc}",
                failure_category="evaluation_contract_failure",
                failing_modules=["package.evaluation.hooks"],
                failing_files=["package/evaluation/hooks.py"],
            ),
            None,
        )
    return (
        CheckOutcome(
            name="eval_hook_sanity",
            status="passed",
            message="Generated evaluation hooks executed successfully during preflight.",
            details={"task_count": len(payload["tasks"]), "evaluation_path": str(evaluation_path)},
        ),
        evaluation_path,
    )
