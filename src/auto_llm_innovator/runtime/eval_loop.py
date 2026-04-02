from __future__ import annotations

from pathlib import Path
from typing import Any

from auto_llm_innovator.filesystem import write_json
from auto_llm_innovator.runtime.config import RuntimePhaseConfig


def run_evaluation_tasks(
    runtime_config: RuntimePhaseConfig,
    *,
    run_dir: Path,
    loss_value: float,
    evaluation_hooks: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Path]:
    tasks = []
    for task in runtime_config.evaluation.tasks:
        metrics = {}
        for metric_name in task.metrics:
            if metric_name == "perplexity":
                metrics[metric_name] = round(pow(2.718281828, min(loss_value, 8.0)), 4)
            elif metric_name == "relative_quality":
                metrics[metric_name] = round(max(0.0, 1.0 - (loss_value / 10.0)), 4)
            elif metric_name == "validation_score":
                metrics[metric_name] = round(max(0.0, 100.0 - (loss_value * 10.0)), 4)
            else:
                metrics[metric_name] = round(loss_value, 4)

        if evaluation_hooks and callable(evaluation_hooks.get(task.name)):
            hook_result = evaluation_hooks[task.name](loss_value=loss_value, runtime_config=runtime_config)
            if isinstance(hook_result, dict):
                metrics.update(hook_result)

        tasks.append(
            {
                "name": task.name,
                "description": task.description,
                "phase": task.phase,
                "comparison_targets": list(task.comparison_targets),
                "metrics": metrics,
            }
        )

    payload = {
        "idea_id": runtime_config.idea_id,
        "phase": runtime_config.phase,
        "objective": runtime_config.objective,
        "tasks": tasks,
        "ablation_names": list(runtime_config.ablation_names),
    }
    report_path = run_dir / runtime_config.logging.evaluation_filename
    write_json(report_path, payload)
    return payload, report_path
