from __future__ import annotations

from typing import Any

from auto_llm_innovator.validation import PreflightResult

from .models import FailureClassification


_PREFLIGHT_CATEGORY_MAP = {
    "missing_generated_file": "package_import_failure",
    "package_import_failure": "package_import_failure",
    "plugin_contract_failure": "plugin_contract_failure",
    "checkpoint_failure": "checkpoint_failure",
    "evaluation_contract_failure": "evaluation_contract_failure",
    "model_instantiation_failure": "model_construction_failure",
    "forward_pass_failure": "runtime_output_shape_failure",
    "train_step_failure": "runtime_output_shape_failure",
}

_REPAIRABLE_CATEGORIES = {
    "package_import_failure",
    "plugin_contract_failure",
    "invalid_runtime_settings",
    "runtime_output_shape_failure",
    "checkpoint_failure",
    "evaluation_contract_failure",
    "model_construction_failure",
}


def classify_preflight_failure(result: PreflightResult) -> FailureClassification:
    raw_category = result.failure_categories[0] if result.failure_categories else "unknown_runtime_failure"
    category = _PREFLIGHT_CATEGORY_MAP.get(raw_category, "unknown_runtime_failure")
    return FailureClassification(
        source="preflight",
        category=category,
        repairable=category in _REPAIRABLE_CATEGORIES,
        summary=result.failure_signals[0] if result.failure_signals else "Preflight failed.",
        failure_signals=list(result.failure_signals),
        failing_modules=list(result.failing_modules),
        failing_files=list(result.failing_files),
        details={"raw_failure_categories": list(result.failure_categories)},
    )


def classify_runtime_failure(payload: dict[str, Any]) -> FailureClassification:
    stop_reason = str(payload.get("consumed_budget", {}).get("stop_reason") or payload.get("stop_reason") or "")
    status = str(payload.get("status", "failed"))
    recommendation = payload.get("next_action_recommendation")
    failure_signals = list(payload.get("failure_signals", []))

    if status == "passed_with_warnings" or stop_reason == "max_wall_time_reached":
        category = "unknown_runtime_failure"
        repairable = False
    elif stop_reason == "invalid_runtime_settings" or recommendation == "repair_runtime_config":
        category = "invalid_runtime_settings"
        repairable = True
    elif stop_reason == "runtime_validation_failed":
        category = "runtime_output_shape_failure"
        repairable = True
    elif stop_reason == "model_construction_failed":
        category = "model_construction_failure"
        repairable = True
    elif recommendation == "repair_plugin_contract":
        category = "plugin_contract_failure"
        repairable = True
    elif any("checkpoint" in signal.lower() for signal in failure_signals):
        category = "checkpoint_failure"
        repairable = True
    elif any("evaluation" in signal.lower() or "hook" in signal.lower() for signal in failure_signals):
        category = "evaluation_contract_failure"
        repairable = True
    elif any("nan" in signal.lower() or "inf" in signal.lower() for signal in failure_signals):
        category = "nan_loss_like_failure"
        repairable = False
    elif any("out of memory" in signal.lower() or "oom" in signal.lower() for signal in failure_signals):
        category = "oom_like_failure"
        repairable = False
    else:
        category = "unknown_runtime_failure"
        repairable = False

    return FailureClassification(
        source="runtime",
        category=category,
        repairable=repairable and category in _REPAIRABLE_CATEGORIES,
        summary=failure_signals[0] if failure_signals else "Runtime failed.",
        stop_reason=stop_reason or None,
        next_action_recommendation=str(recommendation) if recommendation is not None else None,
        failure_signals=failure_signals,
        details={"status": status},
    )
