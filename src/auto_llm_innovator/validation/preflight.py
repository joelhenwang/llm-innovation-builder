from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from auto_llm_innovator.filesystem import write_json
from auto_llm_innovator.runtime.config import RuntimePhaseConfig

from .model_checks import (
    CheckOutcome,
    forward_pass_check,
    model_instantiation_check,
    package_import_check,
    plugin_contract_check,
)
from .train_checks import checkpoint_roundtrip_check, eval_hook_sanity_check, loss_sanity_check, train_step_sanity_check


RETRYABLE_FAILURE_CATEGORIES = {
    "missing_generated_file",
    "package_import_failure",
    "plugin_contract_failure",
}


@dataclass(slots=True)
class PreflightResult:
    status: str
    attempt_id: str
    phase: str
    checks: list[CheckOutcome] = field(default_factory=list)
    failure_categories: list[str] = field(default_factory=list)
    failure_signals: list[str] = field(default_factory=list)
    failing_modules: list[str] = field(default_factory=list)
    failing_files: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    plugin_module: ModuleType | None = None
    report_path: Path | None = None

    @property
    def retryable(self) -> bool:
        return any(category in RETRYABLE_FAILURE_CATEGORIES for category in self.failure_categories)

    def to_report_payload(self, *, retry_attempted: bool, retry_outcome: str) -> dict[str, Any]:
        return {
            "status": self.status,
            "attempt_id": self.attempt_id,
            "phase": self.phase,
            "checks": [check.to_dict() for check in self.checks],
            "failure_categories": self.failure_categories,
            "failure_signals": self.failure_signals,
            "retry_attempted": retry_attempted,
            "retry_outcome": retry_outcome,
            "failing_modules": self.failing_modules,
            "failing_files": self.failing_files,
            "artifacts": self.artifacts,
        }


def run_preflight(
    *,
    idea_dir: Path,
    run_dir: Path,
    runtime_config: RuntimePhaseConfig,
    attempt_id: str,
) -> PreflightResult:
    checks: list[CheckOutcome] = []
    artifacts: list[str] = []

    package_check, _package_module, plugin_module, _hooks_module = package_import_check(idea_dir)
    checks.append(package_check)
    if package_check.status != "passed" or plugin_module is None:
        return _finalize_result(
            status="failed",
            attempt_id=attempt_id,
            phase=runtime_config.phase,
            checks=checks,
            artifacts=artifacts,
            plugin_module=None,
        )

    contract_check = plugin_contract_check(plugin_module, runtime_config)
    checks.append(contract_check)
    if contract_check.status != "passed":
        return _finalize_result(
            status="failed",
            attempt_id=attempt_id,
            phase=runtime_config.phase,
            checks=checks,
            artifacts=artifacts,
            plugin_module=plugin_module,
        )

    instantiation_check, model, _model_config = model_instantiation_check(plugin_module)
    checks.append(instantiation_check)
    if instantiation_check.status != "passed" or model is None:
        return _finalize_result(
            status="failed",
            attempt_id=attempt_id,
            phase=runtime_config.phase,
            checks=checks,
            artifacts=artifacts,
            plugin_module=plugin_module,
        )

    plugin_descriptor = getattr(plugin_module, "describe_plugin")()
    forward_check, forward_output = forward_pass_check(model, runtime_config, plugin_descriptor)
    checks.append(forward_check)
    if forward_check.status != "passed" or forward_output is None:
        return _finalize_result(
            status="failed",
            attempt_id=attempt_id,
            phase=runtime_config.phase,
            checks=checks,
            artifacts=artifacts,
            plugin_module=plugin_module,
        )

    loss_check, synthetic_loss = loss_sanity_check(forward_output)
    checks.append(loss_check)
    if loss_check.status != "passed" or synthetic_loss is None:
        return _finalize_result(
            status="failed",
            attempt_id=attempt_id,
            phase=runtime_config.phase,
            checks=checks,
            artifacts=artifacts,
            plugin_module=plugin_module,
        )

    train_check, preflight_loss = train_step_sanity_check(
        runtime_config=runtime_config,
        build_model=getattr(plugin_module, "build_model"),
        model_config_cls=getattr(plugin_module, "ModelConfig", None),
        run_dir=run_dir,
        plugin_descriptor=plugin_descriptor,
    )
    checks.append(train_check)
    if train_check.status != "passed" or preflight_loss is None:
        return _finalize_result(
            status="failed",
            attempt_id=attempt_id,
            phase=runtime_config.phase,
            checks=checks,
            artifacts=artifacts,
            plugin_module=plugin_module,
        )

    checkpoint_check, checkpoint_path = checkpoint_roundtrip_check(run_dir, runtime_config, preflight_loss)
    checks.append(checkpoint_check)
    if checkpoint_path is not None:
        artifacts.append(str(checkpoint_path))
    if checkpoint_check.status != "passed":
        return _finalize_result(
            status="failed",
            attempt_id=attempt_id,
            phase=runtime_config.phase,
            checks=checks,
            artifacts=artifacts,
            plugin_module=plugin_module,
        )

    evaluation_hooks = getattr(plugin_module, "register_evaluation_hooks", lambda: {})()
    eval_check, evaluation_path = eval_hook_sanity_check(run_dir, runtime_config, preflight_loss, evaluation_hooks)
    checks.append(eval_check)
    if evaluation_path is not None:
        artifacts.append(str(evaluation_path))
    return _finalize_result(
        status="passed" if eval_check.status == "passed" else "failed",
        attempt_id=attempt_id,
        phase=runtime_config.phase,
        checks=checks,
        artifacts=artifacts,
        plugin_module=plugin_module,
    )


def write_preflight_report(
    run_dir: Path,
    result: PreflightResult,
    *,
    retry_attempted: bool,
    retry_outcome: str,
) -> Path:
    report_path = run_dir / "preflight-report.json"
    if str(report_path) not in result.artifacts:
        result.artifacts.append(str(report_path))
    write_json(report_path, result.to_report_payload(retry_attempted=retry_attempted, retry_outcome=retry_outcome))
    result.report_path = report_path
    return report_path


def _finalize_result(
    *,
    status: str,
    attempt_id: str,
    phase: str,
    checks: list[CheckOutcome],
    artifacts: list[str],
    plugin_module: ModuleType | None,
) -> PreflightResult:
    failure_categories = []
    failure_signals = []
    failing_modules: list[str] = []
    failing_files: list[str] = []
    for check in checks:
        if check.status == "failed":
            if check.failure_category is not None:
                failure_categories.append(check.failure_category)
            failure_signals.append(check.message)
            failing_modules.extend(check.failing_modules)
            failing_files.extend(check.failing_files)
    return PreflightResult(
        status=status,
        attempt_id=attempt_id,
        phase=phase,
        checks=checks,
        failure_categories=sorted(set(failure_categories)),
        failure_signals=failure_signals,
        failing_modules=sorted(set(failing_modules)),
        failing_files=sorted(set(failing_files)),
        artifacts=artifacts,
        plugin_module=plugin_module,
    )
