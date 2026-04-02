from __future__ import annotations

import importlib
import sys
from shutil import rmtree
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

from auto_llm_innovator.filesystem import read_json
from auto_llm_innovator.runtime.config import RuntimePhaseConfig
from auto_llm_innovator.runtime.train_loop import (
    build_model_instance,
    call_model,
    shape_of,
    synthetic_batch,
    validate_plugin_contract,
)


@dataclass(slots=True)
class CheckOutcome:
    name: str
    status: str
    message: str
    failure_category: str | None = None
    failing_modules: list[str] = field(default_factory=list)
    failing_files: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "details": self.details,
        }
        if self.failure_category is not None:
            payload["failure_category"] = self.failure_category
        if self.failing_modules:
            payload["failing_modules"] = self.failing_modules
        if self.failing_files:
            payload["failing_files"] = self.failing_files
        return payload


@contextmanager
def idea_package_import_context(idea_dir: Path) -> Iterator[None]:
    import_root = str(idea_dir)
    for cache_dir in idea_dir.rglob("__pycache__"):
        rmtree(cache_dir, ignore_errors=True)
    removed_modules = {
        name: module for name, module in sys.modules.items() if name == "package" or name.startswith("package.")
    }
    for name in list(removed_modules):
        sys.modules.pop(name, None)
    sys.path.insert(0, import_root)
    importlib.invalidate_caches()
    try:
        yield
    finally:
        if sys.path and sys.path[0] == import_root:
            sys.path.pop(0)
        for name in list(sys.modules):
            if name == "package" or name.startswith("package."):
                sys.modules.pop(name, None)
        sys.modules.update(removed_modules)


def load_generated_package_modules(idea_dir: Path) -> tuple[ModuleType, ModuleType, ModuleType]:
    with idea_package_import_context(idea_dir):
        package_module = importlib.import_module("package")
        plugin_module = importlib.import_module("package.plugin")
        hooks_module = importlib.import_module("package.evaluation.hooks")
    return package_module, plugin_module, hooks_module


def package_import_check(idea_dir: Path) -> tuple[CheckOutcome, ModuleType | None, ModuleType | None, ModuleType | None]:
    try:
        manifest = read_json(idea_dir / "generation_manifest.json")
    except Exception as exc:
        return (
            CheckOutcome(
                name="package_import",
                status="failed",
                message=f"Unable to read generation manifest: {exc}",
                failure_category="missing_generated_file",
                failing_files=["generation_manifest.json"],
            ),
            None,
            None,
            None,
        )
    missing_files = sorted(
        str((idea_dir / relative_path).relative_to(idea_dir))
        for relative_path in manifest.get("generated_files", [])
        if not (idea_dir / relative_path).exists()
    )
    if missing_files:
        return (
            CheckOutcome(
                name="package_import",
                status="failed",
                message="Generated package is missing files declared in the generation manifest.",
                failure_category="missing_generated_file",
                failing_files=missing_files,
                details={"manifest_path": "generation_manifest.json"},
            ),
            None,
            None,
            None,
        )

    try:
        package_module, plugin_module, hooks_module = load_generated_package_modules(idea_dir)
    except Exception as exc:
        return (
            CheckOutcome(
                name="package_import",
                status="failed",
                message=f"Unable to import generated package modules: {exc}",
                failure_category="package_import_failure",
                failing_modules=["package", "package.plugin", "package.evaluation.hooks"],
                failing_files=["package/__init__.py", "package/plugin.py", "package/evaluation/hooks.py"],
            ),
            None,
            None,
            None,
        )

    return (
        CheckOutcome(
            name="package_import",
            status="passed",
            message="Generated package imports resolved successfully.",
            details={"manifest_files": len(manifest.get("generated_files", []))},
        ),
        package_module,
        plugin_module,
        hooks_module,
    )


def plugin_contract_check(plugin_module: ModuleType | Any, runtime_config: RuntimePhaseConfig) -> CheckOutcome:
    failure_signals = validate_plugin_contract(plugin_module, runtime_config)
    if failure_signals:
        return CheckOutcome(
            name="plugin_contract",
            status="failed",
            message="Generated plugin does not satisfy the shared runtime contract.",
            failure_category="plugin_contract_failure",
            failing_modules=["package.plugin"],
            failing_files=["package/plugin.py"],
            details={"failure_signals": failure_signals},
        )

    descriptor = getattr(plugin_module, "describe_plugin")()
    return CheckOutcome(
        name="plugin_contract",
        status="passed",
        message="Generated plugin satisfied the shared runtime contract.",
        details={"declared_modules": list(descriptor.get("module_names", []))},
    )


def model_instantiation_check(plugin_module: ModuleType | Any) -> tuple[CheckOutcome, Any | None, Any | None]:
    try:
        model_config_cls = getattr(plugin_module, "ModelConfig", None)
        build_model = getattr(plugin_module, "build_model")
        model, model_config = build_model_instance(build_model, model_config_cls)
    except Exception as exc:
        return (
            CheckOutcome(
                name="model_instantiation",
                status="failed",
                message=f"Unable to instantiate ModelConfig/build_model: {exc}",
                failure_category="model_instantiation_failure",
                failing_modules=["package.plugin"],
                failing_files=["package/plugin.py", "package/config.py"],
            ),
            None,
            None,
        )

    return (
        CheckOutcome(
            name="model_instantiation",
            status="passed",
            message="ModelConfig and build_model(...) instantiated successfully.",
            details={"model_type": type(model).__name__, "config_type": type(model_config).__name__},
        ),
        model,
        model_config,
    )


def forward_pass_check(
    model: Any,
    runtime_config: RuntimePhaseConfig,
    plugin_descriptor: dict[str, Any],
) -> tuple[CheckOutcome, dict[str, Any] | None]:
    batch = synthetic_batch(runtime_config, 0)
    try:
        forward_output = call_model(model, batch, runtime_config=runtime_config, plugin_descriptor=plugin_descriptor)
    except Exception as exc:
        return (
            CheckOutcome(
                name="forward_pass",
                status="failed",
                message=f"Generated model forward pass failed on a synthetic batch: {exc}",
                failure_category="forward_pass_failure",
                failing_modules=["package.modeling.model"],
                failing_files=["package/modeling/model.py"],
            ),
            None,
        )

    if not isinstance(forward_output, dict) or "logits" not in forward_output:
        return (
            CheckOutcome(
                name="forward_pass",
                status="failed",
                message="Generated model forward output must be a dict containing 'logits'.",
                failure_category="forward_pass_failure",
                failing_modules=["package.modeling.model"],
                failing_files=["package/modeling/model.py"],
            ),
            None,
        )

    missing_outputs = []
    if runtime_config.plugin.has_recurrent_state and "state_tensor" not in forward_output:
        missing_outputs.append("state_tensor")
    if runtime_config.plugin.has_external_memory and "memory_tensor" not in forward_output:
        missing_outputs.append("memory_tensor")
    if runtime_config.plugin.has_cache_path and "cache_tensor" not in forward_output:
        missing_outputs.append("cache_tensor")
    if missing_outputs:
        return (
            CheckOutcome(
                name="forward_pass",
                status="failed",
                message=f"Generated model omitted required runtime outputs: {', '.join(missing_outputs)}.",
                failure_category="forward_pass_failure",
                failing_modules=["package.modeling.model"],
                failing_files=["package/modeling/model.py"],
            ),
            None,
        )

    logits_shape = shape_of(forward_output["logits"])
    if len(logits_shape) < 2:
        return (
            CheckOutcome(
                name="forward_pass",
                status="failed",
                message="Generated model logits must expose at least batch and vocabulary dimensions.",
                failure_category="forward_pass_failure",
                failing_modules=["package.modeling.model"],
                failing_files=["package/modeling/model.py"],
                details={"logits_shape": logits_shape},
            ),
            None,
        )

    return (
        CheckOutcome(
            name="forward_pass",
            status="passed",
            message="Generated model completed a synthetic forward pass with runtime-compatible outputs.",
            details={"logits_shape": logits_shape},
        ),
        forward_output,
    )
