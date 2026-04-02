from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path

from auto_llm_innovator.design_ir import DesignIR
from auto_llm_innovator.filesystem import ensure_dir, read_json, read_text, write_json, write_text
from auto_llm_innovator.generation.layout import build_idea_package_layout
from auto_llm_innovator.generation.postprocess import normalize_generated_source
from auto_llm_innovator.generation.renderers.package import render_idea_package_sources
from auto_llm_innovator.idea_spec import IdeaSpec
from auto_llm_innovator.runtime import RuntimePhaseConfig, default_runtime_settings_for_phase

from .models import FailureClassification, RepairAttempt, RepairLoopResult


def persist_failure_classification(run_dir: Path, classification: FailureClassification) -> Path:
    repair_dir = ensure_dir(run_dir / "repair")
    path = repair_dir / "failure-classification.json"
    write_json(path, classification.to_dict())
    return path


def persist_repair_history(run_dir: Path, history: list[RepairAttempt]) -> Path:
    repair_dir = ensure_dir(run_dir / "repair")
    path = repair_dir / "repair-history.json"
    write_json(path, [attempt.to_dict() for attempt in history])
    return path


def new_repair_loop_result(max_repairs: int) -> RepairLoopResult:
    return RepairLoopResult(repairs_remaining=max_repairs)


@dataclass(slots=True)
class _Context:
    idea_dir: Path
    run_dir: Path
    phase: str
    classification: FailureClassification
    attempt_index: int
    runtime_config: RuntimePhaseConfig | None

    @property
    def repair_dir(self) -> Path:
        return ensure_dir(self.run_dir / "repair")


class RepairStrategy:
    name = "base"
    categories: tuple[str, ...] = ()

    def can_handle(self, classification: FailureClassification) -> bool:
        return classification.category in self.categories

    def apply(self, context: _Context) -> RepairAttempt:
        raise NotImplementedError


class RenderedFilesStrategy(RepairStrategy):
    file_keys: tuple[str, ...] = ()
    rationale = "Restore generated files to the deterministic renderer output."

    def apply(self, context: _Context) -> RepairAttempt:
        rendered = _render_expected_sources(context.idea_dir)
        targets = [context.idea_dir / key for key in self.file_keys]
        if not any(str(path) in rendered or path.exists() for path in targets):
            targets = [context.idea_dir / relative for relative in context.classification.failing_files]
        return _apply_rendered_files(context, rendered, targets, self.name, self.rationale)


class PackageImportRepairStrategy(RenderedFilesStrategy):
    name = "repair_package_import"
    categories = ("package_import_failure",)
    file_keys = (
        "generation_manifest.json",
        "package/__init__.py",
        "package/plugin.py",
        "package/evaluation/__init__.py",
        "package/evaluation/hooks.py",
    )
    rationale = "Restore generated import surfaces and manifest to the deterministic package state."


class PluginContractRepairStrategy(RenderedFilesStrategy):
    name = "repair_plugin_contract"
    categories = ("plugin_contract_failure",)
    file_keys = ("package/plugin.py", "package/__init__.py")
    rationale = "Restore plugin exports and descriptor flags required by the runtime contract."


class RuntimeOutputRepairStrategy(RenderedFilesStrategy):
    name = "repair_runtime_outputs"
    categories = ("runtime_output_shape_failure",)
    file_keys = ("package/modeling/model.py", "package/modeling/state.py")
    rationale = "Restore the generated forward output contract expected by the shared runtime."


class EvaluationRepairStrategy(RenderedFilesStrategy):
    name = "repair_evaluation_hooks"
    categories = ("evaluation_contract_failure",)
    file_keys = ("package/evaluation/hooks.py",)
    rationale = "Restore dict-returning evaluation hooks for the generated package."


class ModelConstructionRepairStrategy(RenderedFilesStrategy):
    name = "repair_model_construction"
    categories = ("model_construction_failure",)
    file_keys = ("package/config.py", "package/plugin.py", "package/modeling/model.py")
    rationale = "Restore generated model construction surfaces to the deterministic contract."


class CheckpointRepairStrategy(RenderedFilesStrategy):
    name = "repair_checkpoint_wiring"
    categories = ("checkpoint_failure",)
    file_keys = ("package/plugin.py", "package/evaluation/hooks.py")
    rationale = "Restore generated runtime-adjacent wiring used by checkpoint and evaluation flows."


class RuntimeSettingsRepairStrategy(RepairStrategy):
    name = "repair_runtime_settings"
    categories = ("invalid_runtime_settings",)

    def apply(self, context: _Context) -> RepairAttempt:
        config_path = context.idea_dir / "config" / f"{context.phase}.json"
        before_paths = _snapshot_before(context, [config_path])
        payload = read_json(config_path)
        defaults = default_runtime_settings_for_phase(context.phase).to_dict()
        runtime = payload.get("runtime", {})
        repaired = dict(defaults)
        repaired["resume_enabled"] = bool(runtime.get("resume_enabled", defaults["resume_enabled"]))
        repaired["evaluation_scope"] = str(runtime.get("evaluation_scope", defaults["evaluation_scope"]))
        repaired["dataset_slice"] = str(runtime.get("dataset_slice", defaults["dataset_slice"]))
        for key in ("max_steps", "max_wall_time_seconds", "sequence_length", "batch_size", "checkpoint_every_steps"):
            value = runtime.get(key, defaults[key])
            try:
                repaired[key] = int(value)
            except Exception:
                repaired[key] = int(defaults[key])
        if repaired["max_steps"] <= 0:
            repaired["max_steps"] = defaults["max_steps"]
        if repaired["max_wall_time_seconds"] < 0:
            repaired["max_wall_time_seconds"] = defaults["max_wall_time_seconds"]
        if repaired["sequence_length"] <= 0:
            repaired["sequence_length"] = defaults["sequence_length"]
        if repaired["batch_size"] <= 0:
            repaired["batch_size"] = defaults["batch_size"]
        if repaired["checkpoint_every_steps"] < 0:
            repaired["checkpoint_every_steps"] = defaults["checkpoint_every_steps"]
        if repaired["evaluation_scope"] not in {"minimal", "standard", "full"}:
            repaired["evaluation_scope"] = defaults["evaluation_scope"]
        payload["runtime"] = repaired
        write_json(config_path, payload)
        return _finalize_attempt(
            context=context,
            strategy=self.name,
            target_files=[config_path],
            before_paths=before_paths,
            rationale="Normalize invalid runtime settings back to the validated defaults for this phase.",
        )


_STRATEGIES: tuple[RepairStrategy, ...] = (
    RuntimeSettingsRepairStrategy(),
    PackageImportRepairStrategy(),
    PluginContractRepairStrategy(),
    RuntimeOutputRepairStrategy(),
    EvaluationRepairStrategy(),
    ModelConstructionRepairStrategy(),
    CheckpointRepairStrategy(),
)


def apply_repair(
    *,
    idea_dir: Path,
    run_dir: Path,
    phase: str,
    classification: FailureClassification,
    attempt_index: int,
    runtime_config: RuntimePhaseConfig | None,
) -> RepairAttempt | None:
    context = _Context(
        idea_dir=idea_dir,
        run_dir=run_dir,
        phase=phase,
        classification=classification,
        attempt_index=attempt_index,
        runtime_config=runtime_config,
    )
    for strategy in _STRATEGIES:
        if strategy.can_handle(classification):
            return strategy.apply(context)
    return None


def _render_expected_sources(idea_dir: Path) -> dict[str, str]:
    spec = IdeaSpec.from_dict(read_json(idea_dir / "idea_spec.json"))
    design_ir = DesignIR.from_dict(read_json(idea_dir / "design_ir.json"))
    layout = build_idea_package_layout(idea_dir)
    return {
        path: normalize_generated_source(content)
        for path, content in render_idea_package_sources(layout, spec, design_ir).items()
    }


def _apply_rendered_files(
    context: _Context,
    rendered: dict[str, str],
    target_files: list[Path],
    strategy: str,
    rationale: str,
) -> RepairAttempt:
    before_paths = _snapshot_before(context, target_files)
    for path in target_files:
        rendered_content = rendered.get(str(path))
        if rendered_content is not None:
            write_text(path, rendered_content)
    return _finalize_attempt(
        context=context,
        strategy=strategy,
        target_files=target_files,
        before_paths=before_paths,
        rationale=rationale,
    )


def _snapshot_before(context: _Context, target_files: list[Path]) -> list[tuple[Path, str]]:
    before_dir = ensure_dir(context.repair_dir / f"attempt-{context.attempt_index:04d}-before")
    snapshots: list[tuple[Path, str]] = []
    for path in target_files:
        if path.exists():
            content = read_text(path)
            snapshots.append((path, content))
            write_text(before_dir / path.relative_to(context.idea_dir), content)
    return snapshots


def _finalize_attempt(
    *,
    context: _Context,
    strategy: str,
    target_files: list[Path],
    before_paths: list[tuple[Path, str]],
    rationale: str,
) -> RepairAttempt:
    after_dir = ensure_dir(context.repair_dir / f"attempt-{context.attempt_index:04d}-after")
    for path in target_files:
        if path.exists():
            write_text(after_dir / path.relative_to(context.idea_dir), read_text(path))
    diff_path = context.repair_dir / f"attempt-{context.attempt_index:04d}-diff.patch"
    rationale_path = context.repair_dir / f"attempt-{context.attempt_index:04d}-rationale.json"
    write_text(diff_path, _build_diff(context.idea_dir, before_paths, target_files))
    write_json(
        rationale_path,
        {
            "attempt_index": context.attempt_index,
            "source": context.classification.source,
            "category": context.classification.category,
            "strategy": strategy,
            "summary": context.classification.summary,
            "rationale": rationale,
            "target_files": [str(path.relative_to(context.idea_dir)) for path in target_files],
        },
    )
    return RepairAttempt(
        attempt_index=context.attempt_index,
        source=context.classification.source,
        category=context.classification.category,
        strategy=strategy,
        target_files=[str(path.relative_to(context.idea_dir)) for path in target_files],
        before_snapshot_dir=str(context.repair_dir / f"attempt-{context.attempt_index:04d}-before"),
        after_snapshot_dir=str(after_dir),
        diff_path=str(diff_path),
        rationale_path=str(rationale_path),
        outcome="applied",
        rationale=rationale,
    )


def _build_diff(idea_dir: Path, before_paths: list[tuple[Path, str]], target_files: list[Path]) -> str:
    before_map = {path: content for path, content in before_paths}
    chunks: list[str] = []
    for path in target_files:
        before = before_map.get(path, "")
        after = read_text(path) if path.exists() else ""
        chunks.extend(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=str(path.relative_to(idea_dir)),
                tofile=str(path.relative_to(idea_dir)),
            )
        )
    return "".join(chunks)
