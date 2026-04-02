from __future__ import annotations

from pathlib import Path

from auto_llm_innovator.design_ir import DesignIR
from auto_llm_innovator.filesystem import ensure_dir, read_json, write_json
from auto_llm_innovator.generation import generate_idea_package
from auto_llm_innovator.idea_spec import IdeaSpec
from auto_llm_innovator.modeling.interfaces import PhaseResult
from auto_llm_innovator.repair import (
    apply_repair,
    classify_preflight_failure,
    classify_runtime_failure,
    new_repair_loop_result,
    persist_failure_classification,
    persist_repair_history,
)
from auto_llm_innovator.runtime import compile_runtime_phase_config, run_phase_with_plugin
from auto_llm_innovator.validation import run_preflight, write_preflight_report


def _regenerate_package(idea_dir: Path) -> None:
    spec = IdeaSpec.from_dict(read_json(idea_dir / "idea_spec.json"))
    design_ir = DesignIR.from_dict(read_json(idea_dir / "design_ir.json"))
    generate_idea_package(idea_dir, spec, design_ir)


def execute_phase(idea_dir: Path, attempt_id: str, phase: str, run_dir: Path) -> PhaseResult:
    run_dir = ensure_dir(run_dir)
    config_path = run_dir / "resolved-config.json"
    if not config_path.exists():
        config_path = idea_dir / "config" / f"{phase}.json"
    design_ir = DesignIR.from_dict(read_json(idea_dir / "design_ir.json"))
    phase_config = read_json(config_path)
    max_repairs = int(phase_config.get("max_retries_visible", 1))
    repair_state = new_repair_loop_result(max_repairs)

    while True:
        try:
            runtime_config = compile_runtime_phase_config(design_ir, read_json(config_path), attempt_id=attempt_id, phase=phase)
        except Exception as exc:
            classification = classify_runtime_failure(
                {
                    "status": "failed",
                    "failure_signals": [f"Runtime config compilation failed: {exc}"],
                    "next_action_recommendation": "repair_runtime_config",
                    "consumed_budget": {"stop_reason": "invalid_runtime_settings"},
                }
            )
            failed_result = _handle_failure(
                idea_dir=idea_dir,
                run_dir=run_dir,
                phase=phase,
                attempt_id=attempt_id,
                max_repairs=max_repairs,
                repair_state=repair_state,
                classification=classification,
                failure_signals=list(classification.failure_signals),
                reviewer_notes=["Runtime config compilation failed before preflight."],
                runtime_config=None,
            )
            if failed_result is None:
                continue
            return failed_result

        preflight = run_preflight(idea_dir=idea_dir, run_dir=run_dir, runtime_config=runtime_config, attempt_id=attempt_id)
        retry_outcome = "recovered" if repair_state.repair_attempted and preflight.status == "passed" else (
            "failed" if repair_state.repair_attempted and preflight.status != "passed" else "not_requested"
        )
        report_path = write_preflight_report(
            run_dir,
            preflight,
            retry_attempted=repair_state.repair_attempted,
            retry_outcome=retry_outcome,
        )
        repair_state.artifact_paths.append(str(report_path))
        if preflight.status != "passed" or preflight.plugin_module is None:
            classification = classify_preflight_failure(preflight)
            failed_result = _handle_failure(
                idea_dir=idea_dir,
                run_dir=run_dir,
                phase=phase,
                attempt_id=attempt_id,
                max_repairs=max_repairs,
                repair_state=repair_state,
                classification=classification,
                failure_signals=list(preflight.failure_signals),
                reviewer_notes=[
                    "Preflight validation ran before phase execution.",
                    "Shared runtime training was skipped because preflight failed.",
                ],
                runtime_config=runtime_config,
                inherited_artifacts=list(preflight.artifacts),
            )
            if failed_result is None:
                continue
            return failed_result

        raw = run_phase_with_plugin(
            phase=phase,
            idea_dir=str(idea_dir),
            run_dir=str(run_dir),
            config_path=str(config_path),
            attempt_id=attempt_id,
            plugin_module=preflight.plugin_module,
        )
        if raw["status"] == "failed":
            classification = classify_runtime_failure(raw)
            failed_result = _handle_failure(
                idea_dir=idea_dir,
                run_dir=run_dir,
                phase=phase,
                attempt_id=attempt_id,
                max_repairs=max_repairs,
                repair_state=repair_state,
                classification=classification,
                failure_signals=list(raw["failure_signals"]),
                reviewer_notes=["Preflight validation passed before phase execution.", *raw["reviewer_notes"]],
                runtime_config=runtime_config,
                inherited_artifacts=[*preflight.artifacts, *raw["artifacts_produced"]],
                next_action_recommendation=raw["next_action_recommendation"],
                consumed_budget=raw["consumed_budget"],
            )
            if failed_result is None:
                continue
            return failed_result

        artifacts = list(dict.fromkeys([*preflight.artifacts, *raw["artifacts_produced"], *repair_state.artifact_paths]))
        reviewer_notes = ["Preflight validation passed before phase execution.", *raw["reviewer_notes"]]
        if repair_state.repair_attempted:
            reviewer_notes.append(f"Repair loop recovered after {repair_state.repair_count} attempt(s).")
            if (
                repair_state.failure_classification is not None
                and repair_state.failure_classification.source == "preflight"
                and repair_state.failure_classification.category == "package_import_failure"
            ):
                reviewer_notes.append("Generated package was regenerated once and passed preflight on retry.")
        return PhaseResult(
            idea_id=idea_dir.name,
            attempt_id=attempt_id,
            phase=phase,
            status=raw["status"],
            key_metrics=raw["key_metrics"],
            failure_signals=raw["failure_signals"],
            artifacts_produced=artifacts,
            reviewer_notes=reviewer_notes,
            next_action_recommendation=raw["next_action_recommendation"],
            consumed_budget=raw["consumed_budget"],
            repair_attempted=repair_state.repair_attempted,
            repair_count=repair_state.repair_count,
            repair_outcome="recovered" if repair_state.repair_attempted else "not_attempted",
            failure_classification=repair_state.failure_classification.to_dict() if repair_state.failure_classification else None,
        )


def _handle_failure(
    *,
    idea_dir: Path,
    run_dir: Path,
    phase: str,
    attempt_id: str,
    max_repairs: int,
    repair_state,
    classification,
    failure_signals: list[str],
    reviewer_notes: list[str],
    runtime_config,
    inherited_artifacts: list[str] | None = None,
    next_action_recommendation: str | None = None,
    consumed_budget: dict | None = None,
) -> PhaseResult | None:
    repair_state.failure_classification = classification
    repair_state.artifact_paths.append(str(persist_failure_classification(run_dir, classification)))
    if repair_state.repair_count < max_repairs and classification.repairable:
        try:
            repair_attempt = apply_repair(
                idea_dir=idea_dir,
                run_dir=run_dir,
                phase=phase,
                classification=classification,
                attempt_index=repair_state.repair_count + 1,
                runtime_config=runtime_config,
            )
        except Exception as exc:
            repair_state.repair_attempted = True
            repair_state.repair_count += 1
            repair_state.repairs_remaining = max(max_repairs - repair_state.repair_count, 0)
            failure_signals.append(f"Deterministic package regeneration failed: {exc}")
            repair_attempt = None
        if repair_attempt is not None:
            repair_state.repair_attempted = True
            repair_state.history.append(repair_attempt)
            repair_state.repair_count = len(repair_state.history)
            repair_state.repairs_remaining = max(max_repairs - repair_state.repair_count, 0)
            repair_state.artifact_paths.extend(
                [
                    repair_attempt.before_snapshot_dir,
                    repair_attempt.after_snapshot_dir,
                    repair_attempt.diff_path,
                    repair_attempt.rationale_path,
                    str(persist_repair_history(run_dir, repair_state.history)),
                ]
            )
            return None

    budget = consumed_budget or {
        "requested_parameters": runtime_config.target_parameters if runtime_config is not None else 0,
        "steps": 0,
        "device": "cpu-dry-run",
        "resumed": False,
        "stop_reason": classification.stop_reason or "repair_not_attempted",
    }
    recommendation = next_action_recommendation or ("manual_repair_required" if not classification.repairable else "repair_preflight")
    if classification.source == "preflight":
        report_path = run_dir / "preflight-report.json"
        if report_path.exists():
            report = read_json(report_path)
            report["retry_attempted"] = repair_state.repair_attempted
            report["retry_outcome"] = "failed" if repair_state.repair_attempted else report.get("retry_outcome", "not_requested")
            write_json(report_path, report)
    return PhaseResult(
        idea_id=idea_dir.name,
        attempt_id=attempt_id,
        phase=phase,
        status="failed",
        key_metrics={},
        failure_signals=failure_signals,
        artifacts_produced=list(dict.fromkeys([*(inherited_artifacts or []), *repair_state.artifact_paths])),
        reviewer_notes=[
            *reviewer_notes,
            *([f"Repair loop attempted {repair_state.repair_count} time(s)."] if repair_state.repair_attempted else []),
        ],
        next_action_recommendation=recommendation,
        consumed_budget=budget,
        repair_attempted=repair_state.repair_attempted,
        repair_count=repair_state.repair_count,
        repair_outcome="failed_after_repairs" if repair_state.repair_attempted else "not_attempted",
        failure_classification=classification.to_dict(),
    )
