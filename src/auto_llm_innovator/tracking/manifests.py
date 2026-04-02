from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from auto_llm_innovator.env import EnvironmentReport
from auto_llm_innovator.filesystem import read_json, write_json
from auto_llm_innovator.modeling.interfaces import PhaseResult

from .lineage import collect_artifact_record, hash_json_payload


@dataclass(slots=True)
class PhaseLineageManifest:
    idea_id: str
    attempt_id: str
    phase: str
    status: str
    seeds: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    planning: dict[str, Any] = field(default_factory=dict)
    generation: dict[str, Any] = field(default_factory=dict)
    repair: dict[str, Any] = field(default_factory=dict)
    outputs: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_phase_lineage_manifest(
    *,
    idea_dir: Path,
    run_dir: Path,
    idea_id: str,
    attempt_id: str,
    phase: str,
    lineage_status: str,
    environment: EnvironmentReport,
    result: PhaseResult,
    resource_plan_path: Path,
    dataset_plan_path: Path,
    resolved_config_path: Path,
) -> PhaseLineageManifest:
    resolved_config = _read_json_if_present(resolved_config_path)
    generation_manifest_path = idea_dir / "generation_manifest.json"
    generation_manifest = _read_json_if_present(generation_manifest_path)
    generated_files = [str(item) for item in generation_manifest.get("generated_files", [])]
    generated_records = [
        collect_artifact_record(idea_dir / relative_path, kind="generated_file", relative_to=idea_dir)
        for relative_path in generated_files
    ]
    generation_manifest_record = collect_artifact_record(
        generation_manifest_path,
        kind="generation_manifest",
        relative_to=idea_dir,
    )
    key_files = [
        collect_artifact_record(idea_dir / relative_path, kind="generated_key_file", relative_to=idea_dir)
        for relative_path in ("train.py", "eval.py", "package/__init__.py", "package/plugin.py")
    ]

    repair_records = _collect_repair_artifacts(run_dir=run_dir, idea_dir=idea_dir)
    output_records = _collect_output_artifacts(result=result, idea_dir=idea_dir)

    bundle_inputs = {
        record["path"]: _bundle_hash_value(record)
        for record in [*generated_records, generation_manifest_record]
    }

    return PhaseLineageManifest(
        idea_id=idea_id,
        attempt_id=attempt_id,
        phase=phase,
        status=lineage_status,
        seeds=_seed_fields_from_resolved_config(resolved_config),
        environment=environment.to_dict(),
        planning={
            "resolved_config": collect_artifact_record(resolved_config_path, kind="resolved_config", relative_to=idea_dir),
            "resource_plan": collect_artifact_record(resource_plan_path, kind="resource_plan", relative_to=idea_dir),
            "dataset_plan": collect_artifact_record(dataset_plan_path, kind="dataset_plan", relative_to=idea_dir),
        },
        generation={
            "bundle": {
                "path": ".",
                "kind": "generated_bundle",
                "sha256": hash_json_payload(bundle_inputs),
                "file_count": len(bundle_inputs),
            },
            "manifest": generation_manifest_record,
            "generated_files": generated_records,
            "key_files": key_files,
            "generated_file_paths": generated_files,
        },
        repair={
            "repair_attempted": result.repair_attempted,
            "repair_count": result.repair_count,
            "failure_classification": result.failure_classification,
            "artifacts": repair_records,
        },
        outputs=output_records,
        result={
            "status": result.status,
            "key_metrics": dict(result.key_metrics),
            "failure_signals": list(result.failure_signals),
            "next_action_recommendation": result.next_action_recommendation,
            "consumed_budget": dict(result.consumed_budget),
        },
    )


def persist_phase_lineage_manifest(run_dir: Path, manifest: PhaseLineageManifest) -> Path:
    path = run_dir / "lineage-manifest.json"
    write_json(path, manifest.to_dict())
    return path


def _read_json_if_present(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return read_json(path)
    except Exception:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}


def _seed_fields_from_resolved_config(resolved_config: dict[str, Any]) -> dict[str, Any]:
    runtime_payload = dict(resolved_config.get("runtime", {}))
    return {
        key: value
        for key, value in runtime_payload.items()
        if "seed" in key.lower()
    }


def _collect_repair_artifacts(*, run_dir: Path, idea_dir: Path) -> list[dict[str, Any]]:
    repair_dir = run_dir / "repair"
    if not repair_dir.exists():
        return []
    records = [
        collect_artifact_record(repair_dir / "failure-classification.json", kind="repair_failure_classification", relative_to=idea_dir),
        collect_artifact_record(repair_dir / "repair-history.json", kind="repair_history", relative_to=idea_dir),
    ]
    for path in sorted(repair_dir.glob("*-diff.patch")):
        records.append(collect_artifact_record(path, kind="repair_diff", relative_to=idea_dir))
    for path in sorted(repair_dir.glob("*-rationale.json")):
        records.append(collect_artifact_record(path, kind="repair_rationale", relative_to=idea_dir))
    for path in sorted(repair_dir.glob("*-before")):
        records.append(collect_artifact_record(path, kind="repair_before_snapshot", relative_to=idea_dir))
    for path in sorted(repair_dir.glob("*-after")):
        records.append(collect_artifact_record(path, kind="repair_after_snapshot", relative_to=idea_dir))
    return records


def _collect_output_artifacts(*, result: PhaseResult, idea_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for artifact in result.artifacts_produced:
        path = Path(artifact)
        if not path.is_absolute():
            path = idea_dir / artifact
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        records.append(collect_artifact_record(path, kind="phase_output", relative_to=idea_dir))
    return records


def _bundle_hash_value(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("missing"):
        return {"missing": True}
    payload = {"sha256": record["sha256"]}
    if "bytes" in record:
        payload["bytes"] = record["bytes"]
    if "file_count" in record:
        payload["file_count"] = record["file_count"]
    return payload
