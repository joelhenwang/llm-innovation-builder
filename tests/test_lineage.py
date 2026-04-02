import json
from pathlib import Path

from auto_llm_innovator.env import EnvironmentReport
from auto_llm_innovator.modeling.interfaces import PhaseResult
from auto_llm_innovator.orchestration import InnovatorEngine
from auto_llm_innovator.tracking import (
    build_phase_lineage_manifest,
    hash_directory,
    hash_file,
    hash_json_payload,
    persist_phase_lineage_manifest,
)


def _bootstrap_baseline(root: Path) -> None:
    baseline_dir = root / "baselines" / "internal_reference"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    (baseline_dir / "manifest.json").write_text(
        json.dumps(
            {
                "baseline_id": "internal-reference-v1",
                "family": "internal_reference",
                "label": "Internal Reference",
                "metric_targets": [
                    {"phase": "smoke", "metric_name": "loss", "target_value": 6.0},
                    {"phase": "small", "metric_name": "val_loss", "target_value": 4.2},
                    {"phase": "full", "metric_name": "val_loss", "target_value": 3.7},
                ],
                "token_budget_assumptions": {
                    "smoke": 50_000,
                    "small": 3_000_000,
                    "full": 12_000_000,
                },
            }
        ),
        encoding="utf-8",
    )


def _environment(*, backend: str, vram_bytes_per_device: list[int] | None = None, system_ram_bytes: int = 64_000_000_000) -> EnvironmentReport:
    return EnvironmentReport(
        torch_available=backend != "none",
        accelerator_backend=backend,
        rocm_available=backend == "rocm",
        device_count=len(vram_bytes_per_device or []),
        gpu_names=[f"{backend}-gpu-{index}" for index in range(len(vram_bytes_per_device or []))],
        vram_bytes_per_device=list(vram_bytes_per_device or []),
        cpu_count=8,
        system_ram_bytes=system_ram_bytes,
        free_disk_bytes=400_000_000_000,
        default_dtype="float32",
        torch_version="2.8.0" if backend != "none" else None,
        platform_system="Darwin",
        platform_machine="arm64",
        message="test environment",
    )


def test_hashes_are_stable_for_text_and_json_payloads(tmp_path: Path):
    text_path = tmp_path / "artifact.txt"
    text_path.write_text("lineage\n", encoding="utf-8")

    first = hash_file(text_path)
    second = hash_file(text_path)

    assert first.sha256 == second.sha256
    assert hash_json_payload({"b": 2, "a": 1}) == hash_json_payload({"a": 1, "b": 2})


def test_directory_hash_is_stable_regardless_of_creation_order(tmp_path: Path):
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "b.txt").write_text("b", encoding="utf-8")
    (root / "a.txt").write_text("a", encoding="utf-8")
    first = hash_directory(root, relative_to=tmp_path)

    (root / "a.txt").unlink()
    (root / "b.txt").unlink()
    (root / "a.txt").write_text("a", encoding="utf-8")
    (root / "b.txt").write_text("b", encoding="utf-8")
    second = hash_directory(root, relative_to=tmp_path)

    assert first.sha256 == second.sha256
    assert first.files == second.files


def test_engine_persists_executed_phase_lineage_manifest(tmp_path: Path):
    _bootstrap_baseline(tmp_path)
    engine = InnovatorEngine(root=tmp_path)
    engine.opencode.executable = "definitely-missing-opencode"
    submit = engine.submit("Invent a recurrent retrieval decoder with cache-aware routing and explicit memory.")
    idea_dir = tmp_path / "ideas" / submit.idea_id
    (idea_dir / "environment.json").write_text(
        json.dumps(_environment(backend="cuda", vram_bytes_per_device=[24_000_000_000]).to_dict()),
        encoding="utf-8",
    )

    payload = engine.run(submit.idea_id, phase="smoke")
    phase_dir = idea_dir / "runs" / payload["attempt_id"] / "smoke"
    manifest = json.loads((phase_dir / "lineage-manifest.json").read_text(encoding="utf-8"))

    assert manifest["status"] == "executed"
    assert manifest["planning"]["resolved_config"]["path"] == f"runs/{payload['attempt_id']}/smoke/resolved-config.json"
    assert manifest["generation"]["bundle"]["sha256"]
    assert str(phase_dir / "lineage-manifest.json") in payload["phases"][0]["artifacts_produced"]


def test_engine_persists_rejected_phase_lineage_manifest(tmp_path: Path):
    _bootstrap_baseline(tmp_path)
    engine = InnovatorEngine(root=tmp_path)
    engine.opencode.executable = "definitely-missing-opencode"
    submit = engine.submit("Invent a recurrent retrieval decoder with cache-aware routing and explicit memory.")
    idea_dir = tmp_path / "ideas" / submit.idea_id
    (idea_dir / "environment.json").write_text(
        json.dumps(_environment(backend="cpu", system_ram_bytes=8_000_000_000).to_dict()),
        encoding="utf-8",
    )

    payload = engine.run(submit.idea_id, phase="full")
    phase_dir = idea_dir / "runs" / payload["attempt_id"] / "full"
    manifest = json.loads((phase_dir / "lineage-manifest.json").read_text(encoding="utf-8"))

    assert manifest["status"] == "rejected_before_execution"
    assert manifest["result"]["status"] == "failed"
    assert manifest["planning"]["resource_plan"]["path"] == f"runs/{payload['attempt_id']}/full/resource-plan.json"


def test_repaired_phase_lineage_manifest_contains_repair_artifacts(tmp_path: Path):
    _bootstrap_baseline(tmp_path)
    engine = InnovatorEngine(root=tmp_path)
    engine.opencode.executable = "definitely-missing-opencode"
    submit = engine.submit("Invent a recurrent retrieval decoder with cache-aware routing and explicit memory.")
    idea_dir = tmp_path / "ideas" / submit.idea_id
    (idea_dir / "environment.json").write_text(
        json.dumps(_environment(backend="cuda", vram_bytes_per_device=[24_000_000_000]).to_dict()),
        encoding="utf-8",
    )
    model_path = idea_dir / "package" / "modeling" / "model.py"
    model_path.write_text(
        model_path.read_text(encoding="utf-8").replace('output = {"logits": logits}', 'output = {"broken": logits}'),
        encoding="utf-8",
    )

    payload = engine.run(submit.idea_id, phase="smoke")
    phase_dir = idea_dir / "runs" / payload["attempt_id"] / "smoke"
    manifest = json.loads((phase_dir / "lineage-manifest.json").read_text(encoding="utf-8"))
    kinds = {artifact["kind"] for artifact in manifest["repair"]["artifacts"] if not artifact.get("missing")}

    assert manifest["repair"]["repair_attempted"] is True
    assert "repair_before_snapshot" in kinds
    assert "repair_after_snapshot" in kinds
    assert "repair_diff" in kinds


def test_manifest_records_missing_generated_files_without_crashing(tmp_path: Path):
    _bootstrap_baseline(tmp_path)
    engine = InnovatorEngine(root=tmp_path)
    engine.opencode.executable = "definitely-missing-opencode"
    submit = engine.submit("Invent a recurrent retrieval decoder with cache-aware routing and explicit memory.")
    idea_dir = tmp_path / "ideas" / submit.idea_id
    run_dir = idea_dir / "runs" / "attempt-0001" / "smoke"
    run_dir.mkdir(parents=True, exist_ok=True)
    resource_plan_path = run_dir / "resource-plan.json"
    dataset_plan_path = run_dir / "dataset-plan.json"
    resolved_config_path = run_dir / "resolved-config.json"
    resource_plan_path.write_text("{}", encoding="utf-8")
    dataset_plan_path.write_text("{}", encoding="utf-8")
    resolved_config_path.write_text(json.dumps(json.loads((idea_dir / "config" / "smoke.json").read_text(encoding="utf-8"))), encoding="utf-8")
    (idea_dir / "package" / "plugin.py").unlink()

    manifest = build_phase_lineage_manifest(
        idea_dir=idea_dir,
        run_dir=run_dir,
        idea_id=submit.idea_id,
        attempt_id="attempt-0001",
        phase="smoke",
        lineage_status="planned",
        environment=EnvironmentReport.from_dict(json.loads((idea_dir / "environment.json").read_text(encoding="utf-8"))),
        result=PhaseResult(
            idea_id=submit.idea_id,
            attempt_id="attempt-0001",
            phase="smoke",
            status="failed",
            key_metrics={},
            failure_signals=[],
            artifacts_produced=[],
            reviewer_notes=[],
            next_action_recommendation="manual_review",
            consumed_budget={},
        ),
        resource_plan_path=resource_plan_path,
        dataset_plan_path=dataset_plan_path,
        resolved_config_path=resolved_config_path,
    )
    persist_phase_lineage_manifest(run_dir, manifest)

    plugin_records = [record for record in manifest.generation["generated_files"] if record["path"] == "package/plugin.py"]

    assert plugin_records
    assert plugin_records[0]["missing"] is True
    assert (run_dir / "lineage-manifest.json").exists()
