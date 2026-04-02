import json
from pathlib import Path

from auto_llm_innovator.datasets import dataset_plan_for_phase
from auto_llm_innovator.design_ir import compile_design_ir, project_idea_spec
from auto_llm_innovator.env import EnvironmentReport
from auto_llm_innovator.generation import generate_idea_package
from auto_llm_innovator.handoff import load_research_idea_bundle
from auto_llm_innovator.idea_spec import review_originality
from auto_llm_innovator.orchestration import InnovatorEngine
from auto_llm_innovator.repair import classify_runtime_failure
from auto_llm_innovator.runtime import default_runtime_settings_for_phase
from auto_llm_innovator.training import execute_phase


def _write_phase_config(idea_dir: Path, phase: str, *, max_retries_visible: int) -> None:
    config_dir = idea_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / f"{phase}.json").write_text(
        json.dumps(
            {
                "phase": phase,
                "target_parameters": 600_000_000,
                "prefer_rocm": True,
                "dataset": dataset_plan_for_phase(phase),
                "runtime": default_runtime_settings_for_phase(phase).to_dict(),
                "novelty_claims": ["repair coverage"],
                "max_retries_visible": max_retries_visible,
            }
        ),
        encoding="utf-8",
    )


def _build_generated_idea(tmp_path: Path, *, idea_id: str) -> Path:
    idea_dir = tmp_path / "ideas" / idea_id
    idea_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_research_idea_bundle(
        raw_brief="Invent a recurrent retrieval decoder with cache-aware routing and explicit memory."
    )
    design_ir = compile_design_ir(bundle, idea_id=idea_id)
    spec = project_idea_spec(design_ir, bundle)
    review_originality(spec)
    (idea_dir / "design_ir.json").write_text(json.dumps(design_ir.to_dict()), encoding="utf-8")
    (idea_dir / "idea_spec.json").write_text(json.dumps(spec.to_dict()), encoding="utf-8")
    generate_idea_package(idea_dir, spec, design_ir)
    _write_phase_config(idea_dir, "smoke", max_retries_visible=2)
    return idea_dir


def test_preflight_import_failure_is_repaired_and_phase_passes(tmp_path: Path):
    idea_dir = _build_generated_idea(tmp_path, idea_id="idea-repair-0001")
    (idea_dir / "package" / "plugin.py").unlink()

    result = execute_phase(idea_dir=idea_dir, attempt_id="attempt-0001", phase="smoke", run_dir=idea_dir / "runs" / "attempt-0001" / "smoke")

    assert result.status == "passed"
    assert result.repair_attempted is True
    assert result.repair_outcome == "recovered"
    assert result.failure_classification["category"] == "package_import_failure"
    assert (idea_dir / "runs" / "attempt-0001" / "smoke" / "repair" / "repair-history.json").exists()


def test_invalid_runtime_settings_are_normalized_before_preflight(tmp_path: Path):
    idea_dir = _build_generated_idea(tmp_path, idea_id="idea-repair-0002")
    config_path = idea_dir / "config" / "smoke.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["runtime"]["max_steps"] = 0
    payload["runtime"]["evaluation_scope"] = "broken"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    result = execute_phase(idea_dir=idea_dir, attempt_id="attempt-0001", phase="smoke", run_dir=idea_dir / "runs" / "attempt-0001" / "smoke")
    repaired = json.loads(config_path.read_text(encoding="utf-8"))

    assert result.status == "passed"
    assert result.repair_attempted is True
    assert repaired["runtime"]["max_steps"] > 0
    assert repaired["runtime"]["evaluation_scope"] == "minimal"


def test_runtime_output_shape_failure_is_repaired_and_rerun(tmp_path: Path):
    idea_dir = _build_generated_idea(tmp_path, idea_id="idea-repair-0003")
    model_path = idea_dir / "package" / "modeling" / "model.py"
    model_path.write_text(
        model_path.read_text(encoding="utf-8").replace('output = {"logits": logits}', 'output = {"broken": logits}'),
        encoding="utf-8",
    )

    result = execute_phase(idea_dir=idea_dir, attempt_id="attempt-0001", phase="smoke", run_dir=idea_dir / "runs" / "attempt-0001" / "smoke")

    assert result.status == "passed"
    assert result.repair_attempted is True
    assert result.failure_classification["category"] == "runtime_output_shape_failure"


def test_unknown_runtime_failure_is_non_repairable():
    classification = classify_runtime_failure(
        {
            "status": "failed",
            "failure_signals": ["Unexpected runtime edge case."],
            "next_action_recommendation": "manual_review",
            "consumed_budget": {"stop_reason": "mystery_failure"},
        }
    )

    assert classification.category == "unknown_runtime_failure"
    assert classification.repairable is False


def test_engine_repaired_run_persists_lineage_manifest(tmp_path: Path):
    baseline_dir = tmp_path / "baselines" / "internal_reference"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    (baseline_dir / "manifest.json").write_text(
        json.dumps({"baseline_id": "internal-reference-v1", "reference_metrics": {"smoke.loss": 6.0}}),
        encoding="utf-8",
    )
    engine = InnovatorEngine(root=tmp_path)
    engine.opencode.executable = "definitely-missing-opencode"
    submit = engine.submit("Invent a recurrent retrieval decoder with cache-aware routing and explicit memory.")
    idea_dir = tmp_path / "ideas" / submit.idea_id
    environment = EnvironmentReport(
        torch_available=True,
        accelerator_backend="cuda",
        rocm_available=False,
        device_count=1,
        gpu_names=["cuda-gpu-0"],
        vram_bytes_per_device=[24_000_000_000],
        cpu_count=8,
        system_ram_bytes=64_000_000_000,
        free_disk_bytes=400_000_000_000,
        default_dtype="float32",
        torch_version="2.8.0",
        platform_system="Darwin",
        platform_machine="arm64",
        message="test environment",
    )
    (idea_dir / "environment.json").write_text(json.dumps(environment.to_dict()), encoding="utf-8")
    model_path = idea_dir / "package" / "modeling" / "model.py"
    model_path.write_text(
        model_path.read_text(encoding="utf-8").replace('output = {"logits": logits}', 'output = {"broken": logits}'),
        encoding="utf-8",
    )

    payload = engine.run(submit.idea_id, phase="smoke")
    lineage_path = idea_dir / "runs" / payload["attempt_id"] / "smoke" / "lineage-manifest.json"
    manifest = json.loads(lineage_path.read_text(encoding="utf-8"))

    assert lineage_path.exists()
    assert manifest["repair"]["repair_attempted"] is True
    assert any(artifact["kind"] == "repair_diff" for artifact in manifest["repair"]["artifacts"] if not artifact.get("missing"))
    assert any(path.endswith("agents/planner-request.json") for path in payload["phases"][0]["artifacts_produced"])
    assert any(path.endswith("agents/reviewer-response.json") for path in payload["phases"][0]["artifacts_produced"])
