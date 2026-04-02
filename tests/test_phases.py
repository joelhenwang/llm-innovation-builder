import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from auto_llm_innovator.datasets import dataset_plan_for_phase
from auto_llm_innovator.design_ir import compile_design_ir
from auto_llm_innovator.handoff import load_research_idea_bundle
from auto_llm_innovator.orchestration import InnovatorEngine
from auto_llm_innovator.runtime import compile_runtime_phase_config, default_runtime_settings_for_phase, run_phase_with_plugin


def _bundle_design_ir():
    return compile_design_ir(
        load_research_idea_bundle(raw_brief="Invent a recurrent retrieval decoder with cache-aware routing and explicit memory."),
        idea_id="idea-0001",
    )


def _phase_payload(phase: str) -> dict:
    return {
        "phase": phase,
        "target_parameters": 600_000_000,
        "prefer_rocm": True,
        "dataset": dataset_plan_for_phase(phase),
        "runtime": default_runtime_settings_for_phase(phase).to_dict(),
        "novelty_claims": ["phase semantics"],
    }


def test_phase_json_compiles_to_distinct_runtime_settings():
    design_ir = _bundle_design_ir()
    smoke = compile_runtime_phase_config(design_ir, _phase_payload("smoke"), attempt_id="attempt-0001", phase="smoke")
    small = compile_runtime_phase_config(design_ir, _phase_payload("small"), attempt_id="attempt-0001", phase="small")
    full = compile_runtime_phase_config(design_ir, _phase_payload("full"), attempt_id="attempt-0001", phase="full")

    assert smoke.settings.max_steps == 2
    assert small.settings.max_steps == 6
    assert full.settings.max_steps == 12
    assert smoke.settings.sequence_length < small.settings.sequence_length < full.settings.sequence_length
    assert smoke.settings.batch_size < small.settings.batch_size < full.settings.batch_size
    assert smoke.checkpoints.resume is False
    assert small.checkpoints.resume is True
    assert full.checkpoints.every_n_steps == 4
    assert len(smoke.evaluation.tasks) <= len(small.evaluation.tasks)


def test_phase_runtime_summary_includes_stop_reason_and_runtime_settings(tmp_path: Path):
    idea_dir = tmp_path / "ideas" / "idea-0001"
    (idea_dir / "config").mkdir(parents=True, exist_ok=True)
    design_ir = _bundle_design_ir()
    (idea_dir / "design_ir.json").write_text(json.dumps(design_ir.to_dict()), encoding="utf-8")
    config_path = idea_dir / "config" / "smoke.json"
    config_path.write_text(json.dumps(_phase_payload("smoke")), encoding="utf-8")
    run_dir = idea_dir / "runs" / "attempt-0001" / "smoke"

    @dataclass(slots=True)
    class ModelConfig:
        architecture_name: str = "runtime-test"

    class DemoModel:
        def forward(self, input_ids, state_tensor=None, memory_tensor=None, cache_tensor=None):
            batch = len(input_ids)
            seq = len(input_ids[0])
            logits = [[[0.0 for _ in range(16)] for _ in range(seq)] for _ in range(batch)]
            return {"logits": logits, "state_tensor": state_tensor, "memory_tensor": memory_tensor, "cache_tensor": cache_tensor}

    plugin_module = SimpleNamespace(
        ModelConfig=ModelConfig,
        build_model=lambda config=None: DemoModel(),
        describe_plugin=lambda: {
            "module_names": ["token_embedding", "core_backbone", "state_adapter", "memory_adapter", "routing_cache", "lm_head"],
            "supports": {"recurrent_state": True, "external_memory": True, "cache_path": True},
        },
        register_evaluation_hooks=lambda: {},
    )

    result = run_phase_with_plugin(
        phase="smoke",
        idea_dir=str(idea_dir),
        run_dir=str(run_dir),
        config_path=str(config_path),
        attempt_id="attempt-0001",
        plugin_module=plugin_module,
    )
    summary = json.loads((run_dir / "smoke-summary.json").read_text(encoding="utf-8"))

    assert result["status"] == "passed"
    assert summary["stop_reason"] == "max_steps_reached"
    assert summary["runtime"]["settings"]["max_steps"] == 2
    assert summary["runtime"]["settings"]["evaluation_scope"] == "minimal"


def test_runtime_wall_time_limit_marks_phase_passed_with_warnings(tmp_path: Path):
    idea_dir = tmp_path / "ideas" / "idea-0001"
    (idea_dir / "config").mkdir(parents=True, exist_ok=True)
    design_ir = _bundle_design_ir()
    (idea_dir / "design_ir.json").write_text(json.dumps(design_ir.to_dict()), encoding="utf-8")
    payload = _phase_payload("small")
    payload["runtime"]["max_wall_time_seconds"] = 0
    config_path = idea_dir / "config" / "small.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    run_dir = idea_dir / "runs" / "attempt-0001" / "small"

    @dataclass(slots=True)
    class ModelConfig:
        architecture_name: str = "runtime-test"

    class DemoModel:
        def forward(self, input_ids, state_tensor=None, memory_tensor=None, cache_tensor=None):
            batch = len(input_ids)
            seq = len(input_ids[0])
            logits = [[[0.0 for _ in range(16)] for _ in range(seq)] for _ in range(batch)]
            return {"logits": logits, "state_tensor": state_tensor, "memory_tensor": memory_tensor, "cache_tensor": cache_tensor}

    plugin_module = SimpleNamespace(
        ModelConfig=ModelConfig,
        build_model=lambda config=None: DemoModel(),
        describe_plugin=lambda: {
            "module_names": ["token_embedding", "core_backbone", "state_adapter", "memory_adapter", "routing_cache", "lm_head"],
            "supports": {"recurrent_state": True, "external_memory": True, "cache_path": True},
        },
        register_evaluation_hooks=lambda: {},
    )

    result = run_phase_with_plugin(
        phase="small",
        idea_dir=str(idea_dir),
        run_dir=str(run_dir),
        config_path=str(config_path),
        attempt_id="attempt-0001",
        plugin_module=plugin_module,
    )

    assert result["status"] == "passed_with_warnings"
    assert result["consumed_budget"]["stop_reason"] == "max_wall_time_reached"


def test_submit_writes_runtime_block_for_each_phase(tmp_path: Path):
    baseline_dir = tmp_path / "baselines" / "internal_reference"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    (baseline_dir / "manifest.json").write_text(json.dumps({"baseline_id": "x", "reference_metrics": {}}), encoding="utf-8")
    engine = InnovatorEngine(root=tmp_path)
    result = engine.submit("Invent a recurrent retrieval decoder with cache-aware routing and explicit memory.")
    idea_dir = tmp_path / "ideas" / result.idea_id

    smoke = json.loads((idea_dir / "config" / "smoke.json").read_text(encoding="utf-8"))
    small = json.loads((idea_dir / "config" / "small.json").read_text(encoding="utf-8"))
    full = json.loads((idea_dir / "config" / "full.json").read_text(encoding="utf-8"))

    assert smoke["runtime"]["resume_enabled"] is False
    assert smoke["runtime"]["evaluation_scope"] == "minimal"
    assert small["runtime"]["checkpoint_every_steps"] == 3
    assert full["runtime"]["max_steps"] == 12


def test_direct_small_phase_run_does_not_backfill_smoke(tmp_path: Path):
    baseline_dir = tmp_path / "baselines" / "internal_reference"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    (baseline_dir / "manifest.json").write_text(json.dumps({"baseline_id": "x", "reference_metrics": {}}), encoding="utf-8")
    engine = InnovatorEngine(root=tmp_path)
    result = engine.submit("Invent a recurrent retrieval decoder with cache-aware routing and explicit memory.")

    payload = engine.run(result.idea_id, phase="small")

    assert [phase["phase"] for phase in payload["phases"]] == ["small"]


def test_passed_with_warnings_stops_all_phase_promotion(tmp_path: Path):
    baseline_dir = tmp_path / "baselines" / "internal_reference"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    (baseline_dir / "manifest.json").write_text(json.dumps({"baseline_id": "x", "reference_metrics": {}}), encoding="utf-8")
    engine = InnovatorEngine(root=tmp_path)
    result = engine.submit("Invent a recurrent retrieval decoder with cache-aware routing and explicit memory.")
    idea_dir = tmp_path / "ideas" / result.idea_id
    small_config_path = idea_dir / "config" / "small.json"
    small_config = json.loads(small_config_path.read_text(encoding="utf-8"))
    small_config["runtime"]["max_wall_time_seconds"] = 0
    small_config_path.write_text(json.dumps(small_config), encoding="utf-8")

    payload = engine.run(result.idea_id, phase="all")
    status = engine.status(result.idea_id)

    assert [phase["phase"] for phase in payload["phases"]] == ["smoke", "small"]
    assert payload["phases"][-1]["status"] == "passed_with_warnings"
    assert "full" not in status["attempts"][0]["phases"]
