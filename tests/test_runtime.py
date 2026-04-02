import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from auto_llm_innovator.datasets import dataset_plan_for_phase
from auto_llm_innovator.design_ir import compile_design_ir
from auto_llm_innovator.handoff import load_research_idea_bundle
from auto_llm_innovator.runtime import compile_runtime_phase_config, default_runtime_settings_for_phase, run_phase_with_plugin


def _write_runtime_inputs(root: Path, *, phase: str = "smoke") -> tuple[Path, Path]:
    idea_dir = root / "ideas" / "idea-0001"
    (idea_dir / "config").mkdir(parents=True, exist_ok=True)
    bundle = load_research_idea_bundle(
        raw_brief="Invent a recurrent retrieval decoder with cache-aware routing and explicit memory."
    )
    design_ir = compile_design_ir(bundle, idea_id="idea-0001")
    (idea_dir / "design_ir.json").write_text(json.dumps(design_ir.to_dict()), encoding="utf-8")
    phase_config = {
        "phase": phase,
        "target_parameters": 600_000_000,
        "prefer_rocm": True,
        "dataset": dataset_plan_for_phase(phase),
        "runtime": default_runtime_settings_for_phase(phase).to_dict(),
        "novelty_claims": ["runtime integration"],
    }
    config_path = idea_dir / "config" / f"{phase}.json"
    config_path.write_text(json.dumps(phase_config), encoding="utf-8")
    return idea_dir, config_path


def test_runtime_config_derives_state_memory_and_cache_expectations(tmp_path: Path):
    idea_dir, config_path = _write_runtime_inputs(tmp_path)
    design_ir = compile_design_ir(
        load_research_idea_bundle(raw_brief="Invent a recurrent retrieval decoder with cache-aware routing and explicit memory."),
        idea_id="idea-0001",
    )
    phase_config = json.loads(config_path.read_text(encoding="utf-8"))

    runtime_config = compile_runtime_phase_config(design_ir, phase_config, attempt_id="attempt-0001", phase="smoke")

    assert runtime_config.plugin.has_recurrent_state is True
    assert runtime_config.plugin.has_external_memory is True
    assert runtime_config.plugin.has_cache_path is True
    assert runtime_config.optimizer.name == "AdamW"
    assert runtime_config.scheduler.name == "constant"
    assert runtime_config.gradient_accumulation_steps == 1
    assert runtime_config.settings.max_steps == 2
    assert runtime_config.settings.evaluation_scope == "minimal"
    assert runtime_config.checkpoints.resume is False


def test_runtime_fails_missing_plugin_hooks_with_readable_failure_signals(tmp_path: Path):
    idea_dir, config_path = _write_runtime_inputs(tmp_path)
    run_dir = idea_dir / "runs" / "attempt-0001" / "smoke"
    plugin_module = SimpleNamespace(ModelConfig=object)

    result = run_phase_with_plugin(
        phase="smoke",
        idea_dir=str(idea_dir),
        run_dir=str(run_dir),
        config_path=str(config_path),
        attempt_id="attempt-0001",
        plugin_module=plugin_module,
    )

    assert result["status"] == "failed"
    assert any("build_model" in item for item in result["failure_signals"])
    assert (run_dir / "smoke-summary.json").exists()


def test_runtime_writes_checkpoint_and_resumes_same_phase_directory(tmp_path: Path):
    idea_dir, config_path = _write_runtime_inputs(tmp_path, phase="small")
    run_dir = idea_dir / "runs" / "attempt-0001" / "small"

    @dataclass(slots=True)
    class ModelConfig:
        architecture_name: str = "runtime-test"

    class DemoModel:
        def forward(self, input_ids, state_tensor=None, memory_tensor=None, cache_tensor=None):
            batch = len(input_ids)
            seq = len(input_ids[0])
            logits = [[[0.0 for _ in range(16)] for _ in range(seq)] for _ in range(batch)]
            return {
                "logits": logits,
                "state_tensor": state_tensor,
                "memory_tensor": memory_tensor,
                "cache_tensor": cache_tensor,
            }

    plugin_module = SimpleNamespace(
        ModelConfig=ModelConfig,
        build_model=lambda config=None: DemoModel(),
        describe_plugin=lambda: {
            "module_names": ["token_embedding", "core_backbone", "state_adapter", "memory_adapter", "routing_cache", "lm_head"],
            "supports": {"recurrent_state": True, "external_memory": True, "cache_path": True},
        },
        register_evaluation_hooks=lambda: {},
    )

    first = run_phase_with_plugin(
        phase="small",
        idea_dir=str(idea_dir),
        run_dir=str(run_dir),
        config_path=str(config_path),
        attempt_id="attempt-0001",
        plugin_module=plugin_module,
    )
    second = run_phase_with_plugin(
        phase="small",
        idea_dir=str(idea_dir),
        run_dir=str(run_dir),
        config_path=str(config_path),
        attempt_id="attempt-0001",
        plugin_module=plugin_module,
    )

    assert first["status"] == "passed"
    assert first["consumed_budget"]["device"] == "cpu-dry-run"
    assert (run_dir / "checkpoint.json").exists()
    assert (run_dir / "evaluation-report.json").exists()
    assert second["consumed_budget"]["resumed"] is True
    assert any("Resumed from existing checkpoint metadata." in item for item in second["reviewer_notes"])


def test_runtime_smoke_does_not_resume_same_phase_directory(tmp_path: Path):
    idea_dir, config_path = _write_runtime_inputs(tmp_path, phase="smoke")
    run_dir = idea_dir / "runs" / "attempt-0001" / "smoke"

    @dataclass(slots=True)
    class ModelConfig:
        architecture_name: str = "runtime-test"

    class DemoModel:
        def forward(self, input_ids, state_tensor=None, memory_tensor=None, cache_tensor=None):
            batch = len(input_ids)
            seq = len(input_ids[0])
            logits = [[[0.0 for _ in range(16)] for _ in range(seq)] for _ in range(batch)]
            return {
                "logits": logits,
                "state_tensor": state_tensor,
                "memory_tensor": memory_tensor,
                "cache_tensor": cache_tensor,
            }

    plugin_module = SimpleNamespace(
        ModelConfig=ModelConfig,
        build_model=lambda config=None: DemoModel(),
        describe_plugin=lambda: {
            "module_names": ["token_embedding", "core_backbone", "state_adapter", "memory_adapter", "routing_cache", "lm_head"],
            "supports": {"recurrent_state": True, "external_memory": True, "cache_path": True},
        },
        register_evaluation_hooks=lambda: {},
    )

    first = run_phase_with_plugin(
        phase="smoke",
        idea_dir=str(idea_dir),
        run_dir=str(run_dir),
        config_path=str(config_path),
        attempt_id="attempt-0001",
        plugin_module=plugin_module,
    )
    second = run_phase_with_plugin(
        phase="smoke",
        idea_dir=str(idea_dir),
        run_dir=str(run_dir),
        config_path=str(config_path),
        attempt_id="attempt-0001",
        plugin_module=plugin_module,
    )

    assert first["status"] == "passed"
    assert second["consumed_budget"]["resumed"] is False
