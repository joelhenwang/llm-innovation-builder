from __future__ import annotations

from textwrap import dedent

from auto_llm_innovator.design_ir.models import DesignIR
from auto_llm_innovator.generation.layout import IdeaPackageLayout
from auto_llm_innovator.idea_spec.models import IdeaSpec


def render_idea_package_sources(layout: IdeaPackageLayout, spec: IdeaSpec, design_ir: DesignIR) -> dict[str, str]:
    supports = {
        "recurrent_state": design_ir.architecture.state_semantics.has_recurrent_state,
        "external_memory": design_ir.architecture.state_semantics.has_external_memory,
        "cache_path": design_ir.architecture.state_semantics.has_cache_path,
    }
    has_state_file = any(supports.values())
    module_names = [module.name for module in design_ir.modules]
    evaluation_task_names = [task.name for task in design_ir.evaluation_plan]

    files = {
        str(layout.package_dir / "__init__.py"): _render_package_init(),
        str(layout.package_dir / "config.py"): _render_package_config(spec, design_ir),
        str(layout.package_dir / "plugin.py"): _render_package_plugin(design_ir, module_names, supports),
        str(layout.package_modeling_dir / "__init__.py"): _render_modeling_init(),
        str(layout.package_modeling_dir / "components.py"): _render_components(spec, design_ir),
        str(layout.package_modeling_dir / "model.py"): _render_model(design_ir, has_state_file),
        str(layout.package_evaluation_dir / "__init__.py"): _render_evaluation_init(),
        str(layout.package_evaluation_dir / "hooks.py"): _render_evaluation_hooks(evaluation_task_names),
        str(layout.tests_dir / "test_imports.py"): _render_test_imports(),
        str(layout.tests_dir / "test_shapes.py"): _render_test_shapes(has_state_file),
        str(layout.tests_dir / "test_smoke.py"): _render_test_smoke(),
        str(layout.train_entrypoint): _render_train_entrypoint(),
        str(layout.eval_entrypoint): _render_eval_entrypoint(),
    }
    if has_state_file:
        files[str(layout.package_modeling_dir / "state.py")] = _render_state_helpers()
    return files


def _render_package_init() -> str:
    return dedent(
        """
        from .config import ModelConfig
        from .plugin import build_model, describe_plugin, register_evaluation_hooks

        __all__ = ["ModelConfig", "build_model", "describe_plugin", "register_evaluation_hooks"]
        """
    )


def _render_package_config(spec: IdeaSpec, design_ir: DesignIR) -> str:
    borrowed = spec.borrowed_mechanisms if spec.borrowed_mechanisms else []
    return dedent(
        f"""
        from __future__ import annotations

        from dataclasses import dataclass


        @dataclass(slots=True)
        class ModelConfig:
            idea_id: str = "{spec.idea_id}"
            architecture_name: str = "{design_ir.architecture.pattern_label}"
            target_parameters: int = {spec.estimated_parameter_budget}
            hidden_size: int = 2048
            num_layers: int = 24
            num_heads: int = 16
            tokenizer: str = "{spec.tokenizer}"
            borrowed_mechanisms: list[str] | None = None

            def __post_init__(self) -> None:
                if self.borrowed_mechanisms is None:
                    self.borrowed_mechanisms = {borrowed!r}
        """
    )


def _render_package_plugin(design_ir: DesignIR, module_names: list[str], supports: dict[str, bool]) -> str:
    ablation_names = [ablation.name for ablation in design_ir.ablation_plan]
    return dedent(
        f"""
        from __future__ import annotations

        from package.config import ModelConfig
        from package.evaluation.hooks import register_evaluation_hooks
        from package.modeling.model import CreativeIdeaModel


        def build_model(config: ModelConfig | None = None) -> CreativeIdeaModel:
            return CreativeIdeaModel(config or ModelConfig())


        def describe_plugin() -> dict:
            return {{
                "architecture_name": "{design_ir.architecture.pattern_label}",
                "module_names": {module_names!r},
                "supports": {supports!r},
                "ablation_names": {ablation_names!r},
            }}
        """
    )


def _render_modeling_init() -> str:
    return dedent(
        """
        from .model import CreativeIdeaModel

        __all__ = ["CreativeIdeaModel"]
        """
    )


def _render_components(spec: IdeaSpec, design_ir: DesignIR) -> str:
    novelty = spec.novelty_claims
    purpose_map = {module.name: module.purpose for module in design_ir.modules}
    return dedent(
        f"""
        from __future__ import annotations

        try:
            import torch
            import torch.nn as nn
        except Exception:  # pragma: no cover - optional dependency path
            torch = None
            nn = None


        MODULE_PURPOSES = {purpose_map!r}
        ARCHITECTURE_RATIONALE = {novelty!r}


        def build_numpy_like_logits(input_ids, vocab_size: int = 1024):
            batch = len(input_ids) if hasattr(input_ids, "__len__") else 1
            seq = len(input_ids[0]) if batch and hasattr(input_ids[0], "__len__") else 1
            return [[[0.0 for _ in range(vocab_size)] for _ in range(seq)] for _ in range(batch)]


        class ComponentStack(nn.Module if nn is not None else object):
            def __init__(self, hidden_size: int) -> None:
                if nn is not None:
                    super().__init__()
                    width = min(hidden_size, 256)
                    self.embedding = nn.Embedding(1024, width)
                    self.gate = nn.Linear(width, width)
                    self.head = nn.Linear(width, 1024)
                self.hidden_size = hidden_size

            def forward(self, input_ids):
                if nn is None or torch is None:
                    return build_numpy_like_logits(input_ids)
                hidden = self.embedding(torch.as_tensor(input_ids, dtype=torch.long))
                gated = torch.tanh(self.gate(hidden))
                logits = self.head(gated)
                return logits.detach().cpu().tolist()
        """
    )


def _render_model(design_ir: DesignIR, has_state_file: bool) -> str:
    lines = [
        "from __future__ import annotations",
        "",
        "from package.config import ModelConfig",
        "from package.modeling.components import ARCHITECTURE_RATIONALE, ComponentStack",
    ]
    if has_state_file:
        lines.append("from package.modeling.state import attach_state_outputs")
    lines.extend(
        [
            "",
            "",
            "class CreativeIdeaModel:",
            "    def __init__(self, config: ModelConfig) -> None:",
            "        self.config = config",
            "        self.architecture_rationale = ARCHITECTURE_RATIONALE",
            "        self.components = ComponentStack(config.hidden_size)",
            "",
            "    def forward(self, input_ids, state_tensor=None, memory_tensor=None, cache_tensor=None):",
            "        logits = self.components.forward(input_ids)",
            '        output = {"logits": logits}',
        ]
    )
    if has_state_file:
        lines.append("        return attach_state_outputs(output, state_tensor, memory_tensor, cache_tensor)")
    else:
        lines.append("        return output")
    lines.append("")
    return "\n".join(lines)


def _render_state_helpers() -> str:
    return dedent(
        """
        from __future__ import annotations


        def attach_state_outputs(output: dict, state_tensor=None, memory_tensor=None, cache_tensor=None) -> dict:
            if state_tensor is not None:
                output["state_tensor"] = state_tensor
            if memory_tensor is not None:
                output["memory_tensor"] = memory_tensor
            if cache_tensor is not None:
                output["cache_tensor"] = cache_tensor
            return output
        """
    )


def _render_evaluation_init() -> str:
    return dedent(
        """
        from .hooks import register_evaluation_hooks

        __all__ = ["register_evaluation_hooks"]
        """
    )


def _render_evaluation_hooks(task_names: list[str]) -> str:
    lines = [
        "from __future__ import annotations",
        "",
        "",
        "def register_evaluation_hooks() -> dict:",
        "    hooks = {}",
    ]
    for task_name in task_names or ["evaluation_task_1"]:
        lines.extend(
            [
                f"    def {task_name}(**kwargs) -> dict:",
                f"        return {{'hook': '{task_name}', 'loss_snapshot': round(kwargs['loss_value'], 4)}}",
                "",
                f"    hooks['{task_name}'] = {task_name}",
            ]
        )
    lines.extend(["", "    return hooks", ""])
    return "\n".join(lines)


def _render_test_imports() -> str:
    return dedent(
        """
        from package import ModelConfig, build_model, describe_plugin, register_evaluation_hooks


        def test_package_contract_exports():
            assert ModelConfig is not None
            assert callable(build_model)
            assert callable(describe_plugin)
            assert callable(register_evaluation_hooks)
        """
    )


def _render_test_shapes(has_state_file: bool) -> str:
    kwargs = ", state_tensor=[[0.0]], memory_tensor=[[0.0]], cache_tensor=[[0.0]]" if has_state_file else ""
    return dedent(
        f"""
        from package import build_model


        def test_generated_model_returns_logits_dict():
            model = build_model()
            output = model.forward([[1, 2, 3]]{kwargs})
            assert "logits" in output
            assert isinstance(output["logits"], list)
        """
    )


def _render_test_smoke() -> str:
    return dedent(
        """
        from package import describe_plugin


        def test_plugin_descriptor_has_module_names():
            descriptor = describe_plugin()
            assert descriptor["module_names"]
        """
    )


def _render_train_entrypoint() -> str:
    return dedent(
        """
        from __future__ import annotations

        from pathlib import Path

        from package import plugin as plugin_module

        from auto_llm_innovator.runtime import run_phase_with_plugin


        def run_phase(phase: str, run_dir: str, config_path: str, attempt_id: str) -> dict:
            idea_dir = Path(__file__).resolve().parent
            return run_phase_with_plugin(
                phase=phase,
                idea_dir=str(idea_dir),
                run_dir=run_dir,
                config_path=config_path,
                attempt_id=attempt_id,
                plugin_module=plugin_module,
            )
        """
    )


def _render_eval_entrypoint() -> str:
    return dedent(
        """
        from __future__ import annotations

        import json
        from pathlib import Path

        from package.evaluation.hooks import register_evaluation_hooks


        def run_evaluation(run_ref: str) -> dict:
            run_path = Path(run_ref)
            hooks = register_evaluation_hooks()
            phase_files = sorted(run_path.glob("*-summary.json"))
            losses = {}
            for path in phase_files:
                payload = json.loads(path.read_text(encoding="utf-8"))
                losses[payload["phase"]] = payload["metrics"]["loss"]
            trend = "improving" if list(losses.values()) == sorted(losses.values(), reverse=True) else "mixed"
            report = {
                "run_ref": str(run_path),
                "phase_losses": losses,
                "trend": trend,
                "evaluation_hooks": sorted(hooks),
            }
            report_path = run_path / "evaluation-report.json"
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            return {"report_path": str(report_path), "summary": report}
        """
    )
