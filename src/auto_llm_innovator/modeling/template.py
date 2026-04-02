from __future__ import annotations

from textwrap import dedent

from auto_llm_innovator.design_ir.models import DesignIR
from auto_llm_innovator.idea_spec.models import IdeaSpec


def render_model_template(spec: IdeaSpec, design_ir: DesignIR) -> str:
    architecture_name = design_ir.architecture.pattern_label
    borrowed = spec.borrowed_mechanisms if spec.borrowed_mechanisms else []
    module_names = [module.name for module in design_ir.modules]
    supports = {
        "recurrent_state": design_ir.architecture.state_semantics.has_recurrent_state,
        "external_memory": design_ir.architecture.state_semantics.has_external_memory,
        "cache_path": design_ir.architecture.state_semantics.has_cache_path,
    }
    return dedent(
        f"""
        from __future__ import annotations

        from dataclasses import dataclass

        try:
            import torch
            import torch.nn as nn
        except Exception:  # pragma: no cover - optional dependency path
            torch = None
            nn = None


        @dataclass(slots=True)
        class ModelConfig:
            idea_id: str = "{spec.idea_id}"
            architecture_name: str = "{architecture_name}"
            target_parameters: int = {spec.estimated_parameter_budget}
            hidden_size: int = 2048
            num_layers: int = 24
            num_heads: int = 16
            tokenizer: str = "{spec.tokenizer}"
            borrowed_mechanisms: list[str] = None

            def __post_init__(self) -> None:
                if self.borrowed_mechanisms is None:
                    self.borrowed_mechanisms = {borrowed!r}


        class CreativeIdeaModel(nn.Module if nn is not None else object):
            def __init__(self, config: ModelConfig) -> None:
                if nn is not None:
                    super().__init__()
                self.config = config
                self.architecture_rationale = {spec.novelty_claims!r}
                if nn is not None:
                    self.embedding = nn.Embedding(1024, min(config.hidden_size, 256))
                    self.gate = nn.Linear(min(config.hidden_size, 256), min(config.hidden_size, 256))
                    self.head = nn.Linear(min(config.hidden_size, 256), 1024)

            def forward(self, input_ids, state_tensor=None, memory_tensor=None, cache_tensor=None):
                if nn is None or torch is None:
                    batch = len(input_ids) if hasattr(input_ids, "__len__") else 1
                    seq = len(input_ids[0]) if batch and hasattr(input_ids[0], "__len__") else 1
                    logits = [[[0.0 for _ in range(1024)] for _ in range(seq)] for _ in range(batch)]
                    output = {{"logits": logits}}
                    if state_tensor is not None:
                        output["state_tensor"] = state_tensor
                    if memory_tensor is not None:
                        output["memory_tensor"] = memory_tensor
                    if cache_tensor is not None:
                        output["cache_tensor"] = cache_tensor
                    return output
                hidden = self.embedding(torch.as_tensor(input_ids, dtype=torch.long))
                gated = torch.tanh(self.gate(hidden))
                logits = self.head(gated)
                output = {{"logits": logits.detach().cpu().tolist()}}
                if state_tensor is not None:
                    output["state_tensor"] = state_tensor
                if memory_tensor is not None:
                    output["memory_tensor"] = memory_tensor
                if cache_tensor is not None:
                    output["cache_tensor"] = cache_tensor
                return output


        def build_model(config: ModelConfig | None = None) -> CreativeIdeaModel:
            return CreativeIdeaModel(config or ModelConfig())


        def describe_plugin() -> dict:
            return {{
                "architecture_name": "{architecture_name}",
                "module_names": {module_names!r},
                "supports": {supports!r},
            }}


        def register_evaluation_hooks() -> dict:
            def evaluation_task_1(**kwargs) -> dict:
                return {{"hook": "default-runtime-eval", "loss_snapshot": round(kwargs["loss_value"], 4)}}

            return {{"evaluation_task_1": evaluation_task_1}}
        """
    ).strip() + "\n"


def render_train_template(spec: IdeaSpec, design_ir: DesignIR) -> str:
    return dedent(
        f"""
        from __future__ import annotations

        from pathlib import Path

        import model as plugin_module

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
    ).strip() + "\n"


def render_eval_template() -> str:
    return dedent(
        """
        from __future__ import annotations

        import json
        from pathlib import Path


        def run_evaluation(run_ref: str) -> dict:
            run_path = Path(run_ref)
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
            }
            report_path = run_path / "evaluation-report.json"
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            return {"report_path": str(report_path), "summary": report}
        """
    ).strip() + "\n"
