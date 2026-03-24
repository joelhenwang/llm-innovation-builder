from __future__ import annotations

from textwrap import dedent

from auto_llm_innovator.idea_spec.models import IdeaSpec


def render_model_template(spec: IdeaSpec) -> str:
    architecture_name = f"{spec.idea_id.replace('-', '_')}_creative_lm"
    borrowed = ", ".join(spec.borrowed_mechanisms) if spec.borrowed_mechanisms else "[]"
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
                    self.borrowed_mechanisms = {borrowed}


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

            def forward(self, input_ids):
                if nn is None or torch is None:
                    return {{"logits_shape": [len(input_ids), 1024] if hasattr(input_ids, "__len__") else [1, 1024]}}
                hidden = self.embedding(input_ids)
                gated = torch.tanh(self.gate(hidden))
                logits = self.head(gated)
                return logits


        def build_model(config: ModelConfig | None = None) -> CreativeIdeaModel:
            return CreativeIdeaModel(config or ModelConfig())
        """
    ).strip() + "\n"


def render_train_template(spec: IdeaSpec) -> str:
    return dedent(
        f"""
        from __future__ import annotations

        import json
        from pathlib import Path

        from model import ModelConfig, build_model


        PHASE_DEFAULTS = {{
            "smoke": {{"loss": 5.9, "steps": 8}},
            "small": {{"loss": 4.3, "steps": 24}},
            "full": {{"loss": 3.8, "steps": 96}},
        }}


        def run_phase(phase: str, run_dir: str, config_path: str, attempt_id: str) -> dict:
            config = json.loads(Path(config_path).read_text(encoding="utf-8"))
            model = build_model(ModelConfig())
            metrics = PHASE_DEFAULTS[phase].copy()
            metrics["target_parameters"] = config["target_parameters"]
            metrics["tokenizer"] = "{spec.tokenizer}"
            Path(run_dir).mkdir(parents=True, exist_ok=True)
            artifact = Path(run_dir) / f"{{phase}}-summary.json"
            artifact.write_text(json.dumps({{
                "phase": phase,
                "attempt_id": attempt_id,
                "metrics": metrics,
                "architecture_name": model.config.architecture_name,
                "novelty_claims": {spec.novelty_claims!r},
            }}, indent=2), encoding="utf-8")
            return {{
                "status": "passed",
                "key_metrics": {{"loss": metrics["loss"], "steps": metrics["steps"]}},
                "failure_signals": [],
                "artifacts_produced": [str(artifact)],
                "reviewer_notes": ["Synthetic trainer scaffold executed successfully."],
                "next_action_recommendation": "advance" if phase != "full" else "complete",
                "consumed_budget": {{
                    "requested_parameters": config["target_parameters"],
                    "steps": metrics["steps"],
                    "device": "rocm" if config.get("prefer_rocm") else "cpu-dry-run",
                }},
            }}
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
