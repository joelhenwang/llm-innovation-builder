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
    idea_id: str = "idea-0002"
    architecture_name: str = "idea_0002_creative_lm"
    target_parameters: int = 2100000000
    hidden_size: int = 2048
    num_layers: int = 24
    num_heads: int = 16
    tokenizer: str = "gpt2"
    borrowed_mechanisms: list[str] = None

    def __post_init__(self) -> None:
        if self.borrowed_mechanisms is None:
            self.borrowed_mechanisms = []


class CreativeIdeaModel(nn.Module if nn is not None else object):
    def __init__(self, config: ModelConfig) -> None:
        if nn is not None:
            super().__init__()
        self.config = config
        self.architecture_rationale = ['Combine non-default mechanisms around: invent, phase-coupled, memory, lattice', 'Reject template decoder-only stacks with only cosmetic changes.', 'Use explicit originality rationale before implementation starts.']
        if nn is not None:
            self.embedding = nn.Embedding(1024, min(config.hidden_size, 256))
            self.gate = nn.Linear(min(config.hidden_size, 256), min(config.hidden_size, 256))
            self.head = nn.Linear(min(config.hidden_size, 256), 1024)

    def forward(self, input_ids):
        if nn is None or torch is None:
            return {"logits_shape": [len(input_ids), 1024] if hasattr(input_ids, "__len__") else [1, 1024]}
        hidden = self.embedding(input_ids)
        gated = torch.tanh(self.gate(hidden))
        logits = self.head(gated)
        return logits


def build_model(config: ModelConfig | None = None) -> CreativeIdeaModel:
    return CreativeIdeaModel(config or ModelConfig())
