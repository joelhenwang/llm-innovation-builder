from __future__ import annotations

import re
from textwrap import shorten

from auto_llm_innovator.constants import DEFAULT_SMALLER_SCALE_CAP, PARAMETER_CAP
from auto_llm_innovator.idea_spec.models import IdeaSpec


GENERIC_FALLBACK_PATTERNS = [
    "vanilla transformer",
    "plain transformer",
    "copy gpt",
    "standard decoder-only stack",
    "default llama clone",
]


def _extract_keywords(raw_brief: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]+", raw_brief.lower())
    stop_words = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "into",
        "from",
        "must",
        "model",
        "idea",
        "agent",
        "architecture",
        "llm",
    }
    unique: list[str] = []
    for token in tokens:
        if token in stop_words or len(token) < 5:
            continue
        if token not in unique:
            unique.append(token)
    return unique[:8]


def normalize_idea_spec(idea_id: str, raw_brief: str) -> IdeaSpec:
    keywords = _extract_keywords(raw_brief)
    headline = shorten(" ".join(keywords) or raw_brief, width=96, placeholder="...")
    novelty_claims = [
        f"Combine non-default mechanisms around: {', '.join(keywords[:4]) or 'novel routing and memory'}",
        "Reject template decoder-only stacks with only cosmetic changes.",
        "Use explicit originality rationale before implementation starts.",
    ]
    training_curriculum = [
        f"Smoke-test reduced variant near {DEFAULT_SMALLER_SCALE_CAP:,} parameters for math and logic.",
        "Run a learnability check on the target-scale model with a small training subset.",
        "Escalate to a production-like corpus only after the agent records why the idea looks promising.",
    ]
    inspirations = [
        "SOTA models may be studied for inspiration only.",
        "Any borrowed mechanism must be declared and recombined into an original design.",
    ]
    return IdeaSpec(
        idea_id=idea_id,
        raw_brief=raw_brief.strip(),
        normalized_brief=headline,
        hypothesis=f"This idea can improve a general-purpose LM by exploring: {headline}.",
        novelty_claims=novelty_claims,
        forbidden_fallback_patterns=GENERIC_FALLBACK_PATTERNS.copy(),
        intended_learning_objective="Train a novel sub-2.1B autoregressive LM that shows learnability and promising general capabilities.",
        estimated_parameter_budget=PARAMETER_CAP,
        training_curriculum_outline=training_curriculum,
        inspirations_consulted=inspirations,
        public_references=[],
    )
