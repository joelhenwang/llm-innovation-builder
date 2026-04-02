from __future__ import annotations

from auto_llm_innovator.handoff.compiler import compile_idea_spec
from auto_llm_innovator.handoff.loaders import bundle_from_free_text
from auto_llm_innovator.idea_spec.models import IdeaSpec


def normalize_idea_spec(idea_id: str, raw_brief: str) -> IdeaSpec:
    bundle = bundle_from_free_text(raw_brief)
    return compile_idea_spec(bundle, idea_id=idea_id)
