from __future__ import annotations

from auto_llm_innovator.design_ir import compile_design_ir, project_idea_spec
from auto_llm_innovator.handoff.models import ResearchIdeaBundle
from auto_llm_innovator.idea_spec.models import IdeaSpec


def compile_idea_spec(bundle: ResearchIdeaBundle, idea_id: str) -> IdeaSpec:
    design_ir = compile_design_ir(bundle, idea_id=idea_id)
    return project_idea_spec(design_ir, bundle)
