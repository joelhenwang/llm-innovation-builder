from .loaders import HandoffValidationError, bundle_from_free_text, bundle_from_payload, load_research_idea_bundle
from .models import BundleKinds, ResearchIdeaBundle


def compile_idea_spec(bundle: ResearchIdeaBundle, idea_id: str):
    from .compiler import compile_idea_spec as _compile_idea_spec

    return _compile_idea_spec(bundle, idea_id)

__all__ = [
    "BundleKinds",
    "HandoffValidationError",
    "ResearchIdeaBundle",
    "bundle_from_free_text",
    "bundle_from_payload",
    "compile_idea_spec",
    "load_research_idea_bundle",
]
