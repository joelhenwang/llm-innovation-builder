from .models import IdeaSpec
from .originality import OriginalityReview, review_originality


def normalize_idea_spec(idea_id: str, raw_brief: str) -> IdeaSpec:
    from .normalize import normalize_idea_spec as _normalize_idea_spec

    return _normalize_idea_spec(idea_id, raw_brief)


__all__ = ["IdeaSpec", "normalize_idea_spec", "OriginalityReview", "review_originality"]
