from __future__ import annotations

from dataclasses import asdict, dataclass

from auto_llm_innovator.idea_spec.models import IdeaSpec


DISALLOWED_ARCHITECTURE_TERMS = {
    "gpt-2",
    "gpt2",
    "gpt-j",
    "llama",
    "mistral",
    "standard decoder-only",
    "vanilla transformer",
    "plain transformer",
}


@dataclass(slots=True)
class OriginalityReview:
    passed: bool
    score: float
    findings: list[str]
    required_revisions: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def review_originality(spec: IdeaSpec) -> OriginalityReview:
    findings: list[str] = []
    revisions: list[str] = []
    lowered_text = " ".join(
        [spec.raw_brief, spec.normalized_brief, *spec.novelty_claims, *spec.borrowed_mechanisms]
    ).lower()

    for term in sorted(DISALLOWED_ARCHITECTURE_TERMS):
        if term in lowered_text:
            revisions.append(f"References '{term}' too directly; require a more original recombination.")

    if len(spec.novelty_claims) < 2:
        revisions.append("Novelty claims are too thin; add at least two concrete originality claims.")

    if not spec.forbidden_fallback_patterns:
        revisions.append("Fallback patterns must be declared explicitly.")

    if spec.estimated_parameter_budget > 2_100_000_000:
        revisions.append("Parameter budget exceeds the 2.1B cap.")

    if "generic" in lowered_text and "avoid" not in lowered_text:
        revisions.append("Spec mentions generic patterns without clearly rejecting them.")

    findings.append("Originality review checks for direct SOTA copies, missing novelty claims, and cap violations.")
    score = max(0.0, 1.0 - 0.2 * len(revisions))
    return OriginalityReview(
        passed=not revisions,
        score=score,
        findings=findings,
        required_revisions=revisions,
    )
