from __future__ import annotations

from pathlib import Path
from typing import Any

from auto_llm_innovator.constants import GPT2_TOKENIZER, PARAMETER_CAP
from auto_llm_innovator.filesystem import read_json
from auto_llm_innovator.handoff.models import BundleKinds, ResearchIdeaBundle


class HandoffValidationError(ValueError):
    pass


def load_research_idea_bundle(
    *,
    raw_brief: str | None = None,
    bundle_file: str | Path | None = None,
    payload: dict[str, Any] | None = None,
) -> ResearchIdeaBundle:
    provided = [raw_brief is not None, bundle_file is not None, payload is not None]
    if sum(provided) != 1:
        raise ValueError("Provide exactly one of raw_brief, bundle_file, or payload.")

    if raw_brief is not None:
        bundle = bundle_from_free_text(raw_brief)
    elif bundle_file is not None:
        bundle = bundle_from_payload(read_json(Path(bundle_file)))
    else:
        bundle = bundle_from_payload(payload or {})

    validate_research_idea_bundle(bundle)
    return bundle


def bundle_from_free_text(raw_brief: str) -> ResearchIdeaBundle:
    brief = raw_brief.strip()
    if not brief:
        raise HandoffValidationError("Free-text brief must not be empty.")
    return ResearchIdeaBundle(
        bundle_kind=BundleKinds.FREE_TEXT,
        source_artifact_kind=BundleKinds.FREE_TEXT,
        source_candidate_ids=[],
        source_titles=[],
        title=brief,
        mechanism_summary=brief,
        novelty_rationale="Derived from direct free-text submission.",
        implementation_requirements=[
            "Reproduce the core mechanism at smoke-test scale before escalating training budget."
        ],
        known_constraints=[
            f"Target a sub-{PARAMETER_CAP:,} parameter language model.",
            "Maintain GPT-2 tokenizer compatibility.",
        ],
        dataset_requirements=[],
        evaluation_targets=[
            "Compare against internal baseline, prior runs, and public references."
        ],
        ablation_ideas=[],
        expected_failure_modes=[],
        compute_budget_hint=f"Up to {PARAMETER_CAP:,} parameters.",
        tokenizer_requirement=GPT2_TOKENIZER,
        raw_payload={"raw_brief": brief},
    )


def bundle_from_payload(payload: dict[str, Any]) -> ResearchIdeaBundle:
    if "candidate_id" in payload and "research_item" in payload:
        return _bundle_from_candidate_payload(payload)
    if "mix_id" in payload and "source_candidate_ids" in payload:
        return _bundle_from_mix_payload(payload)
    raise HandoffValidationError(
        "Unsupported bundle format. Expected an ExperimentCandidate-style payload or a MixedExperimentCandidate-style payload."
    )


def validate_research_idea_bundle(bundle: ResearchIdeaBundle) -> None:
    if not bundle.title.strip():
        raise HandoffValidationError("Bundle is missing a title.")
    if not bundle.mechanism_summary.strip():
        raise HandoffValidationError("Bundle is missing a mechanism summary.")
    if not bundle.novelty_rationale.strip():
        raise HandoffValidationError("Bundle is missing a novelty rationale.")

    if bundle.bundle_kind != BundleKinds.FREE_TEXT and not bundle.source_candidate_ids:
        raise HandoffValidationError("Structured bundles must include at least one source candidate id.")

    if not bundle.evaluation_targets and not bundle.implementation_requirements:
        raise HandoffValidationError(
            "Bundle must include experiment intent through evaluation targets or experiment guide steps."
        )

    if bundle.bundle_kind != BundleKinds.FREE_TEXT:
        tokenizer = bundle.tokenizer_requirement.strip().lower()
        if not tokenizer:
            raise HandoffValidationError("Structured bundles must explicitly declare tokenizer compatibility.")
        if tokenizer != GPT2_TOKENIZER:
            raise HandoffValidationError(
                f"Structured bundles must declare tokenizer compatibility as '{GPT2_TOKENIZER}'."
            )


def _bundle_from_candidate_payload(payload: dict[str, Any]) -> ResearchIdeaBundle:
    item = payload.get("research_item") or {}
    source_candidate_id = str(payload.get("candidate_id", "")).strip()
    return ResearchIdeaBundle(
        bundle_kind=BundleKinds.RESEARCH_CANDIDATE,
        source_artifact_kind=BundleKinds.RESEARCH_CANDIDATE,
        source_candidate_ids=[source_candidate_id] if source_candidate_id else [],
        source_titles=[str(item.get("title", "")).strip()] if item.get("title") else [],
        title=str(item.get("title", "")).strip(),
        mechanism_summary=str(payload.get("methodology", "")).strip(),
        novelty_rationale=str(payload.get("novelty_rationale", "")).strip(),
        implementation_requirements=_clean_list(payload.get("experiment_guide")),
        known_constraints=_clean_list(item.get("risks")) + _clean_optional_text(item.get("compatibility_notes")),
        dataset_requirements=[],
        evaluation_targets=_extract_evaluation_targets(payload.get("experiment_guide")),
        ablation_ideas=_extract_ablation_ideas(payload.get("experiment_guide")),
        expected_failure_modes=_clean_list(payload.get("open_questions")),
        compute_budget_hint="Adapt to a <=2.1B PyTorch experiment.",
        tokenizer_requirement=_candidate_tokenizer_requirement(item),
        raw_payload=payload,
    )


def _bundle_from_mix_payload(payload: dict[str, Any]) -> ResearchIdeaBundle:
    source_titles = _clean_list(payload.get("source_titles"))
    title = "Mix: " + " + ".join(source_titles) if source_titles else str(payload.get("mix_id", "")).strip()
    return ResearchIdeaBundle(
        bundle_kind=BundleKinds.RESEARCH_MIX,
        source_artifact_kind=BundleKinds.RESEARCH_MIX,
        source_candidate_ids=_clean_list(payload.get("source_candidate_ids")),
        source_titles=source_titles,
        title=title,
        mechanism_summary=str(payload.get("fusion_methodology", "")).strip(),
        novelty_rationale=str(payload.get("mix_rationale", "")).strip(),
        implementation_requirements=_clean_list(payload.get("experiment_guide")),
        known_constraints=_clean_list(payload.get("sourced_facts")),
        dataset_requirements=[],
        evaluation_targets=_extract_evaluation_targets(payload.get("experiment_guide")),
        ablation_ideas=_extract_ablation_ideas(payload.get("experiment_guide")),
        expected_failure_modes=_clean_list(payload.get("open_questions")),
        compute_budget_hint="Adapt to a <=2.1B PyTorch experiment.",
        tokenizer_requirement=_mix_tokenizer_requirement(payload),
        raw_payload=payload,
    )


def _clean_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    cleaned: list[str] = []
    for value in values:
        text = str(value).strip()
        if text:
            cleaned.append(text)
    return cleaned


def _clean_optional_text(value: Any) -> list[str]:
    text = str(value or "").strip()
    return [text] if text else []


def _extract_evaluation_targets(steps: Any) -> list[str]:
    targets: list[str] = []
    for step in _clean_list(steps):
        lowered = step.lower()
        if any(keyword in lowered for keyword in ("measure", "evaluate", "benchmark", "compare", "perplexity")):
            targets.append(step)
    return targets


def _extract_ablation_ideas(steps: Any) -> list[str]:
    ideas: list[str] = []
    for step in _clean_list(steps):
        if "ablate" in step.lower():
            ideas.append(step)
    return ideas


def _candidate_tokenizer_requirement(item: dict[str, Any]) -> str:
    if item.get("tokenizer_compatible") is True and item.get("compatibility_notes"):
        return GPT2_TOKENIZER if "gpt-2" in str(item["compatibility_notes"]).lower() else ""
    return ""


def _mix_tokenizer_requirement(payload: dict[str, Any]) -> str:
    for fact in _clean_list(payload.get("sourced_facts")):
        if "gpt-2" in fact.lower():
            return GPT2_TOKENIZER
    for text in _clean_optional_text(payload.get("fusion_methodology")):
        if "gpt-2" in text.lower():
            return GPT2_TOKENIZER
    return ""
