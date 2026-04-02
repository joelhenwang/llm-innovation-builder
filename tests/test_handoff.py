import json
from pathlib import Path

import pytest

from auto_llm_innovator.handoff import HandoffValidationError, compile_idea_spec, load_research_idea_bundle


def test_free_text_brief_builds_valid_bundle():
    bundle = load_research_idea_bundle(raw_brief="Create a memory-routed recurrent attention language model.")

    assert bundle.bundle_kind == "free_text"
    assert bundle.title
    assert bundle.mechanism_summary
    assert bundle.novelty_rationale
    assert bundle.tokenizer_requirement == "gpt2"


def test_single_candidate_payload_maps_to_bundle():
    payload = {
        "candidate_id": "cand-123",
        "novelty_rationale": "This introduces a non-default state-space memory path for compact LMs.",
        "methodology": "Implement a hybrid state-space decoder with explicit recurrent memory.",
        "experiment_guide": [
            "Reproduce the architecture at small scale.",
            "Measure perplexity and compare against a compact baseline.",
            "Ablate the recurrent memory path.",
        ],
        "open_questions": ["Will the recurrent memory destabilize training?"],
        "research_item": {
            "title": "Hybrid Memory Decoder",
            "risks": ["Single-source evidence."],
            "compatibility_notes": "Compatible with GPT-2 tokenizer.",
            "tokenizer_compatible": True,
        },
    }

    bundle = load_research_idea_bundle(payload=payload)

    assert bundle.bundle_kind == "research_candidate"
    assert bundle.source_candidate_ids == ["cand-123"]
    assert bundle.title == "Hybrid Memory Decoder"
    assert bundle.evaluation_targets == ["Measure perplexity and compare against a compact baseline."]
    assert bundle.ablation_ideas == ["Ablate the recurrent memory path."]


def test_mix_payload_maps_to_bundle():
    payload = {
        "mix_id": "mix-123",
        "source_candidate_ids": ["cand-a", "cand-b"],
        "source_titles": ["Mechanism A", "Mechanism B"],
        "mix_rationale": "The fusion combines efficient routing with longer-context memory.",
        "fusion_methodology": "Train a fused model while remaining compatible with a GPT-2 tokenizer.",
        "experiment_guide": [
            "Evaluate perplexity against the internal baseline.",
            "Ablate routing frequency.",
        ],
        "open_questions": ["Will the fused model fit the intended budget?"],
        "sourced_facts": ["The recipe is constrained to GPT-2 tokenization."],
    }

    bundle = load_research_idea_bundle(payload=payload)

    assert bundle.bundle_kind == "research_mix"
    assert bundle.source_candidate_ids == ["cand-a", "cand-b"]
    assert bundle.source_titles == ["Mechanism A", "Mechanism B"]
    assert bundle.tokenizer_requirement == "gpt2"


def test_missing_core_fields_raise_clear_validation_error():
    payload = {
        "candidate_id": "cand-123",
        "novelty_rationale": "",
        "methodology": "Implement a hybrid state-space decoder.",
        "experiment_guide": ["Measure perplexity."],
        "research_item": {
            "title": "Hybrid Memory Decoder",
            "compatibility_notes": "Compatible with GPT-2 tokenizer.",
            "tokenizer_compatible": True,
        },
    }

    with pytest.raises(HandoffValidationError, match="novelty rationale"):
        load_research_idea_bundle(payload=payload)


def test_missing_gpt2_compatibility_raises_validation_error():
    payload = {
        "candidate_id": "cand-123",
        "novelty_rationale": "Novel recurrent memory path.",
        "methodology": "Implement a hybrid state-space decoder.",
        "experiment_guide": ["Measure perplexity."],
        "research_item": {
            "title": "Hybrid Memory Decoder",
            "compatibility_notes": "",
            "tokenizer_compatible": True,
        },
    }

    with pytest.raises(HandoffValidationError, match="tokenizer compatibility"):
        load_research_idea_bundle(payload=payload)


def test_compiler_preserves_novelty_and_evaluation_intent():
    payload = {
        "mix_id": "mix-123",
        "source_candidate_ids": ["cand-a", "cand-b"],
        "source_titles": ["Mechanism A", "Mechanism B"],
        "mix_rationale": "The fusion combines efficient routing with longer-context memory.",
        "fusion_methodology": "Train a fused model while remaining compatible with a GPT-2 tokenizer.",
        "experiment_guide": [
            "Evaluate perplexity against the internal baseline.",
            "Ablate routing frequency.",
        ],
        "open_questions": ["Will the fused model fit the intended budget?"],
        "sourced_facts": ["The recipe is constrained to GPT-2 tokenization."],
    }

    bundle = load_research_idea_bundle(payload=payload)
    spec = compile_idea_spec(bundle, idea_id="idea-0009")

    assert spec.idea_id == "idea-0009"
    assert spec.tokenizer == "gpt2"
    assert any("fusion combines efficient routing" in claim.lower() for claim in spec.novelty_claims)
    assert "Evaluate perplexity against the internal baseline." in spec.evaluation_intent


def test_bundle_file_loader_accepts_json_file(tmp_path: Path):
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(
        json.dumps(
            {
                "candidate_id": "cand-123",
                "novelty_rationale": "Novel recurrent memory path.",
                "methodology": "Implement a hybrid state-space decoder.",
                "experiment_guide": ["Measure perplexity."],
                "research_item": {
                    "title": "Hybrid Memory Decoder",
                    "compatibility_notes": "Compatible with GPT-2 tokenizer.",
                    "tokenizer_compatible": True,
                },
            }
        ),
        encoding="utf-8",
    )

    bundle = load_research_idea_bundle(bundle_file=bundle_path)
    assert bundle.source_candidate_ids == ["cand-123"]
