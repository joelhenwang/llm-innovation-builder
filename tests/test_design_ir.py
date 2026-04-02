import pytest

from auto_llm_innovator.design_ir import (
    DesignIR,
    DesignIRValidationError,
    compile_design_ir,
    project_idea_spec,
    validate_design_ir,
)
from auto_llm_innovator.handoff import load_research_idea_bundle


def _candidate_bundle() -> dict:
    return {
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


def _mix_bundle() -> dict:
    return {
        "mix_id": "mix-123",
        "source_candidate_ids": ["cand-a", "cand-b"],
        "source_titles": ["Mechanism A", "Mechanism B"],
        "mix_rationale": "The fusion combines efficient routing with longer-context memory.",
        "fusion_methodology": "Train a fused model with retrieval routing while remaining compatible with a GPT-2 tokenizer.",
        "experiment_guide": [
            "Evaluate perplexity against the internal baseline.",
            "Ablate routing frequency.",
        ],
        "open_questions": ["Will the fused model fit the intended budget?"],
        "sourced_facts": ["The recipe is constrained to GPT-2 tokenization."],
    }


def test_free_text_bundle_compiles_into_valid_design_ir():
    bundle = load_research_idea_bundle(raw_brief="Create a memory-routed recurrent attention language model.")

    design_ir = compile_design_ir(bundle, idea_id="idea-0001")
    validate_design_ir(design_ir)

    assert design_ir.idea_id == "idea-0001"
    assert {stage.stage for stage in design_ir.training_plan} == {"smoke", "small", "full"}
    assert any(module.name == "core_backbone" for module in design_ir.modules)
    assert any(task.metrics for task in design_ir.evaluation_plan)


def test_candidate_bundle_preserves_evaluation_and_ablations():
    bundle = load_research_idea_bundle(payload=_candidate_bundle())

    design_ir = compile_design_ir(bundle, idea_id="idea-0002")

    assert any("perplexity" in task.metrics for task in design_ir.evaluation_plan)
    assert any("recurrent" in ablation.description.lower() for ablation in design_ir.ablation_plan)
    assert any(criterion.focus_area == "training_stability" for criterion in design_ir.failure_criteria)


def test_mix_bundle_compiles_to_fusion_pattern():
    bundle = load_research_idea_bundle(payload=_mix_bundle())

    design_ir = compile_design_ir(bundle, idea_id="idea-0003")

    assert design_ir.bundle_kind == "research_mix"
    assert design_ir.source_candidate_ids == ["cand-a", "cand-b"]
    assert "fused" in design_ir.architecture.pattern_label


def test_memory_wording_adds_state_and_memory_modules():
    bundle = load_research_idea_bundle(
        raw_brief="Invent a recurrent memory and retrieval decoder with cache-aware routing."
    )

    design_ir = compile_design_ir(bundle, idea_id="idea-0004")

    module_names = {module.name for module in design_ir.modules}
    assert "state_adapter" in module_names
    assert "memory_adapter" in module_names
    assert "routing_cache" in module_names
    assert design_ir.architecture.state_semantics.has_recurrent_state is True


def test_validator_rejects_undefined_module_dependency():
    bundle = load_research_idea_bundle(raw_brief="Invent a recurrent memory model.")
    design_ir = compile_design_ir(bundle, idea_id="idea-0005")
    payload = design_ir.to_dict()
    payload["modules"][0]["depends_on"] = ["missing_module"]

    with pytest.raises(DesignIRValidationError, match="undefined module"):
        validate_design_ir(DesignIR.from_dict(payload))


def test_validator_rejects_missing_phase():
    bundle = load_research_idea_bundle(raw_brief="Invent a recurrent memory model.")
    design_ir = compile_design_ir(bundle, idea_id="idea-0006")
    payload = design_ir.to_dict()
    payload["training_plan"] = [stage for stage in payload["training_plan"] if stage["stage"] != "full"]

    with pytest.raises(DesignIRValidationError, match="missing stages"):
        validate_design_ir(DesignIR.from_dict(payload))


def test_validator_rejects_tokenizer_mismatch_and_cap_overflow():
    bundle = load_research_idea_bundle(raw_brief="Invent a recurrent memory model.")
    design_ir = compile_design_ir(bundle, idea_id="idea-0007")

    tokenizer_payload = design_ir.to_dict()
    tokenizer_payload["tokenizer_requirement"] = "bpe-other"
    with pytest.raises(DesignIRValidationError, match="tokenizer"):
        validate_design_ir(DesignIR.from_dict(tokenizer_payload))

    cap_payload = design_ir.to_dict()
    cap_payload["parameter_cap"] = 2_100_000_001
    with pytest.raises(DesignIRValidationError, match="parameter cap"):
        validate_design_ir(DesignIR.from_dict(cap_payload))


def test_design_ir_projection_preserves_novelty_and_evaluation_intent():
    bundle = load_research_idea_bundle(payload=_mix_bundle())
    design_ir = compile_design_ir(bundle, idea_id="idea-0008")

    spec = project_idea_spec(design_ir, bundle)

    assert spec.idea_id == "idea-0008"
    assert spec.tokenizer == "gpt2"
    assert any("fusion combines efficient routing" in claim.lower() for claim in spec.novelty_claims)
    assert "Evaluate perplexity against the internal baseline." in spec.evaluation_intent
