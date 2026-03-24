from auto_llm_innovator.idea_spec import normalize_idea_spec, review_originality


def test_normalize_idea_spec_builds_required_fields():
    spec = normalize_idea_spec("idea-0001", "Create a memory-routed recurrent attention language model.")
    assert spec.idea_id == "idea-0001"
    assert spec.tokenizer == "gpt2"
    assert spec.estimated_parameter_budget == 2_100_000_000
    assert spec.novelty_claims
    assert spec.training_curriculum_outline


def test_originality_review_rejects_near_copy_language():
    spec = normalize_idea_spec("idea-0002", "Build a plain transformer inspired by LLaMA and GPT-2.")
    review = review_originality(spec)
    assert not review.passed
    assert any("llama" in item.lower() or "gpt-2" in item.lower() for item in review.required_revisions)
