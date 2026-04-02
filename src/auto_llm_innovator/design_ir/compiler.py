from __future__ import annotations

import re
from textwrap import shorten

from auto_llm_innovator.constants import DEFAULT_SMALLER_SCALE_CAP, GPT2_TOKENIZER, PARAMETER_CAP, PHASES
from auto_llm_innovator.datasets.registry import dataset_plan_for_phase
from auto_llm_innovator.design_ir.models import (
    AblationPlan,
    ArchitecturePlan,
    CompatibilityProjection,
    DesignIR,
    DesignModule,
    EvaluationTask,
    FailureCriterion,
    ImplementationMilestone,
    StateSemantics,
    TensorInterface,
    TrainingStagePlan,
)
from auto_llm_innovator.handoff.models import ResearchIdeaBundle
from auto_llm_innovator.idea_spec.models import IdeaSpec


def compile_design_ir(bundle: ResearchIdeaBundle, idea_id: str) -> DesignIR:
    state_semantics = _infer_state_semantics(bundle)
    pattern_label = _infer_pattern_label(bundle, state_semantics)
    modules = _build_modules(bundle, state_semantics)
    module_names = [module.name for module in modules]
    tensor_interfaces = _build_tensor_interfaces(modules, state_semantics)
    training_plan = _build_training_plan(bundle)
    evaluation_plan = _build_evaluation_plan(bundle)
    ablation_plan = _build_ablation_plan(bundle, module_names)
    failure_criteria = _build_failure_criteria(bundle, module_names, state_semantics)
    implementation_milestones = _build_implementation_milestones(bundle, module_names)
    compatibility_projection = _build_compatibility_projection(bundle, pattern_label, state_semantics)

    architecture = ArchitecturePlan(
        pattern_label=pattern_label,
        mechanism_summary=bundle.mechanism_summary,
        module_graph_summary=_build_module_graph_summary(module_names, state_semantics),
        state_semantics=state_semantics,
        assumption_source="heuristic_bundle_compiler",
    )

    return DesignIR(
        idea_id=idea_id,
        title=bundle.title,
        bundle_kind=bundle.bundle_kind,
        source_candidate_ids=bundle.source_candidate_ids.copy(),
        source_titles=bundle.source_titles.copy(),
        tokenizer_requirement=bundle.tokenizer_requirement or GPT2_TOKENIZER,
        parameter_cap=PARAMETER_CAP,
        compute_budget_hint=bundle.compute_budget_hint or f"Up to {PARAMETER_CAP:,} parameters.",
        known_constraints=bundle.known_constraints.copy(),
        architecture=architecture,
        modules=modules,
        tensor_interfaces=tensor_interfaces,
        training_plan=training_plan,
        evaluation_plan=evaluation_plan,
        ablation_plan=ablation_plan,
        failure_criteria=failure_criteria,
        implementation_milestones=implementation_milestones,
        compatibility_projection=compatibility_projection,
    )


def project_idea_spec(design_ir: DesignIR, bundle: ResearchIdeaBundle) -> IdeaSpec:
    projection = design_ir.compatibility_projection
    return IdeaSpec(
        idea_id=design_ir.idea_id,
        raw_brief=projection.raw_brief,
        normalized_brief=projection.normalized_brief,
        hypothesis=projection.hypothesis,
        novelty_claims=projection.novelty_claims.copy(),
        forbidden_fallback_patterns=[
            "vanilla transformer",
            "plain transformer",
            "copy gpt",
            "standard decoder-only stack",
            "default llama clone",
        ],
        intended_learning_objective="Train a novel sub-2.1B autoregressive LM that shows learnability and promising general capabilities.",
        estimated_parameter_budget=design_ir.parameter_cap,
        training_curriculum_outline=projection.training_curriculum_outline.copy(),
        inspirations_consulted=projection.inspirations_consulted.copy(),
        tokenizer=design_ir.tokenizer_requirement or GPT2_TOKENIZER,
        intended_model_target="general-purpose autoregressive language model",
        evaluation_intent=projection.evaluation_intent,
        borrowed_mechanisms=bundle.source_titles.copy(),
        public_references=bundle.source_candidate_ids.copy(),
    )


def _infer_state_semantics(bundle: ResearchIdeaBundle) -> StateSemantics:
    haystack = " ".join(
        [
            bundle.mechanism_summary,
            bundle.novelty_rationale,
            " ".join(bundle.implementation_requirements),
            " ".join(bundle.expected_failure_modes),
        ]
    ).lower()
    has_recurrent_state = any(keyword in haystack for keyword in ("recurrent", "state-space", "state space", "memory"))
    has_external_memory = any(keyword in haystack for keyword in ("memory", "retrieval"))
    has_cache_path = any(keyword in haystack for keyword in ("cache", "routing", "retrieval", "memory"))
    summary_parts = ["Autoregressive token stream with a decoder-facing hidden-state pathway."]
    if has_recurrent_state:
        summary_parts.append("Maintain recurrent or stateful activations across steps.")
    if has_external_memory:
        summary_parts.append("Include a dedicated memory or retrieval pathway.")
    if has_cache_path:
        summary_parts.append("Expose a cache or routing interface for planner-visible debugging.")
    return StateSemantics(
        has_recurrent_state=has_recurrent_state,
        has_external_memory=has_external_memory,
        has_cache_path=has_cache_path,
        summary=" ".join(summary_parts),
        assumption_source="heuristic_bundle_compiler",
    )


def _infer_pattern_label(bundle: ResearchIdeaBundle, state_semantics: StateSemantics) -> str:
    text = f"{bundle.title} {bundle.mechanism_summary}".lower()
    if "mix" in text or "fusion" in text:
        return "fused_experimental_decoder"
    if "retrieval" in text and state_semantics.has_recurrent_state:
        return "retrieval_recurrent_decoder"
    if "state-space" in text or "state space" in text:
        return "state_space_augmented_decoder"
    if state_semantics.has_recurrent_state and state_semantics.has_external_memory:
        return "memory_routed_recurrent_decoder"
    if state_semantics.has_recurrent_state:
        return "recurrent_augmented_decoder"
    return "novel_decoder_variant"


def _build_modules(bundle: ResearchIdeaBundle, state_semantics: StateSemantics) -> list[DesignModule]:
    modules = [
        DesignModule(
            name="token_embedding",
            kind="frontend",
            purpose="Convert GPT-2 token ids into model-ready hidden vectors.",
            inputs=["input_tokens"],
            outputs=["token_embeddings"],
            assumption_source="heuristic_bundle_compiler",
        ),
        DesignModule(
            name="core_backbone",
            kind="core_block",
            purpose=_build_core_purpose(bundle),
            inputs=["token_embeddings", "hidden_state"],
            outputs=["hidden_state"],
            depends_on=["token_embedding"],
            assumption_source="heuristic_bundle_compiler",
        ),
    ]
    if state_semantics.has_recurrent_state:
        modules.append(
            DesignModule(
                name="state_adapter",
                kind="state_path",
                purpose="Manage recurrent state updates and feed them back into the backbone.",
                inputs=["hidden_state", "state_tensor"],
                outputs=["state_tensor", "state_conditioned_hidden"],
                depends_on=["core_backbone"],
                assumption_source="heuristic_bundle_compiler",
            )
        )
    if state_semantics.has_external_memory:
        modules.append(
            DesignModule(
                name="memory_adapter",
                kind="memory_path",
                purpose="Retrieve or update memory features that condition token processing.",
                inputs=["hidden_state", "memory_tensor"],
                outputs=["memory_tensor", "memory_context"],
                depends_on=["core_backbone"],
                assumption_source="heuristic_bundle_compiler",
            )
        )
    if state_semantics.has_cache_path:
        dependencies = ["core_backbone"]
        if state_semantics.has_recurrent_state:
            dependencies.append("state_adapter")
        if state_semantics.has_external_memory:
            dependencies.append("memory_adapter")
        modules.append(
            DesignModule(
                name="routing_cache",
                kind="cache_path",
                purpose="Expose cache or routing metadata used to steer the novel mechanism.",
                inputs=["hidden_state"],
                outputs=["cache_tensor"],
                depends_on=dependencies,
                required=False,
                assumption_source="heuristic_bundle_compiler",
            )
        )
    output_dependencies = ["core_backbone"]
    if state_semantics.has_recurrent_state:
        output_dependencies.append("state_adapter")
    if state_semantics.has_external_memory:
        output_dependencies.append("memory_adapter")
    modules.append(
        DesignModule(
            name="lm_head",
            kind="output_head",
            purpose="Project final hidden states into next-token logits.",
            inputs=["hidden_state"],
            outputs=["logits"],
            depends_on=output_dependencies,
            assumption_source="heuristic_bundle_compiler",
        )
    )
    return modules


def _build_core_purpose(bundle: ResearchIdeaBundle) -> str:
    summary = shorten(bundle.mechanism_summary.strip(), width=120, placeholder="...")
    return f"Implement the main experimental decoding mechanism described by: {summary}"


def _build_tensor_interfaces(modules: list[DesignModule], state_semantics: StateSemantics) -> list[TensorInterface]:
    tensor_interfaces = [
        TensorInterface(
            name="input_tokens",
            semantic_role="token_ids",
            producer="token_embedding",
            consumer="token_embedding",
            shape_notes="Batch-major token ids compatible with the GPT-2 tokenizer.",
            assumption_source="heuristic_bundle_compiler",
        ),
        TensorInterface(
            name="token_embeddings",
            semantic_role="embedded_tokens",
            producer="token_embedding",
            consumer="core_backbone",
            shape_notes="Dense hidden vectors for each input token.",
            assumption_source="heuristic_bundle_compiler",
        ),
        TensorInterface(
            name="hidden_state",
            semantic_role="decoder_hidden",
            producer="core_backbone",
            consumer="lm_head",
            shape_notes="Sequence-aligned hidden activations passed across experimental blocks.",
            assumption_source="heuristic_bundle_compiler",
        ),
        TensorInterface(
            name="logits",
            semantic_role="token_logits",
            producer="lm_head",
            consumer="lm_head",
            shape_notes="Vocabulary-sized next-token logits for autoregressive training and evaluation.",
            assumption_source="heuristic_bundle_compiler",
        ),
    ]
    module_names = {module.name for module in modules}
    if "state_adapter" in module_names:
        tensor_interfaces.extend(
            [
                TensorInterface(
                    name="state_tensor",
                    semantic_role="recurrent_state",
                    producer="state_adapter",
                    consumer="state_adapter",
                    shape_notes="Persistent latent state carried across steps or chunks.",
                    assumption_source="heuristic_bundle_compiler",
                ),
                TensorInterface(
                    name="state_conditioned_hidden",
                    semantic_role="state_augmented_hidden",
                    producer="state_adapter",
                    consumer="lm_head",
                    shape_notes="Hidden activations conditioned on recurrent state.",
                    assumption_source="heuristic_bundle_compiler",
                ),
            ]
        )
    if "memory_adapter" in module_names:
        tensor_interfaces.extend(
            [
                TensorInterface(
                    name="memory_tensor",
                    semantic_role="external_memory",
                    producer="memory_adapter",
                    consumer="memory_adapter",
                    shape_notes="Learned or retrieved memory contents surfaced to the decoder.",
                    assumption_source="heuristic_bundle_compiler",
                ),
                TensorInterface(
                    name="memory_context",
                    semantic_role="memory_features",
                    producer="memory_adapter",
                    consumer="lm_head",
                    shape_notes="Memory-derived features fused into the output pathway.",
                    assumption_source="heuristic_bundle_compiler",
                ),
            ]
        )
    if "routing_cache" in module_names:
        tensor_interfaces.append(
            TensorInterface(
                name="cache_tensor",
                semantic_role="routing_cache",
                producer="routing_cache",
                consumer="routing_cache",
                shape_notes="Optional cache or routing metadata used for debugging and ablations.",
                assumption_source="heuristic_bundle_compiler",
            )
        )
    return tensor_interfaces


def _build_training_plan(bundle: ResearchIdeaBundle) -> list[TrainingStagePlan]:
    plans: list[TrainingStagePlan] = []
    explicit_steps = bundle.implementation_requirements.copy()
    for phase in PHASES:
        dataset = dataset_plan_for_phase(phase)
        success_checks = _phase_success_checks(phase, bundle, explicit_steps)
        plans.append(
            TrainingStagePlan(
                stage=phase,
                objective=_phase_objective(phase, bundle),
                dataset_name=dataset["dataset_name"],
                dataset_description=dataset["description"],
                target_tokens=dataset["target_tokens"],
                success_checks=success_checks,
                assumption_source="heuristic_bundle_compiler",
            )
        )
    return plans


def _phase_objective(phase: str, bundle: ResearchIdeaBundle) -> str:
    if phase == "smoke":
        return "Validate shapes, loss flow, and architectural connectivity for the novel mechanism."
    if phase == "small":
        return "Check learnability and compare the mechanism against a compact baseline."
    return f"Escalate the strongest variant implied by '{bundle.title}' to the full budget tier."


def _phase_success_checks(phase: str, bundle: ResearchIdeaBundle, explicit_steps: list[str]) -> list[str]:
    checks = []
    if phase == "smoke":
        checks.append(f"Model remains below {DEFAULT_SMALLER_SCALE_CAP:,} parameters for smoke validation.")
        checks.append("Forward pass, backward pass, and evaluation hooks execute without shape mismatches.")
    elif phase == "small":
        checks.append("Training shows learnability on the small curated corpus.")
        checks.append("At least one evaluation compares against the internal baseline or prior run.")
    else:
        checks.append("The selected design variant remains within the configured parameter budget.")
        checks.append("Evaluation outputs are ready for comparison and promotion decisions.")
    if explicit_steps:
        checks.append(explicit_steps[min(len(explicit_steps) - 1, 0 if phase == "smoke" else 1 if phase == "small" else 2)])
    return checks


def _build_evaluation_plan(bundle: ResearchIdeaBundle) -> list[EvaluationTask]:
    targets = bundle.evaluation_targets or ["Compare against internal baseline, prior runs, and public references."]
    tasks: list[EvaluationTask] = []
    for index, target in enumerate(targets, start=1):
        tasks.append(
            EvaluationTask(
                name=f"evaluation_task_{index}",
                description=target,
                metrics=_extract_metrics(target),
                comparison_targets=_extract_comparison_targets(target),
                phase="small" if index == 1 else "full",
                assumption_source="heuristic_bundle_compiler",
            )
        )
    return tasks


def _extract_metrics(text: str) -> list[str]:
    lowered = text.lower()
    metrics = []
    if "perplexity" in lowered:
        metrics.append("perplexity")
    if "loss" in lowered:
        metrics.append("loss")
    if "benchmark" in lowered or "compare" in lowered:
        metrics.append("relative_quality")
    if not metrics:
        metrics.append("validation_score")
    return metrics


def _extract_comparison_targets(text: str) -> list[str]:
    lowered = text.lower()
    targets = []
    if "baseline" in lowered or "compare" in lowered:
        targets.append("internal_baseline")
    if "prior" in lowered:
        targets.append("prior_attempts")
    if "public" in lowered:
        targets.append("public_references")
    if not targets:
        targets.append("internal_baseline")
    return targets


def _build_ablation_plan(bundle: ResearchIdeaBundle, module_names: list[str]) -> list[AblationPlan]:
    plans: list[AblationPlan] = []
    for index, idea in enumerate(bundle.ablation_ideas, start=1):
        plans.append(
            AblationPlan(
                name=f"ablation_{index}",
                description=idea,
                target_modules=_match_modules(idea, module_names),
                phase="small",
                assumption_source="heuristic_bundle_compiler",
            )
        )
    return plans


def _build_failure_criteria(
    bundle: ResearchIdeaBundle, module_names: list[str], state_semantics: StateSemantics
) -> list[FailureCriterion]:
    criteria: list[FailureCriterion] = []
    for index, mode in enumerate(bundle.expected_failure_modes, start=1):
        criteria.append(
            FailureCriterion(
                name=f"failure_mode_{index}",
                description=mode,
                focus_area=_failure_focus_area(mode, state_semantics),
                target_modules=_match_modules(mode, module_names),
                assumption_source="heuristic_bundle_compiler",
            )
        )
    return criteria


def _build_implementation_milestones(bundle: ResearchIdeaBundle, module_names: list[str]) -> list[ImplementationMilestone]:
    if bundle.implementation_requirements:
        milestones: list[ImplementationMilestone] = []
        prior_name = ""
        for index, step in enumerate(bundle.implementation_requirements, start=1):
            name = f"milestone_{index}"
            milestones.append(
                ImplementationMilestone(
                    name=name,
                    description=step,
                    depends_on=[prior_name] if prior_name else [],
                    assumption_source="research_bundle",
                )
            )
            prior_name = name
        return milestones

    return [
        ImplementationMilestone(
            name="milestone_1",
            description="Implement the module inventory implied by the architecture plan.",
            depends_on=[],
            assumption_source="compiler_default",
        ),
        ImplementationMilestone(
            name="milestone_2",
            description=f"Connect {'/'.join(module_names[:3])} into a smoke-testable training path.",
            depends_on=["milestone_1"],
            assumption_source="compiler_default",
        ),
        ImplementationMilestone(
            name="milestone_3",
            description="Wire evaluation and ablation hooks before escalating budget tiers.",
            depends_on=["milestone_2"],
            assumption_source="compiler_default",
        ),
    ]


def _build_compatibility_projection(
    bundle: ResearchIdeaBundle, pattern_label: str, state_semantics: StateSemantics
) -> CompatibilityProjection:
    raw_brief = _build_raw_brief(bundle)
    normalized_brief = _build_normalized_brief(bundle, pattern_label)
    novelty_claims = _build_novelty_claims(bundle, pattern_label, state_semantics)
    training_curriculum = _build_training_curriculum(bundle)
    inspirations = _build_inspirations(bundle)
    evaluation_intent = _build_evaluation_intent(bundle)
    return CompatibilityProjection(
        raw_brief=raw_brief,
        normalized_brief=normalized_brief,
        hypothesis=f"This idea can improve a general-purpose LM by exploring: {normalized_brief}.",
        novelty_claims=novelty_claims,
        training_curriculum_outline=training_curriculum,
        inspirations_consulted=inspirations,
        evaluation_intent=evaluation_intent,
    )


def _build_module_graph_summary(module_names: list[str], state_semantics: StateSemantics) -> str:
    summary = f"Primary data path: {' -> '.join(module_names)}."
    if state_semantics.has_recurrent_state or state_semantics.has_external_memory:
        return summary + f" {state_semantics.summary}"
    return summary


def _build_raw_brief(bundle: ResearchIdeaBundle) -> str:
    parts = [bundle.title.strip(), bundle.mechanism_summary.strip(), bundle.novelty_rationale.strip()]
    return " ".join(part for part in parts if part)


def _build_normalized_brief(bundle: ResearchIdeaBundle, pattern_label: str) -> str:
    keywords = _extract_keywords(f"{bundle.title} {bundle.mechanism_summary} {pattern_label}")
    seed = " ".join(keywords) or bundle.title or bundle.mechanism_summary
    return shorten(seed, width=96, placeholder="...")


def _extract_keywords(raw_text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9\\-]+", raw_text.lower())
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
        "decoder",
    }
    unique: list[str] = []
    for token in tokens:
        if token in stop_words or len(token) < 5:
            continue
        if token not in unique:
            unique.append(token)
    return unique[:8]


def _build_novelty_claims(
    bundle: ResearchIdeaBundle, pattern_label: str, state_semantics: StateSemantics
) -> list[str]:
    claims = [bundle.novelty_rationale]
    claims.append(f"Compiler-selected architecture pattern: {pattern_label.replace('_', ' ')}.")
    if state_semantics.has_recurrent_state:
        claims.append("Preserve explicit recurrent or stateful behavior in the final implementation.")
    if bundle.source_titles:
        claims.append(f"Recombine insights from: {', '.join(bundle.source_titles[:3])}.")
    if bundle.ablation_ideas:
        claims.append(f"Plan explicit ablations for: {bundle.ablation_ideas[0]}")
    claims.append("Reject template decoder-only stacks with only cosmetic changes.")
    return claims


def _build_training_curriculum(bundle: ResearchIdeaBundle) -> list[str]:
    curriculum = [
        f"Smoke-test reduced variant near {DEFAULT_SMALLER_SCALE_CAP:,} parameters for math and logic.",
        "Run a learnability check on the target-scale model with a small training subset.",
    ]
    if bundle.implementation_requirements:
        curriculum.append(bundle.implementation_requirements[0])
    else:
        curriculum.append("Escalate to a production-like corpus only after the agent records why the idea looks promising.")
    return curriculum


def _build_inspirations(bundle: ResearchIdeaBundle) -> list[str]:
    inspirations = [
        "SOTA models may be studied for inspiration only.",
        "Any borrowed mechanism must be declared and recombined into an original design.",
    ]
    for title in bundle.source_titles[:3]:
        inspirations.append(f"Inspiration source: {title}")
    return inspirations


def _build_evaluation_intent(bundle: ResearchIdeaBundle) -> str:
    if bundle.evaluation_targets:
        return "; ".join(bundle.evaluation_targets[:3])
    return "compare against internal baseline, prior runs, and public references"


def _match_modules(text: str, module_names: list[str]) -> list[str]:
    lowered = text.lower()
    matches = []
    aliases = {
        "token_embedding": ("embedding", "token"),
        "core_backbone": ("core", "backbone", "decoder", "block", "architecture"),
        "state_adapter": ("state", "recurrent"),
        "memory_adapter": ("memory", "retrieval"),
        "routing_cache": ("cache", "routing"),
        "lm_head": ("head", "logits", "output"),
    }
    for module_name in module_names:
        if module_name in lowered:
            matches.append(module_name)
            continue
        for alias in aliases.get(module_name, ()):
            if alias in lowered:
                matches.append(module_name)
                break
    return matches


def _failure_focus_area(text: str, state_semantics: StateSemantics) -> str:
    lowered = text.lower()
    if "train" in lowered or "loss" in lowered or "destabil" in lowered:
        return "training_stability"
    if "memory" in lowered or "retrieval" in lowered:
        return "memory_path"
    if "cache" in lowered or "route" in lowered:
        return "routing_path"
    if state_semantics.has_recurrent_state:
        return "state_management"
    return "architecture_fit"
