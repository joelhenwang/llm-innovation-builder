# auto-llm-innovator Completion TODO

This document explains what `auto-llm-innovator` still needs in order to become a complete autonomous experiment builder for ideas produced by `auto-llm-researcher`.

The goal is not only to intake an idea and scaffold a bundle, but to autonomously:

- accept structured ideas from `auto-llm-researcher`
- generate real Python and PyTorch packages for each idea
- create model classes, submodules, trainers, evaluators, and tests
- run smoke, small, and full experiments
- debug and repair failures
- compare results against baselines and prior attempts
- decide whether to promote, revise, or reject an idea

## Current State

The repository already has a solid orchestration skeleton:

- idea intake and normalization
- originality review
- phase config generation
- skill routing and skill snapshots
- attempt tracking and reports
- dynamic execution of generated `train.py`

The main gap is that the system mostly scaffolds artifacts today. It does not yet function as a fully autonomous model-building and experiment-running agent.

## Definition Of Complete

`auto-llm-innovator` should be considered complete when it can:

1. Ingest a structured idea bundle from `auto-llm-researcher`.
2. Convert that idea into a rich internal design representation.
3. Generate a real multi-file Python package implementing the idea.
4. Run preflight checks and smoke tests automatically.
5. Train and evaluate the idea across increasing budget tiers.
6. Repair broken generations automatically when possible.
7. Compare outcomes to baselines and previous attempts.
8. Emit reproducible artifacts, reports, and promotion decisions.

## Priority Roadmap

Implementation should happen in roughly this order:

1. Define the researcher-to-innovator handoff contract.
2. Add a design IR between idea text and code generation.
3. Build a reusable PyTorch experiment harness.
4. Replace template-only generation with multi-file code generation.
5. Add automated validation and smoke gates.
6. Add real evaluation and comparison logic.
7. Add failure classification and repair loops.
8. Add run scheduling, resource control, and richer experiment lineage.

## Autonomy track for full lab operation

The next innovator work should also be measured by how well it supports Deer Flow running the workspace as a persistent autonomous operator.

Priority autonomy additions for this repo:

- machine-readable phase outcome summaries: emit concise structured outcomes that let Deer Flow decide whether to continue, stop, or revisit later without reading every artifact
- promotion recommendations: expose soft recommendation signals for phase progression while keeping only obvious hard blockers as hard stops
- stop reasons: persist explicit phase and attempt stop reasons so autonomous loops can distinguish clean completion, budget exhaustion, repair recovery, resource rejection, and hard failure
- resume-friendly latest-state summaries: make the current idea state cheap to load for the next pass, including latest completed phase, active attempt state, and linked report paths
- stronger agent-facing reporting: keep the rich underlying artifacts, but add compact summary surfaces that are optimized for orchestration decisions

Definition of done for the innovator autonomy track:

- Deer Flow can decide whether to continue or defer work from a concise structured idea summary
- interrupted or repeated autonomous passes can reuse innovator state safely
- promotion judgment can rely on visible signals without requiring strict hard gates for every non-ideal outcome

## TODO Items

### 1. Define A Structured Input Contract From `auto-llm-researcher`

Implementation overview:

- Added `src/auto_llm_innovator/handoff/` with a normalized `ResearchIdeaBundle` contract.
- Implemented loaders for free-text briefs, researcher single-candidate JSON artifacts, and researcher mix JSON artifacts.
- Added manual validation for required mechanism, novelty, tokenizer compatibility, and minimum experiment intent.
- Added a compiler from `ResearchIdeaBundle` to `IdeaSpec` so existing orchestration continues to work downstream.
- Updated `innovator submit` to accept either a free-text brief or `--bundle-file <path>`.
- Persisted `handoff_bundle.json` next to `idea_spec.json` for each submitted idea.
- Added test coverage for handoff normalization, compilation, CLI success paths, and validation failures.

Status:

- Phase 1 foundation is now implemented as the intake boundary for the rest of the roadmap.
- The current downstream runtime still consumes `IdeaSpec`, so later phases should extend from the handoff/compiler boundary rather than bypassing it.

What should be implemented:

- A formal schema for idea bundles passed from `auto-llm-researcher`.
- A loader that can accept either a local JSON artifact or direct Python objects.
- Validation code that rejects underspecified ideas before code generation starts.

Why:

- The current system starts from a normalized idea brief, but a complete system needs richer, structured experimental intent.
- The innovator should not infer everything from prose if the upstream researcher already knows important constraints.
- A strict handoff contract will reduce brittle prompt-only behavior.

How:

- Create a new module such as `src/auto_llm_innovator/handoff/`.
- Add a dataclass or typed model like `ResearchIdeaBundle`.
- Include fields such as:
  - `source_candidate_id`
  - `title`
  - `mechanism_summary`
  - `novelty_rationale`
  - `implementation_requirements`
  - `known_constraints`
  - `dataset_requirements`
  - `evaluation_targets`
  - `ablation_ideas`
  - `expected_failure_modes`
  - `compute_budget_hint`
- Add schema validation and normalization code before `IdeaSpec` creation.
- Update the CLI so `innovator submit` can accept either free text or a structured bundle file.

Suggested deliverables:

- `src/auto_llm_innovator/handoff/models.py`
- `src/auto_llm_innovator/handoff/loaders.py`
- `tests/test_handoff.py`

### 2. Introduce A Design IR

Implementation overview:

- Added `src/auto_llm_innovator/design_ir/` with dataclass-based models for `DesignIR`, architecture/state semantics, module inventory, tensor interfaces, training stages, evaluation tasks, ablations, failure criteria, milestones, and the `IdeaSpec` compatibility projection.
- Implemented a deterministic compiler from `ResearchIdeaBundle` to `DesignIR` that infers an explicit module graph and planning structure from Phase 1 bundle fields instead of regenerating everything from prose.
- Implemented `DesignIR` validation covering the repo-wide structural constraints: GPT-2 tokenizer compatibility, parameter cap, required core module presence, valid module dependencies, valid tensor producer/consumer references, complete smoke/small/full training stages, and evaluation-task presence.
- Updated submit-time orchestration so `InnovatorEngine.submit()` now compiles and validates `DesignIR`, persists `design_ir.json`, and then projects `DesignIR` into `IdeaSpec` for downstream compatibility.
- Kept the existing runtime and `run()` flow on `IdeaSpec` for now, while adding a short pointer in `notes/design.md` and `README.md` that `design_ir.json` is now the richer internal planning artifact.
- Added test coverage for `DesignIR` compilation, validation failures, `DesignIR -> IdeaSpec` projection compatibility, and submit-time persistence of `design_ir.json`.

Status:

- Phase 2 foundation is now implemented as the internal planning layer between handoff intake and downstream generation/orchestration.
- `DesignIR` is now the richer source of truth for architecture/training/evaluation intent, but current execution still uses `IdeaSpec` as the compatibility layer.
- Later phases should extend from the `DesignIR` boundary rather than adding more behavior directly on top of `IdeaSpec`.

What should be implemented:

- A design intermediate representation that sits between idea intake and code generation.

Why:

- Natural language is too weak as the direct input to multi-file code generation.
- The system needs a structured, inspectable representation of architecture, training, and evaluation plans.
- The IR gives the system something it can validate, repair, diff, and evolve across retries.

How:

- Add a package such as `src/auto_llm_innovator/design_ir/`.
- Start from the implemented Phase 1 boundary:
  - load `handoff_bundle.json` as the authoritative structured intake artifact
  - compile `ResearchIdeaBundle` into `DesignIR`
  - continue compiling `DesignIR` into `IdeaSpec` for compatibility during the transition
- Do not make Phase 2 a direct `IdeaSpec` replacement on day one.
  - keep `IdeaSpec` as a downstream compatibility layer while `DesignIR` becomes the richer internal planning artifact
  - once `DesignIR` is stable, move generation/orchestration stages to consume it directly
- Represent the following explicitly:
  - architecture graph
  - module inventory
  - tensor interfaces and shape assumptions
  - recurrent/cache/memory semantics
  - optimizer and scheduler plan
  - data pipeline and curriculum plan
  - evaluation tasks and metrics
  - ablations
  - failure criteria
  - implementation milestones
- Build a compiler step:
  - `ResearchIdeaBundle` -> `DesignIR`
  - `DesignIR` -> `IdeaSpec` compatibility projection
- Add validation rules:
  - parameter cap under 2.1B
  - GPT-2 tokenizer compatibility
  - no undefined modules
  - all training and evaluation stages mapped
- Reuse Phase 1 bundle fields instead of re-inferring them from prose:
  - `mechanism_summary` should seed the architecture and module plan
  - `implementation_requirements` should seed milestones and build order
  - `evaluation_targets` and `ablation_ideas` should seed evaluation and ablation sections
  - `expected_failure_modes` should seed failure criteria and validator checks
- Keep the first Phase 2 slice narrow:
  - define `DesignIR` models
  - add a compiler from `ResearchIdeaBundle`
  - validate the IR
  - persist `design_ir.json` in the idea directory
  - leave code generation behavior unchanged until the IR is stable

Suggested deliverables:

- `src/auto_llm_innovator/design_ir/models.py`
- `src/auto_llm_innovator/design_ir/compiler.py`
- `src/auto_llm_innovator/design_ir/validator.py`
- `ideas/<idea_id>/design_ir.json`
- `tests/test_design_ir.py`

### 3. Build A Reusable PyTorch Experiment Harness

Implementation overview:

- Added `src/auto_llm_innovator/runtime/` with a shared runtime config compiler, checkpoint helpers, metric logging, evaluation execution, and a generic phase runner.
- Implemented a `DesignIR` -> runtime phase config adapter so the harness now derives required modules, recurrent/memory/cache expectations, stage objectives, evaluation tasks, and runtime checks from `design_ir.json` plus `config/<phase>.json`.
- Updated generated `train.py` to become a thin adapter that delegates into the shared runtime while preserving the existing `run_phase(...) -> dict` contract used by `execute_phase()`.
- Updated generated `model.py` to expose a minimal harness-compatible plugin contract: `ModelConfig`, `build_model(...)`, plugin metadata, and optional evaluation-hook registration.
- Kept `InnovatorEngine.run()` and `training.execute_phase()` unchanged at the orchestration boundary so reporting and attempt tracking still work without a larger migration.
- Added runtime-focused tests covering runtime config derivation, readable plugin-contract failures, checkpoint/resume behavior, evaluation artifact generation, and end-to-end orchestration compatibility.

Status:

- Phase 3 foundation is now implemented as the shared runtime boundary between `DesignIR` planning and idea-specific generated code.
- The current runtime is intentionally narrow and deterministic: it provides harness ownership of training/evaluation/checkpoint/logging flow, but generated idea code is still template-based rather than true multi-file architecture-specific package generation.
- Later phases should extend from the runtime plugin contract rather than reintroducing bespoke per-idea training loops.

How Phase 3 should start, given the implemented Phase 2 boundary:

- Treat `design_ir.json` as the planning artifact that seeds the runtime contract.
- Do not design the harness around free-form `IdeaSpec` text alone anymore; use `DesignIR` to decide what model hooks, state paths, evaluation hooks, and staged training requirements the harness must expose.
- Start with a thin adapter layer from `DesignIR` to runtime configuration so the harness can consume explicit:
  - module inventory
  - tensor/state/cache semantics
  - stage-specific training goals
  - evaluation tasks and ablations
  - failure criteria that should surface as runtime checks
- Keep `IdeaSpec` only as the compatibility layer for unchanged orchestration/reporting paths while the runtime begins to consume `DesignIR`-derived config.
- Build the Phase 3 first slice so generated idea-specific code implements only model-specific components and registration hooks, while the shared harness owns the generic train/eval/checkpoint/logging flow.
- Prefer runtime interfaces that map cleanly to the current `DesignIR` shape, so Phase 4 can generate code from `DesignIR` into harness-compatible modules instead of bespoke per-idea scripts.

What should be implemented:

- A real internal training and evaluation runtime that generated ideas can plug into.

Why:

- Today the repo can execute generated `train.py`, but it lacks a strong shared harness.
- Without a common harness, each generated idea will have to reinvent training loops, checkpointing, logging, and evaluation.
- A shared harness makes generated code smaller, safer, and easier to repair.

How:

- Add a package like `src/auto_llm_innovator/runtime/`.
- Provide reusable components for:
  - model registry and factory hooks
  - configuration loading
  - dataset and dataloader setup
  - optimizer and scheduler creation
  - mixed precision and gradient scaling
  - gradient accumulation
  - checkpointing and resume
  - metric logging
  - evaluation hooks
  - seed and reproducibility control
- Generated idea-specific files should implement only the custom logic and register with the harness.
- Avoid generating giant custom training loops for every idea when a parameterized harness would work.

Suggested deliverables:

- `src/auto_llm_innovator/runtime/config.py`
- `src/auto_llm_innovator/runtime/train_loop.py`
- `src/auto_llm_innovator/runtime/eval_loop.py`
- `src/auto_llm_innovator/runtime/checkpoints.py`
- `src/auto_llm_innovator/runtime/logging.py`
- `tests/test_runtime.py`

### 4. Replace Flat Templates With Multi-File Project Generation

Implementation overview:

- Added `src/auto_llm_innovator/generation/` with a package layout planner, deterministic renderers, and a lightweight source-normalization pass.
- Replaced the old flat `_write_plugin_bundle` path with `generate_idea_package(...)`, so submit-time generation now emits an idea-local `package/` tree plus compatibility `train.py` and `eval.py` entrypoints.
- Generated package contents from `DesignIR` instead of idea prose, including:
  - `package/config.py`
  - `package/plugin.py`
  - `package/modeling/components.py`
  - `package/modeling/model.py`
  - conditional `package/modeling/state.py`
  - `package/evaluation/hooks.py`
  - idea-local import/shape/smoke tests
- Kept the Phase 3 runtime contract unchanged:
  - top-level `train.py` still exports `run_phase(...)`
  - the shared runtime still validates `ModelConfig`, `build_model(...)`, `describe_plugin()`, and `register_evaluation_hooks()`
- Added a `generation_manifest.json` artifact listing generated files and `DesignIR` module names for debugging and future repair work.
- Added test coverage for package layout generation, conditional state-file emission, generated evaluation-hook names, orchestration compatibility, and CLI bundle submission with package-based outputs.

Status:

- Phase 4 foundation is now implemented as a deterministic multi-file package generator layered on top of the Phase 3 shared runtime.
- The current generator now changes code shape correctly, but it still emits a small fixed package structure rather than a deeply architecture-specific per-module file graph.
- Later phases should extend from the generated package boundary and runtime plugin contract instead of reintroducing single-file model generation.

How Phase 4 should start, given the implemented Phase 3 boundary:

- Treat the shared runtime plugin contract as the target interface for code generation:
  - generated packages should provide harness-compatible model builders, module registration metadata, and optional evaluation hooks
  - generated code should not own bespoke train/eval/checkpoint orchestration anymore
- Use `DesignIR` as the package-generation source of truth:
  - `modules` should map to generated Python modules/classes
  - `tensor_interfaces` should inform function signatures and wiring
  - `architecture.state_semantics` should determine whether generated packages include recurrent state, memory, and cache-path components
- Keep the current generated `train.py` as a thin compatibility adapter, but make it import from a generated package layout instead of a single flat `model.py`.
- Make the first Phase 4 slice generate only harness-compatible code structure:
  - package layout
  - module files
  - registration metadata
  - idea-local tests
  - avoid changing the shared runtime contract at the same time
- Prefer package generation that keeps generic runtime behavior centralized and pushes only idea-specific architecture logic into generated files.

What should be implemented:

- A code generator that creates a full package layout per idea, not just `model.py`, `train.py`, and `eval.py`.

Why:

- Real architecture experiments need multiple modules, helpers, configs, and tests.
- Flat files will not scale to complex ideas like recurrent memory layers, routing blocks, or retrieval modules.
- Multi-file generation is required for maintainability and automated repair.

How:

- Replace the current `_write_plugin_bundle` behavior with generated package structures such as:

```text
ideas/<idea_id>/
  package/
    __init__.py
    config.py
    modeling/
      __init__.py
      blocks.py
      memory.py
      routing.py
      model.py
    training/
      __init__.py
      data.py
      losses.py
      pipeline.py
    evaluation/
      __init__.py
      tasks.py
      metrics.py
  tests/
    test_imports.py
    test_shapes.py
    test_smoke.py
  entrypoints/
    train.py
    eval.py
```

- Create a generator package such as `src/auto_llm_innovator/generation/`.
- Generate code from `DesignIR`, not directly from idea prose.
- Keep generated files narrow and composable.
- Add formatting and linting passes after generation.

Suggested deliverables:

- `src/auto_llm_innovator/generation/layout.py`
- `src/auto_llm_innovator/generation/renderers/`
- `src/auto_llm_innovator/generation/postprocess.py`
- `tests/test_generation.py`

### 5. Add Automated Preflight Validation

Implementation overview:

- Added `src/auto_llm_innovator/validation/` with package import, plugin contract, model instantiation, forward-pass, synthetic loss/train-step, checkpoint, and eval-hook checks.
- Refactored `training.execute_phase()` to validate the generated `package/` tree before phase execution instead of importing `train.py` first.
- Reused the shared runtime path by extracting helper logic from `runtime/train_loop.py` for plugin-contract validation, synthetic batch generation, model calling, and shape inspection.
- Added machine-readable `preflight-report.json` artifacts in each phase run directory, including ordered check results, failure categories, retry metadata, and failing modules/files.
- Added a bounded repair retry:
  - retryable preflight failures trigger one deterministic regeneration of the Phase 4 package from stored `idea_spec.json` and `design_ir.json`
  - preflight reruns once after regeneration
  - unrecovered failures return `repair_preflight` without entering the main runtime loop
- Updated orchestration so `innovator run --phase all` stops after the first failed preflight/phase result instead of continuing into later phases.
- Added coverage for happy-path preflight, unrecoverable failures, regeneration retry recovery, and phase-halting orchestration behavior.

Status:

- Phase 5 foundation is now implemented as a generated-package preflight boundary between Phase 4 code generation and shared-runtime phase spend.
- The current preflight path validates runtime/package wiring well and emits repair-friendly artifacts, but it still uses the synthetic Phase 3/5 runtime semantics rather than true phase-specific experiment budgets.
- Later phases should build on `preflight-report.json`, `execute_phase()`, and the shared runtime helpers instead of adding a second validation path.

How Phase 5 should start, given the implemented Phase 4 boundary:

- Treat the generated package plus the shared runtime as the preflight target surface:
  - validate generated package imports before phase execution
  - validate the runtime plugin contract against the generated `package.plugin` module
  - keep preflight separate from the actual training loop so failures are classified earlier and more cleanly
- Reuse the generated package-local tests and Phase 3 runtime checks as building blocks rather than inventing an entirely separate validation path.
- Make preflight operate on the real generated package structure:
  - package import sanity
  - `ModelConfig` construction
  - `build_model(...)` instantiation
  - dummy forward pass matching `DesignIR` state semantics
  - runtime-compatible output shape validation
  - one-batch loss/optimization/checkpoint sanity checks
- Insert preflight between generation and phase spend:
  - `execute_phase()` should run preflight first
  - on failure, persist preflight artifacts and stop before the main train loop consumes phase budget
- Keep the first Phase 5 slice narrow:
  - validate the generated package and runtime wiring
  - defer full repair-loop integration until the preflight outputs and failure taxonomy are stable
- Prefer preflight outputs that are easy for later repair logic to consume:
  - machine-readable check results
  - clear failure categories
  - references to the failing generated file/module when possible

What should be implemented:

- A preflight stage before each phase, especially before `small` and `full`.

Why:

- Autonomous code generation will frequently produce broken imports, invalid shapes, or non-finite losses.
- Catching these failures early is essential to avoid wasting compute.
- Preflight is the boundary between code generation and real experiment spend.

How:

- Add a validation pipeline such as:
  - import test
  - config parse test
  - model instantiation test
  - parameter count test
  - dummy forward pass
  - loss computation sanity test
  - one-batch optimization step
  - checkpoint save/load test
  - eval entrypoint sanity check
- Implement this in a `validation/` package and make it part of `execute_phase`.
- Persist preflight outputs into the run directory.
- If preflight fails, route into a repair loop instead of starting the actual phase.

Suggested deliverables:

- `src/auto_llm_innovator/validation/preflight.py`
- `src/auto_llm_innovator/validation/model_checks.py`
- `src/auto_llm_innovator/validation/train_checks.py`
- `tests/test_preflight.py`

### 6. Add Real Smoke, Small, And Full Run Semantics

Implementation overview:

- Added `src/auto_llm_innovator/runtime/phases.py` with an explicit `RuntimePhaseSettings` model and default per-phase semantics for `smoke`, `small`, and `full`.
- Refactored `runtime/config.py` so each phase now compiles a nested `runtime` block from `config/<phase>.json` into `RuntimePhaseConfig.settings` instead of relying on hardcoded helper functions for steps, sequence length, and batch size.
- Updated submit-time config generation so each idea now gets phase-local JSON with:
  - `runtime.max_steps`
  - `runtime.max_wall_time_seconds`
  - `runtime.sequence_length`
  - `runtime.batch_size`
  - `runtime.checkpoint_every_steps`
  - `runtime.resume_enabled`
  - `runtime.evaluation_scope`
  - `runtime.dataset_slice`
- Updated shared runtime execution so phase behavior now obeys the compiled settings:
  - step budgets come from `runtime.max_steps`
  - wall-time exits are tracked via `stop_reason`
  - smoke disables resume
  - small/full enable resume
  - periodic checkpoint cadence is phase-specific
  - evaluation scope is phase-specific
- Kept Phase 5 preflight unchanged as the fixed gate before runtime execution.
- Changed promotion semantics in orchestration so `innovator run --phase all` advances only when the previous phase returns exact `status == "passed"`.
- Extended runtime summaries and consumed-budget metadata with resolved phase settings and stop reasons so later repair/evaluation logic can distinguish hard failures from budget-limited warning outcomes.
- Added test coverage for distinct compiled phase settings, smoke-vs-small resume behavior, wall-time warning outcomes, direct single-phase execution, and phase-promotion stopping rules.

Status:

- Phase 6 foundation is now implemented as a runtime-config-driven phase execution layer on top of the Phase 5 preflight boundary.
- The repo now has a real distinction between smoke, small, and full in config shape, runtime budgets, checkpoint policy, evaluation scope, and promotion behavior.
- The current runtime is still synthetic/dry-run oriented, so the new semantics are budget-accurate and promotion-accurate, but not yet backed by true hardware/resource-aware repair decisions.

How Phase 6 should start, given the implemented Phase 5 boundary:

- Treat preflight as the fixed gate before any phase spend:
  - keep `execute_phase()` as `preflight -> phase runtime`
  - do not fold smoke validation back into preflight
- Build the phase distinction inside shared runtime config/execution, not inside generated packages:
  - generated `package/` code should stay phase-agnostic
  - phase meaning should live in runtime settings and promotion rules
- Start by extending the current runtime config boundary:
  - replace the simple `_steps_for_phase`, `_sequence_length_for_phase`, and `_batch_size_for_phase` helpers with explicit phase runtime models
  - make those models drive `max_steps`, wall-time caps, dataset slicing, checkpoint cadence, and evaluation scope
- Keep the first Phase 6 slice narrow:
  - make smoke a true post-preflight tiny run
  - make small a constrained but longer budgeted run
  - leave full as the largest budgeted runtime path without adding repair-loop behavior yet
- Reuse the new preflight artifacts when defining promotion rules:
  - if preflight fails, no phase starts
  - if smoke runtime fails, do not promote to small
  - if small does not meet phase success checks, do not promote to full

What should be implemented:

- A genuine distinction between smoke, small, and full phases beyond config values.

Why:

- Phases should represent different experimental goals, not just different JSON files.
- The system needs clear promotion rules between phases.

How:

- Define each phase explicitly:
  - `smoke`: import, forward, one batch, tiny dataset, very short training
  - `small`: constrained but meaningful experiment, limited wall time and parameter/data budget
  - `full`: full planned budget with checkpointing and evaluation suite
- Add phase-specific runtime settings:
  - max steps
  - max wall time
  - batch size
  - dataset slice
  - checkpoint cadence
  - early stop thresholds
- Persist phase summaries in a common schema.

Suggested deliverables:

- stronger phase runtime config models
- `src/auto_llm_innovator/runtime/phases.py`
- `tests/test_phases.py`

### 7. Add Failure Classification And Repair Loops

Implementation overview:

- Added `src/auto_llm_innovator/repair/` with structured repair models, failure classification, deterministic repair strategies, and persisted repair history/artifacts.
- Refactored `training.execute_phase()` so phase execution now works as a bounded loop around the existing phase boundary:
  - compile runtime config
  - run preflight
  - classify failure when needed
  - apply deterministic repair when the classification is repairable and budget remains
  - restart from the phase boundary after repair
- Kept the first Phase 7 slice narrow and local:
  - only true `failed` outcomes enter repair
  - `passed_with_warnings` outcomes such as wall-time-limited phases remain non-repairable
  - repairs stay phase-local within the same attempt instead of creating new attempt lineage
- Implemented initial deterministic repair targets that match the current generated package/runtime shape:
  - generated import/module contract failures
  - plugin/runtime interface mismatches
  - invalid runtime phase settings in `config/<phase>.json`
  - generated forward-output/runtime shape contract failures
  - checkpoint/evaluation wiring regressions
  - simple model-construction surface regressions
- Persisted repair artifacts under `runs/<attempt>/<phase>/repair/`, including failure classification, repair history, before/after snapshots, diffs, and repair rationale.
- Extended `PhaseResult`, ledger/reporting, and decision-report output so later phases can distinguish:
  - failed without repair
  - recovered after repair
  - failed after exhausting repair budget
  - warning-only non-repair outcomes
- Tightened runtime failure semantics so true runtime contract/wiring failures now return `status == "failed"`, while budget-limited exits still return `passed_with_warnings`.
- Added repair-focused tests covering preflight recovery, runtime-setting normalization, runtime-output repair, non-repairable unknown runtime classification, and bounded repair behavior.

Status:

- Phase 7 foundation is now implemented as a bounded failure-classification and deterministic-repair layer on top of the Phase 6 phase-execution boundary.
- The repo can now classify structured preflight/runtime failures, attempt narrow local repairs, and persist repair lineage inside the phase run directory.
- The current repair system is intentionally deterministic and rule-based; it does not yet make richer evaluation decisions about whether a repaired run is actually promising beyond phase pass/fail semantics.

How Phase 7 should start, given the implemented Phase 6 boundary:

- Treat Phase 5 preflight and Phase 6 runtime outcomes as the two structured failure sources:
  - `preflight-report.json` already contains ordered checks, retry metadata, and failing modules/files
  - phase runtime summaries now include `stop_reason`, exact phase status, and resolved runtime settings
- Build repair around those artifacts instead of inventing a new error surface:
  - classify preflight failures separately from runtime failures
  - use `stop_reason` to distinguish repairable contract/wiring issues from budget-limited warning outcomes
  - preserve the existing `execute_phase()` flow as `preflight -> runtime -> repair decision`, not `repair -> preflight`
- Keep the first Phase 7 slice narrow:
  - only trigger repair for true failed outcomes, not `passed_with_warnings`
  - use exact failure categories already emitted by preflight when available
  - add bounded runtime failure classification before attempting any code/config repair
- Start the repair loop at the phase boundary, not inside generated packages:
  - let `training.execute_phase()` or a nearby orchestration layer decide whether to invoke repair
  - keep generated `package/` code phase-agnostic and repair-targeted from the outside
- Prefer early repair targets that match the current implementation shape:
  - generated import/module contract failures
  - plugin/runtime interface mismatches
  - invalid phase JSON runtime settings
  - runtime output-shape violations
  - checkpoint/evaluation wiring failures

What should be implemented:

- A closed-loop repair system that can inspect failures and patch generated code or configs.

Why:

- This is the difference between a scaffold and an autonomous experimenter.
- Most generated experiments will fail at first for ordinary reasons: imports, shapes, missing config keys, unstable losses, or broken eval hooks.
- The system should attempt bounded repair before abandoning an idea.

How:

- Add a `repair/` package.
- Define failure categories such as:
  - import failure
  - syntax error
  - missing symbol
  - tensor shape mismatch
  - parameter cap violation
  - NaN or inf loss
  - OOM
  - checkpoint failure
  - evaluation contract failure
- Capture structured error reports from preflight and training stages.
- Route the error plus the current generated package back through a repair agent or rule-based fixer.
- Keep a bounded retry count per phase.
- Persist diffs and repair rationale in the run artifacts.

Suggested deliverables:

- `src/auto_llm_innovator/repair/models.py`
- `src/auto_llm_innovator/repair/classifier.py`
- `src/auto_llm_innovator/repair/loop.py`
- `tests/test_repair.py`

### 8. Add A Stronger Evaluation System

Implementation overview:

- Added `src/auto_llm_innovator/evaluation/models.py` with a structured `EvaluationResult` schema plus `EvaluationSignal` and `PhaseEvaluationSummary` so evaluation can aggregate runtime, comparison, and reliability signals without changing the existing `PhaseResult` contract.
- Added `src/auto_llm_innovator/evaluation/runner.py` to build `EvaluationResult` from the current Phase 7/Phase 6 artifacts:
  - `PhaseResult`
  - `<phase>-summary.json`
  - `evaluation-report.json`
  - `repair/failure-classification.json`
  - `repair/repair-history.json`
- Implemented the first reliability-aware evaluation slice:
  - clean passes are treated as the strongest outcome
  - repaired passes remain valid but are downgraded to caution
  - `passed_with_warnings` outcomes are treated as budget-limited or inconclusive rather than equivalent to clean passes
  - failed-after-repair outcomes become explicit stop signals
- Expanded `src/auto_llm_innovator/evaluation/comparator.py` so baseline comparisons now include recommendation, caution flags, repair counts, and stop reasons in addition to raw loss deltas.
- Updated `InnovatorEngine.run()` to build and persist `reports/<attempt>-evaluation.json` after the phase loop, and updated decision reporting so reports now explain both technical execution and whether the run looks promising enough to continue.
- Kept the first Phase 8 slice narrow:
  - no ablation execution yet
  - no new runtime instrumentation yet
  - planned ablations from `DesignIR` are surfaced as planned-but-not-yet-run metadata in the report
- Added focused tests for clean passes, repaired passes, warning-only outcomes, failed-after-repair outcomes, missing baseline metrics, evaluation metric propagation, and report/artifact persistence.

Status:

- Phase 8 foundation is now implemented as a reliability-aware aggregation and reporting layer on top of the existing phase execution, runtime, and repair boundaries.
- The repo can now distinguish “executed successfully” from “worth continuing” using structured evaluation artifacts instead of a raw loss-only comparison.
- The current system still uses coarse runtime practicality signals and phase-local evaluation task outputs; richer baseline families, prior-attempt ranking, and real ablation execution are still future work.

How Phase 8 should start, given the implemented Phase 7 boundary:

- Treat the new Phase 7 artifacts as part of the evaluation input surface, not just debugging metadata:
  - phase summaries already contain exact phase status, stop reason, runtime settings, and basic metrics
  - repair artifacts now show whether a run passed cleanly, passed after repair, or failed after exhausting bounded repair
- Build richer evaluation on top of the current shared runtime/reporting flow instead of bypassing it:
  - keep `execute_phase()` as the owner of phase execution and repair
  - let Phase 8 consume the resulting runtime summaries, evaluation outputs, and repair lineage to decide whether outcomes are meaningful
- Start by making evaluation distinguish “technically passed” from “scientifically promising”:
  - repaired passes should remain valid but carry explicit caution in comparison/reporting
  - warning-only outcomes should be visible as budget-limited rather than silently comparable to clean passes
  - repeated repairs or unstable runtime signals should become evaluation penalties or reviewer-facing flags
- Keep the first Phase 8 slice narrow:
  - define a standard `EvaluationResult` schema
  - aggregate current runtime/evaluation artifacts into that schema
  - add a small number of extra signals beyond loss, such as stability flags, repair counts, and basic runtime practicality signals
  - update reports/comparators to explain whether an idea should continue, not just whether it executed
- Reuse existing `DesignIR` and runtime structure when expanding evaluation:
  - evaluation tasks already exist in `DesignIR`
  - phase-local evaluation scopes already exist in runtime config
  - repair classification now gives an explicit reliability signal that should flow into ranking and promotion logic

What should be implemented:

- Task-aware evaluation, ablations, and experiment comparison.

Why:

- A single loss comparison is not enough to decide whether an architecture is worth promoting.
- The system must evaluate both quality and practicality.

How:

- Expand evaluation to include:
  - train and validation loss curves
  - throughput and memory use
  - stability signals
  - benchmark-specific metrics
  - qualitative sample outputs
  - ablation comparisons
  - regression detection against prior attempts
- Define a standard `EvaluationResult` schema.
- Allow each idea to declare an evaluation matrix in `DesignIR`.
- Update reports to explain not only metrics, but why an idea should continue or stop.

Suggested deliverables:

- `src/auto_llm_innovator/evaluation/models.py`
- `src/auto_llm_innovator/evaluation/runner.py`
- `src/auto_llm_innovator/evaluation/ablation.py`
- richer reports in `tracking/reports.py`

### 9. Add Baseline Management And Experiment Ranking

Implementation overview:

- Added `src/auto_llm_innovator/evaluation/baselines.py` with normalized baseline models and loaders so the repo can consume either:
  - the older flat `reference_metrics` manifest shape
  - or a richer structured manifest with baseline family, label, metric targets, and optional reliability/practicality/hardware/token-budget assumptions
- Upgraded the default internal baseline manifest to the new structured shape while preserving compatibility fallback for older fixtures and tests.
- Updated the evaluation path so baseline-aware code now works against normalized baseline definitions instead of assuming raw `reference_metrics` JSON everywhere.
- Added `src/auto_llm_innovator/tracking/ranking.py` with deterministic same-idea attempt ranking built on top of Phase 8’s persisted `EvaluationResult`.
- Implemented the first ranking slice around existing Phase 8 signals:
  - clean promotable attempts rank above repaired promotable attempts
  - repaired promotable attempts rank above warning-only rerun candidates
  - warning-only rerun candidates rank above failed stop outcomes
  - stop-recommended attempts remain visible in lineage, but are treated as noncompetitive
  - loss deltas only affect rank after reliability ordering is established
- Added prior-attempt comparison for the same idea by reading persisted attempt evaluation artifacts and summarizing whether the latest attempt is:
  - best so far
  - tied with the current best
  - regressed relative to a prior attempt
- Updated `InnovatorEngine.run()` to persist `reports/<attempt>-ranking.json` after Phase 8 evaluation, and updated `compare()` so it now returns:
  - the existing baseline comparison payload
  - ranking metadata
  - prior-attempt comparison summary
- Expanded decision reporting so reports now include ranking context, best-so-far status, strengths, and limiting factors in addition to technical and evaluation summaries.
- Added focused tests for baseline normalization, structured-manifest loading, ranking order across reliability classes, prior-attempt comparison, and orchestration/report persistence.

Status:

- Phase 9 foundation is now implemented as a baseline-normalization and same-idea attempt-ranking layer on top of Phase 8’s `EvaluationResult` boundary.
- The repo can now compare attempts using reliability-aware evaluation semantics, not just flat baseline deltas.
- The current system is intentionally narrow:
  - ranking is still per-idea, not cross-idea
  - mechanism-family comparison is not implemented yet
  - resource-aware admission and scaling decisions are still future work

How Phase 9 should start, given the implemented Phase 8 boundary:

- Treat `reports/<attempt>-evaluation.json` as the new comparison boundary, not just `PhaseResult` plus baseline manifest:
  - Phase 8 already aggregates loss deltas, repair lineage, caution flags, stop reasons, and overall continue/stop recommendations
  - ranking should build on those normalized evaluation summaries instead of recomputing phase meaning from raw artifacts
- Keep baseline management downstream of Phase 8 aggregation:
  - let `execute_phase()` keep owning execution and repair
  - let Phase 8 keep owning per-attempt evaluation semantics
  - let Phase 9 decide how to compare one evaluated attempt against baselines, prior attempts, and mechanism families
- Start by replacing the single static baseline view with structured baseline definitions that align with `EvaluationResult`:
  - baseline family and label
  - phase-specific target metrics
  - expected reliability or practicality envelopes
  - optional hardware or token-budget assumptions
- Keep the first Phase 9 slice narrow:
  - compare the latest `EvaluationResult` against the configured baseline
  - compare the latest attempt against prior attempts for the same idea
  - derive a simple deterministic rank or promotion score from existing evaluation recommendations, caution flags, and baseline deltas
  - avoid adding cross-idea leaderboard complexity until the per-idea ranking path is stable
- Reuse the new Phase 8 signals directly when ranking:
  - repaired passes should rank below clean passes, even when loss is similar
  - warning-only budget-limited outcomes should not outrank clean promotable runs
  - failed or stop-recommended attempts should remain visible in lineage, but should not be treated as competitive results
  - planned-but-not-run ablations should remain metadata only until Phase 8 grows real ablation execution

What should be implemented:

- A richer baseline and leaderboard subsystem.

Why:

- The repo should compare generated ideas not only to a static internal baseline, but also to previous attempts and known architecture families.
- Promotion decisions need ranking context.

How:

- Extend `baselines/` from a simple manifest into structured baseline definitions.
- Track:
  - baseline family
  - metric expectations by phase
  - hardware assumptions
  - training token budget
- Add ranking logic that compares:
  - current phase result vs baseline
  - current attempt vs prior attempts for same idea
  - same mechanism family across ideas

Suggested deliverables:

- `src/auto_llm_innovator/evaluation/baselines.py`
- `src/auto_llm_innovator/tracking/ranking.py`
- `tests/test_baselines.py`

### 10. Add Resource And Hardware Awareness

Implementation overview:

- Expanded `src/auto_llm_innovator/env.py` so `EnvironmentReport` now captures accelerator backend, GPU names, VRAM per device, CPU count, system RAM, free disk, platform metadata, and torch version while still degrading cleanly when PyTorch or accelerators are unavailable.
- Added `src/auto_llm_innovator/planning/` with deterministic Phase 10 planning models and admission helpers:
  - `planning/resources.py` builds `PhaseResourceRequest`, `PhaseResourcePlan`, and ordered `ResourceAdjustment` entries
  - `planning/admission.py` turns those plans into resolved phase configs and persists `resource-plan.json`
- Integrated resource admission into `InnovatorEngine.run()` before `execute_phase()`:
  - each phase now loads `environment.json`, the structured baseline definition, and the latest same-idea ranking/evaluation context when available
  - the engine persists `runs/<attempt>/<phase>/resource-plan.json`
  - the engine writes `runs/<attempt>/<phase>/resolved-config.json` and uses that resolved config for preflight/runtime execution
- Implemented bounded deterministic admission outcomes:
  - `admit` runs the phase with planned settings
  - `downscale` reduces settings in this order: batch size, sequence length, max steps, then target parameters
  - `reject` fails before preflight with `stop_reason == "resource_admission_failed"` and recommendation `adjust_resources_or_stop`
- Kept Phase 8 and Phase 9 as the downstream consumers of the result:
  - runtime summaries now carry the applied `resource_plan`
  - evaluation surfaces resource-downscaled outcomes as practicality/caution context
  - ranking semantics were left unchanged and now consume the improved run context naturally
- Added `tests/test_resource_planning.py` covering GPU admission, constrained downscaling, CPU rejection, advisory baseline hardware warnings, ranking-aware escalation/downscaling, artifact persistence, early rejection before preflight, and resolved-config runtime compatibility.

Status:

- Phase 10 foundation is now implemented as a resource-aware admission and downscaling layer between Phase 9 ranking/baseline context and the existing Phase 6 runtime config boundary.
- The repo can now make deterministic per-phase feasibility decisions before preflight/runtime spend instead of blindly attempting every configured phase.
- The current system is intentionally narrow:
  - admission is still phase-local and same-idea only
  - hardware assumptions from baselines are advisory rather than strict schedulable requirements
  - there is no global queueing, reservation, or cross-idea resource arbitration yet

How Phase 10 should start, given the implemented Phase 9 boundary:

- Treat structured baseline definitions and attempt-ranking outputs as the new planning context for resource decisions, not just static environment facts:
  - baselines now describe expected metric targets and optional practicality or hardware assumptions
  - ranking now distinguishes clean promotable runs from repaired, warning-only, and noncompetitive outcomes
- Keep hardware awareness upstream of execution but downstream of the current planning/evaluation structure:
  - let `DesignIR` continue to describe architecture and phase intent
  - let Phase 8 continue to judge outcomes
  - let Phase 9 continue to rank attempts
  - let Phase 10 decide whether an idea or phase should be scaled, constrained, retried at lower budget, or rejected before expensive execution
- Start by mapping resource planning onto the existing phase config and runtime settings instead of inventing a second config surface:
  - adjust target parameters, sequence length, batch size, checkpoint cadence, and resume expectations through the current phase/runtime config path
  - use baseline hardware assumptions as advisory targets when deciding whether a run is realistically comparable
- Keep the first Phase 10 slice narrow:
  - extend environment probing with real hardware capacity fields
  - add a resource planner that derives feasible runtime settings for each phase from `DesignIR`, current environment, and structured baseline assumptions
  - add admission checks that fail early or downscale settings when the requested phase is clearly infeasible
  - avoid dynamic online scheduling or cross-run queueing until simple admission/downscaling behavior is stable
- Reuse the new Phase 9 artifacts directly:
  - repeated low-ranking or budget-limited attempts should inform when to downscale or stop escalating compute
  - noncompetitive stop-ranked attempts should not automatically consume larger budgets in later phases
  - clean leading attempts are the strongest candidates for full-budget escalation

What should be implemented:

- Compute-aware planning and admission control.

Why:

- A complete innovator must respect real hardware constraints.
- Some ideas should be rejected or downscaled before generation if they cannot fit the environment.

How:

- Extend environment probing so it captures:
  - GPU type
  - VRAM
  - CUDA or ROCm availability
  - CPU and RAM
  - storage limits
- Add a planner that maps `DesignIR` and phase goals to available resources.
- Auto-adjust:
  - model width/depth
  - sequence length
  - batch size
  - grad accumulation
  - checkpoint frequency
- Reject impossible experiments early.

Suggested deliverables:

- `src/auto_llm_innovator/planning/resources.py`
- `src/auto_llm_innovator/planning/admission.py`
- `tests/test_resource_planning.py`

### 11. Add Dataset And Curriculum Planning

Implementation overview:

- Added `src/auto_llm_innovator/datasets/models.py` with structured dataset metadata and planning artifacts:
  - `DatasetDefinition`
  - `DatasetPhasePreset`
  - `DatasetPlan`
- Refactored `src/auto_llm_innovator/datasets/registry.py` so the repo now has a structured registry-backed source of truth for the current three dataset families:
  - `synthetic-shapes`
  - `small-curated-corpus`
  - `production-like-corpus`
- Kept `dataset_plan_for_phase()` as a compatibility helper, but it now reads from the structured registry instead of a hardcoded flat dict.
- Added `src/auto_llm_innovator/datasets/planner.py` with a deterministic `plan_dataset_for_phase(...)` boundary that consumes:
  - `DesignIR.training_plan`
  - the current phase config’s dataset hint
  - the Phase 10 `PhaseResourcePlan`
  - the resolved runtime settings
  - optional baseline token-budget assumptions
- Integrated dataset planning into the same resolved-config path as Phase 10:
  - `InnovatorEngine.run()` now plans datasets after resource admission/downscaling and before `execute_phase()`
  - each phase now persists `runs/<attempt>/<phase>/dataset-plan.json`
  - `resolved-config.json` now carries the final runtime-compatible dataset projection plus optional `dataset_plan` metadata
- Kept the shared runtime contract compatible:
  - runtime still consumes the existing `dataset` block shape
  - `runtime/config.py` now carries optional `dataset_plan` metadata for summaries and downstream reporting
- Reused the structured registry at submit-time and `DesignIR` compile-time via the compatibility helper, so phase configs and `DesignIR.training_plan` stay aligned to the same dataset defaults.
- Added `tests/test_dataset_planning.py` covering registry defaults, `DesignIR` alignment, admitted/downscaled/rejected dataset plans, runtime-config compatibility, and engine artifact persistence.

Status:

- Phase 11 foundation is now implemented as a deterministic dataset-planning layer between Phase 10 resource admission and the existing runtime execution boundary.
- The repo can now persist explicit per-phase dataset plans instead of relying only on a flat phase-level dataset hint.
- The current system is intentionally narrow:
  - dataset planning is still metadata- and config-oriented rather than a real materialization/download pipeline
  - curriculum behavior is still represented as notes and token-budget/slice selection, not multi-stage execution state
  - cross-phase shared dataset caching and dataset lineage are still future work

How Phase 11 should start, given the implemented Phase 10 boundary:

- Treat Phase 10 resource planning as the new feasibility boundary for dataset decisions, not just runtime sizing:
  - `resource-plan.json` now tells later phases whether a run was admitted, downscaled, or rejected
  - resolved runtime settings now capture the actual sequence length, batch size, and step budget the dataset plan must fit
- Keep dataset planning upstream of execution but downstream of the current planning stack:
  - let `DesignIR` continue to describe architecture, training stages, and evaluation intent
  - let Phase 10 continue to decide feasible runtime/resource envelopes
  - let Phase 11 decide which datasets, slices, curricula, and token budgets fit inside that admitted envelope
- Start by extending the current phase config and dataset hint path instead of inventing a separate data runtime:
  - use the existing `dataset` block in `config/<phase>.json`
  - make dataset planning produce a structured per-phase materialization plan that Phase 10-resolved runtime settings can actually sustain
  - avoid changing generated package code or the shared runtime loop in the first slice
- Reuse the new Phase 10 artifacts directly:
  - a downscaled phase should usually receive a correspondingly smaller or simpler dataset slice
  - a rejected high-budget phase should not trigger expensive dataset preparation work
  - cleanly admitted leading attempts are the best candidates for richer curriculum stages or broader eval corpora
- Keep the first Phase 11 slice narrow:
  - expand dataset registry metadata
  - add a planner that maps `DesignIR` training stages plus resolved resource settings into per-phase dataset plans
  - persist those plans as artifacts the runtime can consume later
  - avoid full curriculum execution logic until the plan schema is stable

What should be implemented:

- A proper data planning layer, not just phase-level dataset hints.

Why:

- Many ideas depend on curriculum design, synthetic augmentation, or specific eval corpora.
- Data selection is often as important as architecture selection.

How:

- Expand `datasets/registry.py` into:
  - dataset capability metadata
  - splits
  - token budgets
  - license or availability flags
  - compatibility with training objectives
- Let `DesignIR` request data stages.
- Build dataset materialization plans per phase.

Suggested deliverables:

- `src/auto_llm_innovator/datasets/models.py`
- `src/auto_llm_innovator/datasets/planner.py`
- `tests/test_dataset_planning.py`

### 12. Add Better Experiment Lineage And Reproducibility

Implementation overview:

- Added `src/auto_llm_innovator/tracking/lineage.py` with deterministic SHA-256 helpers for:
  - hashing individual files
  - hashing directories as stable file maps
  - hashing canonical JSON payloads
  - collecting best-effort artifact records that tolerate missing files
- Added `src/auto_llm_innovator/tracking/manifests.py` with a typed `PhaseLineageManifest` plus manifest builders/persistence for per-phase lineage snapshots.
- Integrated lineage assembly into `InnovatorEngine.run()` after Phase 10 resource planning, Phase 11 dataset planning, and prompt/skill artifact persistence:
  - each phase now writes `runs/<attempt>/<phase>/lineage-manifest.json`
  - the manifest is appended to `PhaseResult.artifacts_produced` before status/ledger recording
- Kept Phase 12 downstream of planning and execution:
  - manifests hash `resolved-config.json`, `resource-plan.json`, and `dataset-plan.json`
  - generation lineage is derived from the existing `generation_manifest.json`
  - repair lineage reuses the existing repair snapshots, diffs, rationale files, and failure classification artifacts
- Added test coverage in `tests/test_lineage.py` plus orchestration/repair assertions for:
  - stable hashing
  - executed phase manifests
  - rejected phase manifests
  - repaired phase manifests
  - missing generated files recorded without crashing

Status:

- Phase 12 foundation is now implemented as a local immutable lineage layer on top of the existing artifact tree.
- The repo can now emit deterministic per-phase manifests tying together planning artifacts, generated package state, repair history, environment details, and final phase results.
- The current system is intentionally narrow:
  - lineage is local-only and file-based
  - seeds are captured only from resolved runtime config fields that already exist
  - there is not yet a richer cross-role agent transcript or typed agent-response lineage
  - reports consume the manifest only indirectly through `artifacts_produced`

How Phase 12 should start, given the implemented Phase 11 boundary:

- Treat `resolved-config.json`, `resource-plan.json`, and `dataset-plan.json` as the new reproducibility surface, not just the original submit-time config:
  - Phase 10 now changes runtime/resource settings per attempt and phase
  - Phase 11 now changes dataset scope and token budget per attempt and phase
  - Phase 12 should capture the exact resolved state that execution actually used
- Keep lineage downstream of planning and execution, not inside generated package code:
  - let `DesignIR` keep describing intent
  - let Phase 10 and Phase 11 keep resolving runtime and dataset plans
  - let Phase 12 snapshot those resolved artifacts, generated package contents, repair diffs, and environment details into immutable lineage records
- Start from the run directory boundary that already exists:
  - each phase now has stable artifacts worth hashing and versioning before runtime spend
  - each attempt already has reports, ranking, evaluation, repair, resource, and dataset artifacts that should be tied together in a manifest
- Keep the first Phase 12 slice narrow:
  - add immutable content hashes for generated package files, resolved config, resource plan, and dataset plan
  - persist a run manifest per attempt/phase with seeds, environment summary, and artifact hashes
  - avoid adding cross-machine storage or external artifact backends until the local manifest schema is stable
- Reuse the new Phase 11 artifacts directly:
  - if a phase was downscaled, the lineage should show the exact dataset and runtime reductions that were applied
  - if a phase was rejected before execution, the manifest should still capture the rejected resource and dataset plans
  - repaired runs should point to the exact pre-repair and post-repair snapshots already emitted by the repair system

What should be implemented:

- Stronger versioning and artifact lineage for generated code and run state.

Why:

- Once the system begins modifying generated code across retries, reproducibility becomes critical.
- Every report should be traceable back to the exact generated files and config that produced it.

How:

- Store immutable snapshots or content hashes for:
  - generated package
  - training config
  - evaluation config
  - dataset plan
  - repair diffs
- Add run manifests per attempt.
- Include seed values and environment metadata everywhere.

Suggested deliverables:

- `src/auto_llm_innovator/tracking/lineage.py`
- `src/auto_llm_innovator/tracking/manifests.py`
- `tests/test_lineage.py`

### 13. Add A Better Agentic Runtime Interface

Implementation overview:

- Added `src/auto_llm_innovator/orchestration/agent_runtime.py` with a narrow typed orchestration contract for the first advisory slice:
  - `AgentContextArtifactRef`
  - `AgentRequestEnvelope`
  - `AgentInvocationRecord`
  - `PlannerResponse`
  - `ReviewerResponse`
- Added role-specific request building, payload validation, and artifact persistence for `planner` and `reviewer`:
  - per-phase agent artifacts now live under `runs/<attempt>/<phase>/agents/`
  - each role now records `*-request.json`, `*-response.json`, and `*-runtime.json`
- Upgraded `src/auto_llm_innovator/orchestration/opencode.py` with a best-effort `invoke_structured(...)` path that:
  - composes strict JSON-output instructions
  - records dry-run status when OpenCode is unavailable
  - captures stdout/stderr
  - parses fenced or plain JSON object payloads
  - reports runtime and parse failures without blocking the phase
- Integrated structured planner/reviewer artifacts into `InnovatorEngine.run()`:
  - planner requests are built after Phase 10/11 planning and before `execute_phase()`
  - reviewer requests are built after `lineage-manifest.json` is written, so they consume the immutable Phase 12 record
  - agent artifact paths are appended to `PhaseResult.artifacts_produced`
- Kept the first Phase 13 slice intentionally narrow:
  - only `planner` and `reviewer` are typed
  - outputs are advisory only
  - `lineage-manifest.json` remains immutable and is referenced by agent artifacts rather than rewritten
- Added coverage for the new boundary:
  - `tests/test_opencode.py` for structured invocation parsing and failure handling
  - orchestration tests for planner/reviewer request and response persistence
  - repair/orchestration assertions that agent artifacts are present in run outputs
  - a deterministic test harness in `tests/conftest.py` so local OpenCode installation does not make tests flaky

Status:

- Phase 13 foundation is now implemented as an advisory typed agent-runtime layer around the existing execution flow.
- The repo can now persist structured planner/reviewer inputs and outputs as reproducible per-phase artifacts grounded in Phase 12 lineage.
- The current system is intentionally narrow:
  - only planner and reviewer are typed
  - invalid structured outputs do not gate phase progression
  - generated package code and `execute_phase()` remain unchanged
  - multi-role agent coordination, stateful conversations, and stronger gating policies are still future work

How Phase 13 should start, given the implemented Phase 12 boundary:

- Treat `lineage-manifest.json` as the new grounding artifact for agentic orchestration work, not just prompts and previews:
  - Phase 12 now gives each phase a deterministic record of the resolved runtime plan, dataset plan, generated package state, repair lineage, and outputs
  - Phase 13 should make agent roles consume and emit artifacts that can be attached to that lineage surface
- Keep the first Phase 13 slice downstream of existing orchestration rather than replacing `execute_phase()` or generated package code:
  - let Phase 10 and Phase 11 keep resolving feasible runtime and dataset plans
  - let Phase 12 keep snapshotting what execution actually used
  - let Phase 13 add typed agent payloads, validators, and persisted role outputs around the existing orchestration flow
- Start from the prompt/skill artifact boundary that already exists in each phase directory:
  - each phase already persists `prompt.json`, `skills.json`, and the final lineage manifest
  - Phase 13 should add typed response artifacts beside those files so planner/implementer/trainer/evaluator/repairer outputs become reproducible inputs to later steps
- Reuse the new Phase 12 artifacts directly:
  - planner and reviewer agents should read the prior phase lineage manifest instead of reconstructing context from scattered files
  - repair-oriented roles should point to concrete repair diffs and before/after snapshots already captured in lineage
  - evaluator/reviewer roles should reference exact resolved config and dataset decisions recorded in lineage when explaining outcomes
- Keep the first Phase 13 slice narrow:
  - define typed response schemas for one or two high-leverage roles first, likely planner and reviewer
  - validate and persist those role outputs per phase
  - avoid full conversational multi-agent state machines until the typed artifact boundary is stable

What should be implemented:

- A stronger integration between the orchestration layer and the underlying agent runtime.

Why:

- Today OpenCode integration is mainly preview-oriented inside the innovator.
- A complete system should use agents for planning, generation, repair, and evaluation commentary with clearer contracts and richer artifacts.

How:

- Define explicit agent roles and structured outputs for:
  - planner
  - architect
  - implementer
  - trainer
  - evaluator
  - repairer
  - reviewer
- Make each role return typed JSON payloads rather than only free-text prompts.
- Persist each role's structured output alongside prompts.

Suggested deliverables:

- stronger prompt payload schemas
- role-specific response validators
- updated orchestration artifacts

### 14. Expand Tests To Match The Intended Product

How Phase 14 should start, given the implemented Phase 13 boundary:

- Treat the new advisory agent artifacts as part of the product surface that must be tested, not just the execution/runtime files:
  - each phase now emits prompt artifacts, skill-routing artifacts, lineage artifacts, and typed planner/reviewer artifacts
  - Phase 14 should validate how those layers interact across success, rejection, repair, and comparison flows
- Keep the first Phase 14 slice centered on deterministic end-to-end coverage rather than broad snapshot volume:
  - start from the in-process orchestration and adapter tests that now exist
  - expand them into stable lifecycle coverage for both execution artifacts and agent artifacts together
- Reuse the new Phase 13 boundaries directly:
  - assert that planner requests include the resolved planning artifacts and prior lineage when available
  - assert that reviewer requests consume the immutable `lineage-manifest.json` rather than reconstructing context
  - assert that invalid structured outputs degrade gracefully without blocking phases
  - assert that repaired and rejected runs still emit the expected agent-runtime artifacts
- Start by hardening the deterministic test harness before adding broader integration volume:
  - keep OpenCode mocked or disabled by default in tests unless a test is specifically exercising the adapter
  - keep artifact assertions path- and schema-oriented instead of overfitting to incidental text content
- Keep the first Phase 14 slice narrow:
  - expand the current orchestration, repair, CLI, and lifecycle suites to cover the new agent artifact boundary thoroughly
  - avoid large golden snapshot suites for all generated code until the new artifact schemas settle

What should be implemented:

- A much broader test suite covering generation, runtime, preflight, repair, and evaluation.

Why:

- The current tests are good for scaffold behavior, but they do not yet prove autonomous experiment execution.
- As the codebase gets more agentic and dynamic, regression risk will grow quickly.

How:

- Keep unit tests for normalization and tracking.
- Add integration tests for:
  - structured handoff intake
  - IR generation
  - package generation
  - preflight pass/fail cases
  - smoke run lifecycle
  - repair loop behavior
  - evaluation report generation
- Add golden artifact tests where generated file layouts are compared against expected snapshots.

Suggested deliverables:

- `tests/test_handoff.py`
- `tests/test_design_ir.py`
- `tests/test_generation.py`
- `tests/test_runtime.py`
- `tests/test_preflight.py`
- `tests/test_repair.py`
- `tests/test_evaluation.py`

Implementation overview:

- Added a stronger deterministic test harness in `tests/conftest.py` for the new agent-runtime surface:
  - baseline bootstrap helpers
  - seeded environment helpers
  - reusable structured planner/reviewer stubs
  - shared assertions for `agents/*-{request,response,runtime}.json` persistence
- Expanded orchestration coverage to lock the Phase 13 artifact boundary:
  - planner requests now have tests for required planning artifacts, prior-phase lineage, and prior-attempt lineage
  - reviewer requests now have tests for immutable lineage consumption and prior-attempt ranking/evaluation context when available
  - advisory parse-status handling is now covered for `dry_run`, `invalid_json`, `invalid_schema`, and `runtime_failed`
- Added a focused lifecycle suite in `tests/test_lifecycle.py` with four deterministic end-to-end scenarios:
  - successful smoke run
  - rejected full run before execution
  - repaired smoke run
  - second-attempt rerun with prior comparison context
- Extended CLI lifecycle coverage so `submit -> run -> status -> compare -> report` verifies the on-disk artifact contract, not only stdout payload shape
- Kept the slice intentionally narrow:
  - no production/runtime behavior changes
  - no new typed roles beyond planner/reviewer
  - no large golden snapshot suite for generated code

Status:

- Phase 14 now locks the current Phase 13 product surface with deterministic tests across success, rejection, repair, and rerun/comparison flows.
- The repo now has stable coverage for prompt, skills, lineage, and planner/reviewer agent artifacts as one lifecycle boundary rather than isolated files.
- The current system is still intentionally narrow:
  - planner and reviewer remain the only typed roles
  - advisory agent failures still do not gate phase progression
  - broader generated-code snapshot coverage and multi-role orchestration are still future work


## Recommended Architecture Direction

The implementation should move the repository toward this flow:

```text
auto-llm-researcher candidate
  -> structured handoff bundle
  -> IdeaSpec normalization
  -> DesignIR compiler
  -> DesignIR validation
  -> multi-file package generation
  -> preflight validation
  -> smoke run
  -> repair loop if needed
  -> small run
  -> repair loop if needed
  -> full run
  -> evaluation and ablations
  -> baseline comparison
  -> decision report
```

## Suggested Milestones

### Milestone 1: Make It A Real Experiment Harness

Ship:

- structured handoff
- design IR
- reusable runtime
- multi-file generation

Success criteria:

- an idea can generate a real importable package and instantiate a model

### Milestone 2: Make It Reliable

Ship:

- preflight validation
- smoke-phase semantics
- stronger tests

Success criteria:

- broken generations fail fast with actionable diagnostics

### Milestone 3: Make It Autonomous

Ship:

- repair loop
- automatic retries
- richer agent outputs

Success criteria:

- common codegen failures can be fixed automatically without human intervention

### Milestone 4: Make It Scientifically Useful

Ship:

- evaluation suite
- ablations
- baseline ranking
- promotion decisions

Success criteria:

- the system can justify why an idea should continue, stop, or be revised

## Non-Goals

The following should not be prioritized early:

- full distributed training infrastructure
- support for every model family
- highly optimized production serving
- broad multimodal research support

The first goal is to be excellent at small-LLM autonomous experimentation under the repository's current constraints.

## Final Guidance

The repository should not try to generate everything from scratch for every idea.

The strongest design is:

- shared runtime primitives
- structured design IR
- narrow generated custom modules
- strong preflight checks
- bounded automated repair
- rich tracking and comparison

That path will make `auto-llm-innovator` much more reliable, easier to evolve, and far closer to the intended premise of a true autonomous LLM experiment builder.
