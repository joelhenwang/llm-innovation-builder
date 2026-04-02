# auto-llm-innovator

`auto-llm-innovator` is a CLI-first framework for turning a language-model architecture idea into a runnable experiment bundle. It accepts either a free-text brief or a structured researcher artifact, compiles that idea into a richer internal plan, generates a multi-file package, validates it before spending budget, runs staged experiments, attempts bounded repairs for certain failures, and writes evaluation, ranking, and decision artifacts.

This repository is already more than a scaffold. The current implementation includes:

- structured handoff intake from upstream research output
- `DesignIR` as the internal architecture and experiment plan
- deterministic multi-file code generation under `ideas/<idea_id>/package/`
- shared runtime semantics for `smoke`, `small`, and `full`
- preflight validation before runtime spend
- bounded deterministic repair loops
- baseline comparison, same-idea ranking, and decision reports
- resource planning, dataset planning, and lineage manifests

## Who This Is For

This README is meant to work for two audiences in one pass:

- users who want to submit an idea, run it, and understand the resulting artifacts
- contributors who want to understand the framework’s subsystem boundaries and where to change behavior

If you are new to the project, start with the mental model and guided first run. If you want to extend the framework, jump to the framework components and contributor sections after that.

## Mental Model

The easiest way to think about `auto-llm-innovator` is as a pipeline that progressively makes an idea more concrete and more expensive:

1. Intake:
   a free-text brief or structured bundle is normalized into a handoff artifact.
2. Planning:
   the handoff is compiled into `DesignIR`, which makes architecture, training, and evaluation intent explicit.
3. Generation:
   the framework generates a multi-file Python package plus compatibility entrypoints.
4. Preflight:
   generated code is validated before runtime budget is spent.
5. Execution:
   the idea is run through `smoke`, `small`, and optionally `full`.
6. Repair:
   certain failed outcomes are classified and repaired with bounded retries.
7. Evaluation:
   the framework summarizes what happened, compares against baselines and prior attempts, and emits reports.
8. Tracking:
   artifacts, lineage, status, and reports are persisted so later runs can build on earlier context.

At a high level, the flow is:

`submit` -> `handoff_bundle.json` -> `design_ir.json` -> generated `package/` -> preflight -> `smoke`/`small`/`full` -> repair when applicable -> evaluation/ranking -> reports and lineage

## Guided First Run

### 1. Install and set up the project

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

### 2. Submit an idea

Use a free-text brief when you want the framework to normalize a concept directly:

```bash
innovator submit "Invent a recurrently modulated attention language model that is not a plain transformer."
```

Use a structured bundle when the idea already comes from an upstream researcher pipeline:

```bash
innovator submit --bundle-file /path/to/researcher-candidate.json
```

Submission creates an idea directory under `ideas/<idea_id>/` and writes the initial planning artifacts, generated package, phase configs, and notes.

### 3. Run the idea

Run a single phase if you want a controlled checkpoint:

```bash
innovator run idea-0001 --phase smoke
```

Run the full promoted lifecycle if you want the framework to attempt `smoke`, then `small`, then `full` when each earlier stage passes cleanly:

```bash
innovator run idea-0001 --phase all
```

Resume the latest incomplete attempt if a prior run stopped before completion:

```bash
innovator resume idea-0001
```

### 4. Inspect the outcome

Use `status` when you want operational state and artifact locations as JSON:

```bash
innovator status idea-0001
```

Use `compare` when you want structured comparison against the configured baseline and prior attempts:

```bash
innovator compare idea-0001
```

Use `report` when you want the latest human-readable decision summary:

```bash
innovator report idea-0001
```

### 5. Inspect skill routing when debugging orchestration

```bash
innovator skills list
innovator skills doctor
innovator skills explain reviewer --phase small
innovator skills explain implementer --phase small --prompt-view
innovator skills sync
```

These commands are mainly useful when you want to inspect which reviewed skills are injected into agent roles and how orchestration context is built.

## What Gets Generated And What To Inspect

The framework writes a lot of useful state. The easiest way to understand a run is to inspect artifacts in the order they are created.

### Submission artifacts

- `ideas/<idea_id>/handoff_bundle.json`
  the normalized structured intake artifact; inspect this first to confirm what the framework believes the idea is.
- `ideas/<idea_id>/design_ir.json`
  the richer internal plan for architecture, modules, training stages, evaluation tasks, and constraints.
- `ideas/<idea_id>/idea_spec.json`
  the compatibility projection used by existing downstream orchestration paths.
- `ideas/<idea_id>/originality_review.json`
  the originality gate result and required revisions if the idea is blocked.
- `ideas/<idea_id>/notes/design.md`
  a short human-oriented summary of the brief, hypothesis, novelty claims, and originality result.
- `ideas/<idea_id>/notes/orchestration.md`
  a short orchestration summary describing runtime and agent assumptions.

### Generation artifacts

- `ideas/<idea_id>/generation_manifest.json`
  the generated file inventory plus summarized `DesignIR` module and evaluation task names.
- `ideas/<idea_id>/package/`
  the generated multi-file package containing the idea-specific plugin, model code, evaluation hooks, and package-local tests.
- `ideas/<idea_id>/train.py`
  the compatibility entrypoint into the generated package for runtime execution.
- `ideas/<idea_id>/eval.py`
  the compatibility evaluation entrypoint into the generated package.

### Phase planning artifacts

- `ideas/<idea_id>/config/<phase>.json`
  the default phase-local config created at submit time.
- `ideas/<idea_id>/runs/<attempt>/<phase>/resource-plan.json`
  the admission or downscaling decision based on environment, baseline context, and prior attempt context.
- `ideas/<idea_id>/runs/<attempt>/<phase>/dataset-plan.json`
  the selected dataset plan for that phase after resource planning.
- `ideas/<idea_id>/runs/<attempt>/<phase>/resolved-config.json`
  the final config that actually drives the phase after planning adjustments.

### Validation and execution artifacts

- `ideas/<idea_id>/runs/<attempt>/<phase>/preflight-report.json`
  the validation report written before runtime spend; inspect this first when a phase fails early.
- `ideas/<idea_id>/runs/<attempt>/<phase>/repair/`
  repair classifications, history, snapshots, diffs, and rationale when a bounded repair attempt happens.
- `ideas/<idea_id>/runs/<attempt>/<phase>/lineage-manifest.json`
  the best single place to inspect reproducibility and artifact lineage for that phase.

### Reporting artifacts

- `ideas/<idea_id>/status.json`
  the canonical operational record of attempts and latest report linkage.
- `ideas/<idea_id>/reports/<attempt>-evaluation.json`
  reliability-aware evaluation summary for one attempt.
- `ideas/<idea_id>/reports/<attempt>-ranking.json`
  ranking context against prior attempts for the same idea.
- `ideas/<idea_id>/reports/<attempt>.md`
  the human-readable decision report.

## Lifecycle And Phase Semantics

### Preflight comes before spend

Every phase is gated by preflight. This is where the framework checks that generated code can be imported, instantiated, exercised with synthetic inputs, and used through the shared runtime contract before the main phase runtime consumes budget.

In practice, preflight is where many ordinary failures are caught:

- broken imports
- invalid plugin contracts
- shape mismatches
- synthetic train-step failures
- checkpoint or evaluation hook wiring issues

If preflight fails, the framework can sometimes attempt a bounded deterministic repair and rerun preflight once. If the issue is not recoverable, the phase stops before the main runtime path starts.

### The three runtime phases mean different things

- `smoke`
  proves the package and runtime path work at tiny scale. It is the cheapest phase and the first place to confirm the idea is runnable at all.
- `small`
  runs a more meaningful but still constrained experiment. It enables resume, uses a larger dataset slice, and is meant to show whether the idea is worth deeper budget.
- `full`
  runs the largest configured budget with fuller evaluation scope and checkpoint behavior.

These differences are implemented in the shared runtime phase settings, not in ad hoc generated code per idea.

### Promotion is intentionally conservative

Automatic promotion follows:

`smoke` -> `small` -> `full`

The framework promotes only after a clean `passed` result from the previous phase. This means:

- preflight failures stop promotion
- runtime failures stop promotion
- `passed_with_warnings` outcomes also stop promotion

That behavior is deliberate. A technically incomplete or warning-limited result should not automatically consume more compute.

## CLI Reference

### Core commands

- `innovator submit "<brief>"`
  submit a free-text idea brief
- `innovator submit --bundle-file /path/to/bundle.json`
  submit a structured researcher bundle
- `innovator run <idea_id> --phase {smoke|small|full|all}`
  run a single phase or the full promoted lifecycle
- `innovator resume <idea_id>`
  resume the latest incomplete attempt
- `innovator status <idea_id>`
  print idea and attempt state as JSON
- `innovator compare <idea_id> [--baseline <baseline_id>]`
  compare the latest attempt against the configured baseline and prior attempts
- `innovator report <idea_id>`
  print the latest human-readable decision report

### Skill and routing commands

- `innovator skills list`
  list reviewed skills in the local registry
- `innovator skills doctor`
  validate the local skill registry and internal skill files
- `innovator skills explain <role> [--phase <phase>] [--prompt-view] [--idea-id <idea_id>]`
  inspect routed skills for one role, optionally as prompt payload data
- `innovator skills sync`
  materialize the reviewed external skill sync manifest

Supported roles for `skills explain` are:

- `planner`
- `implementer`
- `debugger`
- `trainer`
- `evaluator`
- `reviewer`

## Framework Components

For contributors, the codebase is easiest to navigate by subsystem responsibility.

### Handoff and design

- `src/auto_llm_innovator/handoff/`
  owns intake models, loaders, validation, and normalization from free text or researcher artifacts
- `src/auto_llm_innovator/design_ir/`
  owns the internal planning representation and validation rules that make architecture and experiment intent explicit

This is the right area to inspect when you want to change what information enters the system or how idea intent gets made more concrete before generation.

### Generation and runtime

- `src/auto_llm_innovator/generation/`
  owns package layout planning, renderer logic, source normalization, and generation manifests
- `src/auto_llm_innovator/runtime/`
  owns shared phase semantics, config compilation, train/eval loops, checkpointing, and runtime logging

This is the right area to inspect when you want to change the generated package shape or the shared execution contract that generated packages plug into.

### Validation and repair

- `src/auto_llm_innovator/validation/`
  owns preflight checks for imports, model construction, train-step sanity, and runtime contract compatibility
- `src/auto_llm_innovator/repair/`
  owns failure classification, repair modeling, persisted repair history, and deterministic repair application

This is the right area to inspect when you want to change what gets validated before runtime or how repairable failures are classified and retried.

### Planning and datasets

- `src/auto_llm_innovator/planning/`
  owns resource admission, downscaling, and resolved phase planning
- `src/auto_llm_innovator/datasets/`
  owns dataset metadata, compatibility helpers, and per-phase dataset planning

This is the right area to inspect when you want to change feasibility rules, hardware-aware scaling, or dataset selection behavior.

### Evaluation and tracking

- `src/auto_llm_innovator/evaluation/`
  owns evaluation aggregation, baseline definitions, comparison logic, and report inputs
- `src/auto_llm_innovator/tracking/`
  owns status ledgers, ranking, reports, lineage manifests, and related persisted metadata

This is the right area to inspect when you want to change how the framework decides whether a result is promising, how attempts are ranked, or how results are explained back to humans.

### Orchestration and skills

- `src/auto_llm_innovator/orchestration/`
  owns the top-level engine flow, agent context building, and integration between the other subsystems
- `orchestration/skills.json`
  stores reviewed skill routing policy
- `skills/internal/`
  contains internal skills used by the orchestrator

This is the right area to inspect when you want to change end-to-end sequencing, context assembly, or skill routing behavior.

## Where Contributors Should Make Changes

When you want to change framework behavior, start from the responsibility boundary rather than grepping randomly through the repo.

- Change intake rules or bundle validation:
  start in `handoff/`
- Change how structured ideas become explicit architecture/training plans:
  start in `design_ir/`
- Change generated package structure or emitted files:
  start in `generation/`
- Change shared phase budgets, runtime semantics, or execution flow:
  start in `runtime/` and the phase-planning path
- Change preflight checks or failure categories:
  start in `validation/` and `repair/`
- Change repair behavior or retry boundaries:
  start in `repair/`
- Change baseline comparison, evaluation, or ranking behavior:
  start in `evaluation/` and `tracking/`
- Change resource admission or dataset planning:
  start in `planning/` and `datasets/`
- Change top-level sequencing, artifact persistence, or orchestration context:
  start in `orchestration/engine.py`

In general:

- if the change affects what an idea means, it probably belongs in handoff or `DesignIR`
- if the change affects what files are emitted, it probably belongs in generation
- if the change affects how phases run, it probably belongs in runtime, planning, or validation
- if the change affects whether a run is considered good, bad, risky, or promotable, it probably belongs in evaluation or tracking

## How To Read Outputs

Different commands are meant for different readers and jobs.

- `status`
  best for operational inspection and automation; it is JSON-oriented and shows attempts, phase state, and latest report linkage
- `compare`
  best for structured analysis; it returns baseline comparison plus prior-attempt ranking context
- `report`
  best for human review; it renders the latest decision narrative in a readable Markdown-style summary

If you are debugging a failed run, the most useful order is usually:

1. `status`
2. `preflight-report.json` or phase artifacts under `runs/<attempt>/<phase>/`
3. `repair/` artifacts if a repair was attempted
4. `lineage-manifest.json`
5. `report`

## Repository Layout

- `ideas/<idea_id>/`: submission artifacts, generated package, configs, runs, and reports
- `baselines/`: baseline manifests used during evaluation and comparison
- `orchestration/skills.json`: reviewed skill registry and role-to-phase routing policy
- `skills/internal/`: internal orchestration skills
- `src/auto_llm_innovator/`: framework package and CLI entrypoint
- `tests/`: lifecycle, CLI, generation, runtime, planning, repair, evaluation, and lineage coverage
