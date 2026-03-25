---
name: eval-suite-builder
description: Use when an idea needs an evaluation suite with metrics, task families, regression checks, and qualitative judge prompts beyond the default reports.
---

# Eval Suite Builder

## Workflow

1. Start from general-purpose LM evaluation defaults.
2. Add idea-specific checks only when they are justified by the architecture or objective.
3. Separate automatic metrics from qualitative judge prompts.

## Required output

- `evaluation suite`
- `automatic metrics`
- `qualitative prompts`

## Guardrails

- Keep the suite aligned with the idea’s intended model target.
- Avoid turning one novelty feature into the only success criterion.
