---
name: dataset-curriculum-designer
description: Use when designing smoke, small, and production-like training curricula, token budgets, sequencing, or leakage checks for an idea.
---

# Dataset Curriculum Designer

## Workflow

1. Propose separate data plans for smoke, small, and full phases.
2. Include approximate token budgets, corpus purpose, and leakage checks.
3. Explain why the curriculum is appropriate for the architecture and objective.

## Required output

- `phase dataset plan`
- `token budget`
- `leakage checks`

## Guardrails

- Keep smoke data tiny and diagnostic.
- Do not reuse evaluation targets as training data without saying so.
