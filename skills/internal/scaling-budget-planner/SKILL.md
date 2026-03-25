---
name: scaling-budget-planner
description: Use before expensive runs to estimate parameter, optimizer, memory, checkpoint, and token budgets under the single-node ROCm-first setup.
---

# Scaling Budget Planner

## Workflow

1. Estimate target parameter count and training-state footprint.
2. Translate that into batch, sequence, optimizer, and checkpoint tradeoffs.
3. Record requested versus affordable budget for the next phase.

## Required output

- `parameter estimate`
- `memory and checkpoint estimate`
- `recommended training budget`

## Guardrails

- Budgets may be soft, but they must be visible and recorded.
