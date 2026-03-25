---
name: phase-gate-rationale
description: Use after smoke, small, or full runs to convert metrics, artifacts, and logs into an explicit continue, pivot, or stop recommendation.
---

# Phase Gate Rationale

## Workflow

1. Read the latest metrics, error signals, baseline comparison, and originality status.
2. Decide one action:
   - continue
   - pivot and retry
   - stop
3. Justify the decision with evidence from this phase, not generic optimism.

## Required output

- `decision`
- `evidence`
- `next action`
- `open risks`

## Guardrails

- Every phase needs an explicit next action.
- Do not advance on vibes alone.
