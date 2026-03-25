---
name: checkpoint-resume-ops
description: Use when defining save cadence, resume policy, crash recovery, and lineage preservation for multi-step training runs and retries.
---

# Checkpoint Resume Ops

## Workflow

1. Define what gets saved, how often, and where.
2. State how partial progress resumes after interruption.
3. Preserve lineage between retries, resumes, and new attempts.

## Required output

- `checkpoint policy`
- `resume steps`
- `lineage notes`

## Guardrails

- Never reuse a checkpoint without recording its source attempt.
