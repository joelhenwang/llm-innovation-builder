---
name: experiment-ledger-auditor
description: Use before retries or final reports to verify that lineage, artifacts, budgets, and outcomes are written to the experiment ledger.
---

# Experiment Ledger Auditor

## Workflow

1. Check that the attempt has:
   - attempt id and parent lineage
   - phase results
   - artifacts and report paths
   - budget records
2. Flag missing provenance before a retry or final decision.
3. Require write-before-continue discipline.

## Required output

- `ledger status`
- `missing records`
- `blockers to continuation`

## Guardrails

- Block advancement if lineage or artifacts are missing.
- Never overwrite prior attempt history.
