---
name: smoke-test-math-and-shapes
description: Use when validating a new model with smoke tests for tensor shapes, forward logic, gradients, loss computation, initialization sanity, and tiny-batch overfit behavior.
---

# Smoke Test Math And Shapes

## Workflow

1. Verify the forward pass shape contract for embeddings, hidden states, logits, and loss.
2. Check mask logic, broadcasting, residual paths, and any recurrent or memory states.
3. Run a minimal backward pass and confirm gradients are finite.
4. Test checkpoint save/load and a tiny-batch overfit or equivalent sanity loop.
5. Report the first failing invariant, not a vague summary.

## Required output

- `shape checklist`
- `math sanity summary`
- `first failure and suspected root cause`

## Guardrails

- Do not pass smoke if tiny-batch learning or an equivalent sanity proof is missing.
- Prefer localizing one concrete break over speculating about many.
