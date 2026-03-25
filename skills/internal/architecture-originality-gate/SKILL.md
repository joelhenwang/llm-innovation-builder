---
name: architecture-originality-gate
description: Use when reviewing a proposed model architecture for originality, novelty claims, fallback-pattern violations, or similarity to common decoder-only templates.
---

# Architecture Originality Gate

## Use this skill when

- an idea is first normalized into an implementable architecture
- an implementer proposes borrowed mechanisms or references public models
- a reviewer must decide whether a phase can advance

## Workflow

1. Read the idea brief, novelty claims, forbidden fallback patterns, and any borrowed mechanisms.
2. List the concrete architectural commitments, not just marketing language.
3. Check for near-template behavior:
   - plain decoder-only stack with cosmetic changes
   - public model clone with renamed blocks
   - standard hyperparameter bundle with no new mechanism
4. Write a pass/fail review with:
   - novelty claims that appear real
   - suspected generic-transformer drift
   - required revisions before implementation or advancement

## Required output

- `originality verdict`: pass or fail
- `evidence`: short bullets tied to actual design choices
- `revisions`: explicit next changes if the verdict is fail

## Guardrails

- Reject generic transformer drift even when reference-only skills are available.
- Treat public inspirations as ingredients for recombination, not templates.
