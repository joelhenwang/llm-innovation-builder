---
name: rocm-training-debugger
description: Use when PyTorch ROCm runs fail due to device discovery, dtype issues, memory pressure, HIP kernels, compile/runtime mismatches, or unstable fallbacks.
---

# ROCm Training Debugger

## Workflow

1. Confirm device visibility, HIP version, dtype defaults, and accelerator assumptions.
2. Separate failure type:
   - import or environment issue
   - kernel or compilation issue
   - memory or fragmentation issue
   - numerical instability
3. Suggest the smallest safe change first: shape reduction, dtype change, kernel toggle, checkpoint cadence, or batch adjustment.
4. Record whether the change preserves ROCm compatibility or only offers a temporary CPU dry run.

## Required output

- `environment findings`
- `root cause`
- `recommended fix`
- `fallback status`

## Guardrails

- Do not silently switch frameworks.
- If falling back to CPU dry run, say that ROCm validation is still outstanding.
