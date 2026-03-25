# Skill Decisions: idea-0003

- Registry source: `/Users/joelwang/Devs/auto-llm-innovator/orchestration/skills.json`
- Live marketplace installs allowed: `False`

## planner

- Always on: find-skills, ml-pipeline-orchestrator, architecture-research-synthesizer
- Optional: skill-creator, dataset-curriculum-designer
- smoke: none
- small: dataset-curriculum-designer
- full: dataset-curriculum-designer
- Forbidden: pytorch-lightning

## implementer

- Always on: deep-learning-python, architecture-originality-gate
- Optional: transformers
- smoke: none
- small: none
- full: none
- Forbidden: pytorch-lightning

## debugger

- Always on: systematic-debugging, rocm-training-debugger
- Optional: none
- smoke: smoke-test-math-and-shapes
- small: none
- full: none
- Forbidden: pytorch-lightning

## trainer

- Always on: deep-learning-python
- Optional: weights-and-biases, dataset-curriculum-designer, pytorch-lightning
- smoke: none
- small: scaling-budget-planner, checkpoint-resume-ops
- full: scaling-budget-planner, checkpoint-resume-ops
- Forbidden: none

## evaluator

- Always on: phase-gate-rationale
- Optional: eval-suite-builder
- smoke: none
- small: llm-evaluation, baseline-comparator
- full: llm-evaluation, baseline-comparator
- Forbidden: transformers

## reviewer

- Always on: architecture-originality-gate, experiment-ledger-auditor
- Optional: none
- smoke: none
- small: llm-evaluation
- full: llm-evaluation
- Forbidden: transformers, pytorch-lightning
