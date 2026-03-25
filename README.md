# auto-llm-innovator

CLI-first scaffold for autonomous idea intake, originality review, idea-specific model plugins, and phase-based training/evaluation orchestration for creative sub-2.1B language models.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
innovator submit "Invent a recurrently modulated attention language model that is not a plain transformer."
innovator run idea-0001 --phase all
innovator status idea-0001
innovator report idea-0001
innovator skills list
innovator skills explain reviewer --phase small
```

## Layout

- `ideas/<idea_id>/`: generated idea bundles, plugins, configs, run artifacts, and reports
- `orchestration/skills.json`: reviewed skill registry and role-to-phase routing policy
- `skills/internal/`: project-specific internal skills used by the orchestrator
- `baselines/`: internal baseline metadata
- `src/auto_llm_innovator/`: framework package
- `tests/`: lifecycle, CLI, originality, and orchestration tests
