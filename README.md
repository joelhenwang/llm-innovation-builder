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
```

## Layout

- `ideas/<idea_id>/`: generated idea bundles, plugins, configs, run artifacts, and reports
- `baselines/`: internal baseline metadata
- `src/auto_llm_innovator/`: framework package
- `tests/`: lifecycle, CLI, originality, and orchestration tests
