from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path.cwd()
IDEAS_DIRNAME = "ideas"
BASELINES_DIRNAME = "baselines"
GPT2_TOKENIZER = "gpt2"
PARAMETER_CAP = 2_100_000_000
DEFAULT_SMALLER_SCALE_CAP = 600_000_000
DEFAULT_MAX_AUTONOMOUS_RETRIES = 25
PHASES = ("smoke", "small", "full")
