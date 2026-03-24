from __future__ import annotations


def dataset_plan_for_phase(phase: str) -> dict:
    plans = {
        "smoke": {
            "dataset_name": "synthetic-shapes",
            "description": "Toy tensors and token snippets for shape and loss validation.",
            "target_tokens": 100_000,
        },
        "small": {
            "dataset_name": "small-curated-corpus",
            "description": "Small corpus for fast learnability checks.",
            "target_tokens": 5_000_000,
        },
        "full": {
            "dataset_name": "production-like-corpus",
            "description": "Large corpus placeholder for full-scale training evaluation.",
            "target_tokens": 500_000_000,
        },
    }
    return plans[phase]
