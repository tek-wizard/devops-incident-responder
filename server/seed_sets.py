from __future__ import annotations

TRAIN_SEEDS = [101, 103, 107, 109, 113, 127, 131, 137]
VALIDATION_SEEDS = [201, 211, 223, 227]
HOLDOUT_SEEDS = [307, 311, 313, 317]

SEED_SETS = {
    "train": TRAIN_SEEDS,
    "val": VALIDATION_SEEDS,
    "holdout": HOLDOUT_SEEDS,
}


def get_seed_set(name: str) -> list[int]:
    if name not in SEED_SETS:
        raise KeyError(f"Unknown seed set '{name}'")
    return list(SEED_SETS[name])
