from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server.environment import DevOpsEnv
from server.models import IncidentAction
from server.seed_sets import get_seed_set
from server.tasks import TASKS
from scripts.heuristic_policy import choose_action as heuristic_action
from scripts.metrics import aggregate_results
from scripts.random_policy import choose_action as random_action


def _policy_fn(name: str):
    if name == "heuristic":
        return heuristic_action
    if name == "random":
        return random_action
    raise ValueError(f"Unknown policy '{name}'")


def run_eval(policy_name: str, seed_set_name: str) -> dict:
    policy = _policy_fn(policy_name)
    seeds = get_seed_set(seed_set_name)
    results = []

    for task_id in TASKS:
        for seed in seeds:
            env = DevOpsEnv()
            observation = env.reset_episode(task_id=task_id, seed=seed).model_dump()
            history: list[dict[str, str]] = []
            rng = random.Random(seed)

            while not env.done:
                action = policy(observation, history, rng=rng)
                history.append(action)
                observation_model, _, done, _ = env.step_episode(IncidentAction(**action))
                observation = observation_model.model_dump()
                if done:
                    break

            grade = env.grade().model_dump()
            grade.update({"seed": seed, "task_id": task_id})
            results.append(grade)

    return {"policy": policy_name, "seed_set": seed_set_name, "summary": aggregate_results(results), "episodes": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a baseline policy against the seeded incident environment.")
    parser.add_argument("--policy", choices=["heuristic", "random"], default="heuristic")
    parser.add_argument("--seed-set", choices=["train", "val", "holdout"], default="val")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    payload = run_eval(args.policy, args.seed_set)
    rendered = json.dumps(payload, indent=2)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
