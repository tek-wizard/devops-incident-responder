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
from scripts.random_policy import choose_action as random_action


def _policy_fn(name: str):
    if name == "heuristic":
        return heuristic_action
    if name == "random":
        return random_action
    raise ValueError(f"Unknown policy '{name}'")


def collect_rollouts(policy_name: str, seed_set_name: str) -> list[dict]:
    policy = _policy_fn(policy_name)
    seeds = get_seed_set(seed_set_name)
    episodes = []

    for task_id in TASKS:
        for seed in seeds:
            env = DevOpsEnv()
            observation = env.reset_episode(task_id=task_id, seed=seed).model_dump()
            history: list[dict[str, str]] = []
            rng = random.Random(seed)
            steps = []

            while not env.done:
                action = policy(observation, history, rng=rng)
                history.append(action)
                next_observation, reward, done, info = env.step_episode(IncidentAction(**action))
                steps.append(
                    {
                        "observation": observation,
                        "action": action,
                        "reward": reward.model_dump(),
                        "info": info,
                        "done": done,
                    }
                )
                observation = next_observation.model_dump()
                if done:
                    break

            episodes.append(
                {
                    "task_id": task_id,
                    "seed": seed,
                    "policy": policy_name,
                    "grade": env.grade().model_dump(),
                    "steps": steps,
                }
            )

    return episodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect rollout traces from a baseline policy.")
    parser.add_argument("--policy", choices=["heuristic", "random"], default="heuristic")
    parser.add_argument("--seed-set", choices=["train", "val", "holdout"], default="train")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    episodes = collect_rollouts(args.policy, args.seed_set)
    with args.output.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode) + "\n")
    print(f"Wrote {len(episodes)} episodes to {args.output}")


if __name__ == "__main__":
    main()
