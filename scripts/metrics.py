from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_task[result["task_id"]].append(result)

    task_summaries = {}
    for task_id, task_results in by_task.items():
        task_summaries[task_id] = {
            "episodes": len(task_results),
            "solve_rate": round(mean(1.0 if item["resolved"] else 0.0 for item in task_results), 4),
            "avg_score": round(mean(item["score"] for item in task_results), 4),
            "avg_reward": round(mean(item["reward_total"] for item in task_results), 4),
            "avg_steps": round(mean(item["steps_taken"] for item in task_results), 4),
            "avg_investigation_coverage": round(mean(item["investigation_coverage"] for item in task_results), 4),
            "avg_unsafe_actions": round(mean(item["unsafe_actions"] for item in task_results), 4),
        }

    return {
        "episodes": len(results),
        "overall": {
            "solve_rate": round(mean(1.0 if item["resolved"] else 0.0 for item in results), 4) if results else 0.0,
            "avg_score": round(mean(item["score"] for item in results), 4) if results else 0.0,
            "avg_reward": round(mean(item["reward_total"] for item in results), 4) if results else 0.0,
            "avg_steps": round(mean(item["steps_taken"] for item in results), 4) if results else 0.0,
        },
        "by_task": task_summaries,
    }
