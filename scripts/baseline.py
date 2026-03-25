import json
from typing import Any

import requests

BASE_URL = "http://localhost:7860"


def _logs_contain(observation: dict[str, Any], needle: str) -> bool:
    return any(needle in line for line in observation.get("logs", []))


def _latest_warn(observation: dict[str, Any]) -> str | None:
    warn_lines = [line for line in observation.get("logs", []) if "[WARN]" in line]
    return warn_lines[-1] if warn_lines else None


def get_action_from_llm(
    observation: dict[str, Any],
    task_id: str,
    history: list[dict[str, str]],
    pending_retry: dict[str, str] | None = None,
) -> dict[str, str]:
    """Mock policy for local testing without external LLM calls."""
    if pending_retry is not None:
        return pending_retry

    metrics = observation.get("metrics", {})
    logs = observation.get("logs", [])
    logs_blob = "\n".join(logs)

    if task_id == "service_restart":
        return {"command": "restart_service", "target": "auth-api"}

    if task_id == "memory_leak":
        leak_cleared = (
            _logs_contain(observation, "Leaking worker process terminated")
            or any(action["command"] == "kill_process" for action in history)
            and metrics.get("memory_usage", 1.0) < 0.60
        )
        if leak_cleared:
            return {"command": "restart_service", "target": "auth-api"}

        if (
            "worker-process" in logs_blob
            or "OOM" in logs_blob
            or metrics.get("memory_usage", 0.0) > 0.85
        ):
            return {"command": "kill_process", "target": "auth-api"}

        return {"command": "get_logs", "target": "system"}

    if task_id == "db_connection_exhaustion":
        db_scaled = (
            _logs_contain(observation, "Database pool capacity scaled out")
            or any(action["command"] == "scale_db" for action in history)
            and metrics.get("latency", 1.0) < 0.60
        )
        if db_scaled:
            return {"command": "clear_connections", "target": "main-db"}

        if metrics.get("latency", 0.0) > 0.40 or metrics.get("error_rate", 0.0) > 0.10:
            return {"command": "scale_db", "target": "main-db"}

        return {"command": "check_metrics", "target": "main-db"}

    return {"command": "get_logs", "target": "system"}


def run_task(task_id: str) -> float:
    print(f"Starting task: {task_id}")
    observation = requests.post(f"{BASE_URL}/reset", params={"task_id": task_id}, timeout=10).json()
    history: list[dict[str, str]] = []
    pending_retry: dict[str, str] | None = None

    for step in range(10):
        action = get_action_from_llm(observation, task_id, history, pending_retry)
        print(f"Step {step}: {action['command']} -> {action['target']}")

        history.append(action)
        step_response = requests.post(f"{BASE_URL}/step", json=action, timeout=10).json()
        observation = step_response["observation"]
        done = step_response["done"]

        warn_line = _latest_warn(observation)
        if warn_line and "did not stick due to transient control-plane failure" in warn_line:
            pending_retry = action
        else:
            pending_retry = None

        if done:
            break

    final_score = requests.get(f"{BASE_URL}/grader", timeout=10).json()["score"]
    print(f"Completed task: {task_id} score={final_score}")
    return final_score


if __name__ == "__main__":
    task_response = requests.get(f"{BASE_URL}/tasks", timeout=10).json()
    task_ids = [task["id"] for task in task_response["tasks"]]

    results = {task_id: run_task(task_id) for task_id in task_ids}
    print("\nFINAL BASELINE SCORES:")
    print(json.dumps(results, indent=2))
