from typing import Any


TASK_SCORE_DIVISORS = {
    "service_restart": 1.0,
    "memory_leak": 1.5,
    "db_connection_exhaustion": 2.0,
}


def _base_grade(env: Any) -> dict[str, Any]:
    max_steps = env.task.get("max_steps", 15)
    progress_ratio = round(env.progress_index / len(env.required_sequence), 4)
    resolved = env._is_resolved()
    divisor = TASK_SCORE_DIVISORS[env.task_id]
    raw_score = env.cumulative_reward / divisor
    score = max(0.0, min(1.0, raw_score))
    if not resolved:
        score = min(score, 0.4)
    score = round(score, 4)

    notes = []
    if resolved:
        notes.append("Incident resolved and nominal metrics restored.")
    elif env.progress_index > 0:
        notes.append("Partial remediation achieved but the episode ended before recovery.")
    if any(item.value < 0 for item in env.reward_history):
        notes.append("Negative rewards recorded for inefficient or incorrect actions.")
    notes.append(
        f"Penalty-driven score derived from cumulative_reward={env.cumulative_reward:.2f} using divisor={divisor:.2f}."
    )

    return {
        "score": score,
        "resolved": resolved,
        "max_steps": max_steps,
        "steps_taken": env.step_count,
        "progress_ratio": progress_ratio,
        "reward_total": round(env.cumulative_reward, 2),
        "expected_sequence": env.required_sequence,
        "completed_steps": env.completed_steps,
        "notes": notes,
    }


def grade_service_restart(env: Any) -> dict[str, Any]:
    return _base_grade(env)


def grade_memory_leak(env: Any) -> dict[str, Any]:
    result = _base_grade(env)
    if "kill_process" not in env.completed_steps:
        result["notes"].append("Root-cause worker termination was not completed.")
    return result


def grade_db_connection_exhaustion(env: Any) -> dict[str, Any]:
    result = _base_grade(env)
    if env.completed_steps[:1] != ["scale_db"] and env.progress_index > 0:
        result["notes"].append("Database scaling must precede connection clearing.")
    return result


TASK_GRADERS = {
    "service_restart": grade_service_restart,
    "memory_leak": grade_memory_leak,
    "db_connection_exhaustion": grade_db_connection_exhaustion,
}


TASKS = {
    "service_restart": {
        "id": "service_restart",
        "name": "Easy: API Process Crash",
        "difficulty": "easy",
        "max_steps": 5,
        "noise_level": 0.0,
        "required_sequence": ["restart_service"],
        "flakiness_range": (0.10, 0.12),
        "nominal_metrics": {
            "error_rate": 0.02,
            "latency": 0.06,
            "cpu_load": 0.18,
            "memory_usage": 0.26,
        },
        "initial_state": {
            "services": {
                "auth-api": "DOWN",
                "main-db": "RUNNING",
                "cache": "RUNNING",
            },
            "metrics": {
                "cpu_load": 0.22,
                "memory_usage": 0.33,
                "latency": 0.78,
                "error_rate": 0.91,
            },
            "state": {
                "internal_error_rate": 0.91,
                "is_database_locked": False,
                "active_incidents": ["auth_api_process_crash"],
                "leak_active": False,
                "db_pool_exhausted": False,
                "db_scaled": False,
                "connections_cleared": False,
            },
            "logs": [
                "System initialized. All services green except active incident targets.",
                "[ERROR] auth-api exited with code 137",
                "[ERROR] supervisor: auth-api process not running",
                "[HINT] Client traffic is failing because auth-api is DOWN",
            ],
        },
    },
    "memory_leak": {
        "id": "memory_leak",
        "name": "Medium: Worker Memory Leak",
        "difficulty": "medium",
        "max_steps": 10,
        "noise_level": 0.5,
        "required_sequence": ["kill_process", "restart_service"],
        "flakiness_range": (0.12, 0.16),
        "nominal_metrics": {
            "error_rate": 0.03,
            "latency": 0.08,
            "cpu_load": 0.24,
            "memory_usage": 0.38,
        },
        "initial_state": {
            "services": {
                "auth-api": "DEGRADED",
                "main-db": "RUNNING",
                "cache": "RUNNING",
            },
            "metrics": {
                "cpu_load": 0.64,
                "memory_usage": 0.97,
                "latency": 0.83,
                "error_rate": 0.42,
            },
            "state": {
                "internal_error_rate": 0.42,
                "is_database_locked": False,
                "active_incidents": ["worker_memory_leak"],
                "leak_active": True,
                "db_pool_exhausted": False,
                "db_scaled": False,
                "connections_cleared": False,
            },
            "logs": [
                "System initialized. Triage the active incident.",
                "[WARN] worker-process memory usage climbed above 95%",
                "[ERROR] kernel: OOMKilled candidate detected for worker-process",
                "[ERROR] auth-api requests timing out while worker queue backs up",
            ],
        },
    },
    "db_connection_exhaustion": {
        "id": "db_connection_exhaustion",
        "name": "Hard: Distributed DB Pool Failure",
        "difficulty": "hard",
        "max_steps": 15,
        "noise_level": 0.8,
        "required_sequence": ["scale_db", "clear_connections"],
        "flakiness_range": (0.15, 0.20),
        "nominal_metrics": {
            "error_rate": 0.02,
            "latency": 0.09,
            "cpu_load": 0.29,
            "memory_usage": 0.41,
        },
        "initial_state": {
            "services": {
                "auth-api": "RUNNING",
                "main-db": "DEGRADED",
                "cache": "RUNNING",
            },
            "metrics": {
                "cpu_load": 0.46,
                "memory_usage": 0.58,
                "latency": 0.97,
                "error_rate": 0.37,
            },
            "state": {
                "internal_error_rate": 0.37,
                "is_database_locked": True,
                "active_incidents": ["db_connection_pool_exhaustion"],
                "leak_active": False,
                "db_pool_exhausted": True,
                "db_scaled": False,
                "connections_cleared": False,
            },
            "logs": [
                "System initialized. Triage the active incident.",
                "[INFO] auth-api pod health checks passing",
                "[INFO] main-db replication lag within normal range",
                "[WARN] Request latency elevated above SLO without explicit application errors",
            ],
        },
    },
}
