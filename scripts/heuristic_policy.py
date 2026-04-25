from __future__ import annotations

import random
from typing import Any


def _seen(history: list[dict[str, str]], command: str, target: str) -> bool:
    return any(item.get("command") == command and item.get("target") == target for item in history)


def _queue_depth(observation: dict[str, Any]) -> float:
    return float(observation.get("global_signals", {}).get("queue_depth", 0.0))


def _text(observation: dict[str, Any]) -> str:
    facts = observation.get("discovered_facts", [])
    events = observation.get("recent_events", [])
    return " ".join([*facts, *events]).lower()


def choose_action(
    observation: dict[str, Any],
    history: list[dict[str, str]] | None = None,
    *,
    rng: random.Random | None = None,
) -> dict[str, str]:
    rng = rng or random.Random()
    history = history or []
    task_id = observation.get("task_id")
    text = _text(observation)
    queue_depth = _queue_depth(observation)

    if task_id == "bad_auth_deploy":
        if not _seen(history, "get_recent_deploys", "system"):
            return {"command": "get_recent_deploys", "target": "system"}
        if not _seen(history, "query_logs", "auth-api"):
            return {"command": "query_logs", "target": "auth-api"}
        if not _seen(history, "rollback_deploy", "auth-api"):
            return {"command": "rollback_deploy", "target": "auth-api"}
        if not _seen(history, "run_smoke_test", "auth-api"):
            return {"command": "run_smoke_test", "target": "auth-api"}
        if not _seen(history, "post_status_update", "system"):
            return {"command": "post_status_update", "target": "system"}
        return {"command": "query_logs", "target": "auth-api"}

    if task_id == "worker_memory_leak":
        if "memory" not in text and not _seen(history, "query_logs", "worker"):
            return {"command": "query_logs", "target": "worker"}
        if "queue" not in text and not _seen(history, "get_metrics", "queue"):
            return {"command": "get_metrics", "target": "queue"}
        if not _seen(history, "restart_service", "worker"):
            return {"command": "restart_service", "target": "worker"}
        if queue_depth > 120 and not _seen(history, "drain_queue", "queue"):
            return {"command": "drain_queue", "target": "queue"}
        if queue_depth > 180 and not _seen(history, "scale_service", "worker"):
            return {"command": "scale_service", "target": "worker"}
        if not _seen(history, "run_smoke_test", "auth-api"):
            return {"command": "run_smoke_test", "target": "auth-api"}
        return {"command": "query_logs", "target": "queue"}

    if task_id == "db_pool_exhaustion":
        if "pool" not in text and not _seen(history, "get_metrics", "main-db"):
            return {"command": "get_metrics", "target": "main-db"}
        if "stale" not in text and not _seen(history, "query_logs", "main-db"):
            return {"command": "query_logs", "target": "main-db"}
        if not _seen(history, "scale_service", "main-db"):
            return {"command": "scale_service", "target": "main-db"}
        if not _seen(history, "clear_connections", "main-db"):
            return {"command": "clear_connections", "target": "main-db"}
        if not _seen(history, "run_smoke_test", "auth-api"):
            return {"command": "run_smoke_test", "target": "auth-api"}
        return {"command": "check_dependency", "target": "auth-api"}

    if task_id == "cache_flag_storm":
        if "shadow_reads" not in text and not _seen(history, "inspect_config", "auth-api"):
            return {"command": "inspect_config", "target": "auth-api"}
        if "cache" not in text and not _seen(history, "query_logs", "cache"):
            return {"command": "query_logs", "target": "cache"}
        if not _seen(history, "disable_feature_flag", "auth-api"):
            return {"command": "disable_feature_flag", "target": "auth-api"}
        if not _seen(history, "run_smoke_test", "auth-api"):
            return {"command": "run_smoke_test", "target": "auth-api"}
        return {"command": "get_recent_deploys", "target": "system"}

    if task_id == "queue_backlog_regression":
        if "billing" not in text and not _seen(history, "get_recent_deploys", "system"):
            return {"command": "get_recent_deploys", "target": "system"}
        if "malformed" not in text and not _seen(history, "query_logs", "billing-api"):
            return {"command": "query_logs", "target": "billing-api"}
        if "poison" not in text and not _seen(history, "query_logs", "queue"):
            return {"command": "query_logs", "target": "queue"}
        if not _seen(history, "rollback_deploy", "billing-api"):
            return {"command": "rollback_deploy", "target": "billing-api"}
        if queue_depth > 120 and not _seen(history, "drain_queue", "queue"):
            return {"command": "drain_queue", "target": "queue"}
        if queue_depth > 180 and not _seen(history, "scale_service", "worker"):
            return {"command": "scale_service", "target": "worker"}
        if not _seen(history, "run_smoke_test", "billing-api"):
            return {"command": "run_smoke_test", "target": "billing-api"}
        return {"command": "query_logs", "target": "worker"}

    if task_id == "network_partition_failover":
        if "partition" not in text and not _seen(history, "check_dependency", "auth-api"):
            return {"command": "check_dependency", "target": "auth-api"}
        if "zone" not in text and not _seen(history, "query_logs", "auth-api"):
            return {"command": "query_logs", "target": "auth-api"}
        if not _seen(history, "failover_db", "main-db"):
            return {"command": "failover_db", "target": "main-db"}
        if not _seen(history, "run_smoke_test", "auth-api"):
            return {"command": "run_smoke_test", "target": "auth-api"}
        if not _seen(history, "post_status_update", "system"):
            return {"command": "post_status_update", "target": "system"}
        return {"command": "check_dependency", "target": "auth-api"}

    return rng.choice(
        [
            {"command": "query_logs", "target": "auth-api"},
            {"command": "get_metrics", "target": "main-db"},
            {"command": "get_recent_deploys", "target": "system"},
        ]
    )
