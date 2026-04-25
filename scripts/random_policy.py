from __future__ import annotations

import random
from typing import Any


def choose_action(
    observation: dict[str, Any],
    history: list[dict[str, str]] | None = None,
    *,
    rng: random.Random | None = None,
) -> dict[str, str]:
    rng = rng or random.Random()
    commands = list(observation.get("available_commands", []))
    services = list(observation.get("service_status", {}).keys())
    history = history or []

    command = rng.choice(commands)
    if command in {"get_recent_deploys", "post_status_update"}:
        target = "system"
    elif command in {"drain_queue"}:
        target = "queue"
    elif command in {"clear_connections", "failover_db"}:
        target = "main-db"
    elif command in {"inspect_config", "disable_feature_flag"}:
        target = rng.choice(["auth-api", "billing-api", "main-db"])
    elif command in {"check_dependency", "run_smoke_test"}:
        target = rng.choice(["auth-api", "billing-api", "worker"])
    else:
        target = rng.choice(services)
    return {"command": command, "target": target}
