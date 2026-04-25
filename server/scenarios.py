from __future__ import annotations

import random
from dataclasses import asdict, dataclass


SERVICE_GRAPH = {
    "auth-api": ["main-db", "cache", "queue"],
    "billing-api": ["main-db", "queue"],
    "worker": ["queue", "main-db"],
    "main-db": [],
    "cache": [],
    "queue": [],
}


@dataclass(frozen=True)
class ScenarioMetadata:
    id: str
    name: str
    difficulty: str
    description: str
    incident_brief: str
    max_steps: int
    root_causes: tuple[str, ...]
    relevant_evidence: tuple[str, ...]


SCENARIO_METADATA = {
    "bad_auth_deploy": ScenarioMetadata(
        id="bad_auth_deploy",
        name="Auth API Regression After Deploy",
        difficulty="easy",
        description="A fresh auth-api deploy introduced a request-path regression and elevated 5xx.",
        incident_brief="Pager fired: login failures spiked immediately after an auth-api rollout.",
        max_steps=10,
        root_causes=("bad_auth_deploy",),
        relevant_evidence=("deploy_timeline", "auth_regression_logs"),
    ),
    "worker_memory_leak": ScenarioMetadata(
        id="worker_memory_leak",
        name="Worker Memory Leak and Queue Backlog",
        difficulty="medium",
        description="A leaking worker starves queue processing and spills customer latency into auth-api.",
        incident_brief="Incident: queue depth is climbing and auth-api latency is rising behind a degraded worker tier.",
        max_steps=12,
        root_causes=("worker_memory_leak",),
        relevant_evidence=("worker_memory_growth", "queue_backlog"),
    ),
    "db_pool_exhaustion": ScenarioMetadata(
        id="db_pool_exhaustion",
        name="Database Pool Exhaustion",
        difficulty="medium",
        description="Main DB is saturated and stale pooled connections are pinning auth-api requests.",
        incident_brief="High-severity alert: auth-api is timing out while main-db pool usage is pinned near capacity.",
        max_steps=12,
        root_causes=("db_pool_exhaustion",),
        relevant_evidence=("db_pool_saturation", "stale_connections"),
    ),
    "cache_flag_storm": ScenarioMetadata(
        id="cache_flag_storm",
        name="Shadow Reads Feature Flag Storm",
        difficulty="medium",
        description="A feature flag enabled extra reads, crushing cache hit-rate and amplifying DB load.",
        incident_brief="User-facing latency rose after a feature rollout that changed read behavior in auth-api.",
        max_steps=11,
        root_causes=("shadow_reads_flag",),
        relevant_evidence=("flag_shadow_reads", "cache_miss_spike"),
    ),
    "queue_backlog_regression": ScenarioMetadata(
        id="queue_backlog_regression",
        name="Billing Producer Regression",
        difficulty="hard",
        description="A billing deploy emits poison messages that backlog the queue and choke the worker fleet.",
        incident_brief="Billing retries are exploding and queue depth is growing faster than the worker fleet can drain it.",
        max_steps=13,
        root_causes=("billing_queue_regression",),
        relevant_evidence=("billing_regression", "poison_messages"),
    ),
    "network_partition_failover": ScenarioMetadata(
        id="network_partition_failover",
        name="Zone Network Partition",
        difficulty="hard",
        description="A zonal partition isolates auth-api from the primary DB path while the DB itself looks healthy.",
        incident_brief="Customers are hitting intermittent auth failures, but the database still reports green from its own probes.",
        max_steps=12,
        root_causes=("network_partition",),
        relevant_evidence=("dependency_partition", "zone_specific_outage"),
    ),
}


TASKS = {scenario_id: asdict(metadata) for scenario_id, metadata in SCENARIO_METADATA.items()}
DEFAULT_TASK_ID = "bad_auth_deploy"


def build_scenario(task_id: str, seed: int | None = None) -> dict:
    if task_id not in SCENARIO_METADATA:
        raise ValueError(f"Unknown task_id '{task_id}'")

    rng = random.Random(seed)
    if task_id == "bad_auth_deploy":
        return _build_bad_auth_deploy(rng)
    if task_id == "worker_memory_leak":
        return _build_worker_memory_leak(rng)
    if task_id == "db_pool_exhaustion":
        return _build_db_pool_exhaustion(rng)
    if task_id == "cache_flag_storm":
        return _build_cache_flag_storm(rng)
    if task_id == "queue_backlog_regression":
        return _build_queue_backlog_regression(rng)
    return _build_network_partition_failover(rng)


def _version(rng: random.Random, *, stable: bool = False) -> str:
    patch = rng.randint(10, 99)
    suffix = "stable" if stable else f"rc{rng.randint(1, 4)}"
    return f"2026.04.{patch}-{suffix}"


def _service(version: str, replicas: int = 2) -> dict:
    return {
        "status": "RUNNING",
        "version": version,
        "replicas": replicas,
        "error_rate": 0.02,
        "latency": 0.08,
        "cpu_load": 0.24,
        "memory_usage": 0.33,
        "saturation": 0.22,
    }


def _base_world(metadata: ScenarioMetadata, rng: random.Random) -> dict:
    auth_version = _version(rng, stable=False)
    billing_version = _version(rng, stable=False)
    worker_version = _version(rng, stable=True)
    db_version = f"postgres-15.{rng.randint(2, 9)}"
    cache_version = f"redis-7.{rng.randint(0, 5)}"
    queue_version = f"queue-gateway-{rng.randint(3, 6)}.{rng.randint(0, 9)}"

    services = {
        "auth-api": _service(auth_version, replicas=2),
        "billing-api": _service(billing_version, replicas=2),
        "worker": _service(worker_version, replicas=2),
        "main-db": _service(db_version, replicas=2),
        "cache": _service(cache_version, replicas=2),
        "queue": _service(queue_version, replicas=2),
    }
    return {
        "id": metadata.id,
        "name": metadata.name,
        "difficulty": metadata.difficulty,
        "description": metadata.description,
        "incident_brief": metadata.incident_brief,
        "max_steps": metadata.max_steps,
        "services": services,
        "dependencies": {service: list(children) for service, children in SERVICE_GRAPH.items()},
        "deploy_history": [],
        "config": {
            "auth-api": {"shadow_reads": "off", "release_channel": "stable"},
            "billing-api": {"poison_guard": "enabled", "release_channel": "stable"},
            "worker": {"queue_consumer_batch": "200", "oom_restart": "enabled"},
            "main-db": {"pool_limit": "120", "failover_target": "zone-b"},
            "cache": {"hit_ratio_target": "0.94"},
            "queue": {"dead_letter_enabled": "true"},
        },
        "logs": {service: [] for service in services},
        "dependency_health": {
            "auth-api->main-db": "healthy",
            "auth-api->cache": "healthy",
            "auth-api->queue": "healthy",
            "billing-api->main-db": "healthy",
            "billing-api->queue": "healthy",
            "worker->queue": "healthy",
            "worker->main-db": "healthy",
        },
        "smoke_tests": {
            "auth-api": "PASS /login in 84ms",
            "billing-api": "PASS /invoices in 92ms",
            "worker": "PASS background drain loop",
        },
        "timeline": [metadata.incident_brief],
        "hidden": {
            "root_causes": list(metadata.root_causes),
            "relevant_evidence": list(metadata.relevant_evidence),
            "evidence": [],
            "discovered_facts": [],
            "investigations": [],
            "communication_sent": False,
            "unsafe_actions": 0,
            "ineffective_actions": 0,
            "action_history": [],
            "pending_effects": [],
            "service_graph": [f"{source}->{target}" for source, targets in SERVICE_GRAPH.items() for target in targets],
            "queue_depth": 42,
            "db_scaled": False,
            "connections_cleared": False,
            "db_failed_over": False,
            "temp_failover_risk": 0,
            "customer_impact": metadata.incident_brief,
            "notes": [],
        },
    }


def _build_bad_auth_deploy(rng: random.Random) -> dict:
    metadata = SCENARIO_METADATA["bad_auth_deploy"]
    world = _base_world(metadata, rng)
    stable_version = _version(rng, stable=True)
    bad_version = _version(rng, stable=False)
    world["services"]["auth-api"]["version"] = bad_version
    world["deploy_history"] = [
        f"04:05 deploy auth-api {stable_version} by ci-bot",
        f"04:12 deploy auth-api {bad_version} by ci-bot",
    ]
    world["logs"]["auth-api"] = [
        f"[ERROR] release {bad_version} raised KeyError in login serializer",
        "[WARN] auth-api 5xx crossed 30% immediately after rollout",
        "[INFO] upstreams remain healthy; regression isolated to auth-api handlers",
    ]
    world["hidden"].update({"stable_auth_version": stable_version, "bad_auth_deploy_active": True})
    return world


def _build_worker_memory_leak(rng: random.Random) -> dict:
    metadata = SCENARIO_METADATA["worker_memory_leak"]
    world = _base_world(metadata, rng)
    world["hidden"].update({"memory_leak_active": True, "queue_depth": 760})
    world["logs"]["worker"] = [
        "[WARN] worker RSS exceeded 95% of cgroup limit",
        "[ERROR] allocator fragmentation keeps growing across batches",
        "[WARN] queue drain loop is falling behind incoming traffic",
    ]
    world["logs"]["queue"] = [
        "[WARN] queue depth crossed 700 messages",
        "[INFO] consumers are connected but not acknowledging fast enough",
    ]
    return world


def _build_db_pool_exhaustion(rng: random.Random) -> dict:
    metadata = SCENARIO_METADATA["db_pool_exhaustion"]
    world = _base_world(metadata, rng)
    world["hidden"].update({"db_pool_exhausted": True, "db_scaled": False, "connections_cleared": False})
    world["logs"]["main-db"] = [
        "[WARN] connection pool saturation is pinned above 95%",
        "[ERROR] stale idle sessions are holding pool slots for too long",
        "[INFO] replication lag remains within normal limits",
    ]
    world["logs"]["auth-api"] = [
        "[WARN] upstream database acquire timeout breached 1.2s",
        "[ERROR] request retries are stacking behind pool waits",
    ]
    return world


def _build_cache_flag_storm(rng: random.Random) -> dict:
    metadata = SCENARIO_METADATA["cache_flag_storm"]
    world = _base_world(metadata, rng)
    world["config"]["auth-api"]["shadow_reads"] = "on"
    world["hidden"].update({"shadow_reads_enabled": True})
    world["deploy_history"] = [
        f"03:58 deploy auth-api {_version(rng, stable=True)} by ci-bot",
        "04:09 rollout feature flag shadow_reads=on to 100%",
    ]
    world["logs"]["auth-api"] = [
        "[WARN] request fanout doubled after shadow_reads flag enablement",
        "[INFO] cache miss fallback path now hits primary DB on every request",
    ]
    world["logs"]["cache"] = [
        "[WARN] cache miss ratio rose to 43%",
        "[INFO] cache infrastructure itself is still healthy",
    ]
    return world


def _build_queue_backlog_regression(rng: random.Random) -> dict:
    metadata = SCENARIO_METADATA["queue_backlog_regression"]
    world = _base_world(metadata, rng)
    stable_billing_version = _version(rng, stable=True)
    bad_billing_version = _version(rng, stable=False)
    world["services"]["billing-api"]["version"] = bad_billing_version
    world["deploy_history"] = [
        f"03:51 deploy billing-api {stable_billing_version} by ci-bot",
        f"04:08 deploy billing-api {bad_billing_version} by ci-bot",
    ]
    world["logs"]["billing-api"] = [
        f"[ERROR] release {bad_billing_version} is emitting malformed queue payloads",
        "[WARN] dead-letter growth is accelerating with repeated poison messages",
    ]
    world["logs"]["queue"] = [
        "[WARN] queue depth crossed 1100 messages",
        "[ERROR] poison messages are being retried repeatedly",
    ]
    world["hidden"].update(
        {
            "billing_regression_active": True,
            "queue_depth": 1120,
            "stable_billing_version": stable_billing_version,
        }
    )
    return world


def _build_network_partition_failover(rng: random.Random) -> dict:
    metadata = SCENARIO_METADATA["network_partition_failover"]
    world = _base_world(metadata, rng)
    world["dependency_health"]["auth-api->main-db"] = "degraded"
    world["logs"]["auth-api"] = [
        "[WARN] auth-api requests from zone-a are timing out while zone-b remains mostly clean",
        "[INFO] cache reads look normal; upstream DB path is the only suspect hop",
    ]
    world["logs"]["main-db"] = [
        "[INFO] main-db local probes still pass from the primary zone",
        "[WARN] no storage saturation detected despite customer-facing errors",
    ]
    world["hidden"].update({"network_partition_active": True, "db_failed_over": False})
    return world
