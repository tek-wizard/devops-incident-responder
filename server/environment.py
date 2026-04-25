from __future__ import annotations

import json
from typing import Any

from openenv_core import Environment
from openenv_core.env_server.types import State

from .models import (
    EnvironmentState,
    HiddenIncidentState,
    IncidentAction,
    RewardBreakdown,
    RewardOutput,
    ServiceSnapshot,
    SystemObservation,
)
from .rewards import build_grade, new_breakdown, total_reward
from .scenarios import DEFAULT_TASK_ID, SCENARIO_METADATA, build_scenario


class DevOpsEnv(Environment):
    INVESTIGATION_COMMANDS = {
        "query_logs",
        "get_metrics",
        "get_recent_deploys",
        "inspect_config",
        "check_dependency",
        "run_smoke_test",
    }
    MITIGATION_COMMANDS = {
        "restart_service",
        "rollback_deploy",
        "disable_feature_flag",
        "scale_service",
        "clear_connections",
        "drain_queue",
        "failover_db",
        "post_status_update",
    }
    COMMANDS = sorted(INVESTIGATION_COMMANDS | MITIGATION_COMMANDS)
    HIGH_RISK_COMMANDS = {"failover_db"}

    def __init__(self, session_id: str | None = None):
        super().__init__()
        self.session_id = session_id
        self.reset_episode()

    def reset(self, **kwargs: Any) -> str:
        """OpenEnv-compatible reset for dataset-column kwargs."""
        task_id = str(kwargs.get("task_id", DEFAULT_TASK_ID))
        seed_value = kwargs.get("seed")
        seed = int(seed_value) if seed_value is not None else None
        return self.reset_episode(task_id=task_id, seed=seed).model_dump_json()

    def reset_episode(self, task_id: str = DEFAULT_TASK_ID, seed: int | None = None) -> SystemObservation:
        self.seed = seed
        self.step_count = 0
        self.task_id = task_id
        self.done = False
        self.last_info: dict[str, Any] = {}
        self.reward_history: list[RewardOutput] = []
        self.cumulative_reward = 0.0
        self.world = build_scenario(task_id, seed=seed)
        self.max_steps = int(self.world["max_steps"])
        self.last_reward = RewardOutput(
            value=0.0,
            reason="environment_reset",
            breakdown=RewardBreakdown(),
            done=False,
        )
        self._refresh_world()
        return self._get_obs()

    def _get_obs(self) -> SystemObservation:
        return SystemObservation(
            session_id=self.session_id,
            task_id=self.task_id,
            difficulty=self.world["difficulty"],
            incident_brief=self.world["incident_brief"],
            step_count=self.step_count,
            remaining_steps=max(self.max_steps - self.step_count, 0),
            global_signals=self._build_global_signals(),
            service_status={
                name: ServiceSnapshot(**service_state)
                for name, service_state in self.world["services"].items()
            },
            recent_events=list(self.world["timeline"][-6:]),
            discovered_facts=list(self.world["hidden"]["discovered_facts"][-8:]),
            available_commands=list(self.COMMANDS),
        )

    def _build_global_signals(self) -> dict[str, float]:
        customer_services = ["auth-api", "billing-api"]
        active = [self.world["services"][service] for service in customer_services]
        avg_error = sum(service["error_rate"] for service in active) / len(active)
        peak_latency = max(service["latency"] for service in active)
        return {
            "customer_error_rate": round(avg_error, 3),
            "customer_latency": round(peak_latency, 3),
            "queue_depth": float(self.world["hidden"].get("queue_depth", 0)),
            "unsafe_actions": float(self.world["hidden"]["unsafe_actions"]),
            "investigation_coverage": round(self._investigation_coverage(), 3),
        }

    def _append_event(self, message: str) -> None:
        self.world["timeline"].append(message)

    def _add_fact(self, fact: str, tag: str | None, breakdown: RewardBreakdown) -> None:
        if fact not in self.world["hidden"]["discovered_facts"]:
            self.world["hidden"]["discovered_facts"].append(fact)
        if tag and tag not in self.world["hidden"]["evidence"]:
            self.world["hidden"]["evidence"].append(tag)
            breakdown.investigation += 0.1

    def _mark_no_effect(self, breakdown: RewardBreakdown, message: str, *, risky: bool = False) -> None:
        self.world["hidden"]["ineffective_actions"] += 1
        breakdown.safety -= 0.12
        if risky:
            self.world["hidden"]["unsafe_actions"] += 1
            breakdown.safety -= 0.18
        self._append_event(message)

    def _schedule_effect(self, kind: str, delay: int, detail: str) -> None:
        self.world["hidden"]["pending_effects"].append({"kind": kind, "delay": delay, "detail": detail})
        self._append_event(detail)

    def _advance_pending_effects(self) -> None:
        remaining_effects = []
        for effect in self.world["hidden"]["pending_effects"]:
            effect["delay"] -= 1
            if effect["delay"] <= 0:
                self._apply_effect(effect["kind"])
            else:
                remaining_effects.append(effect)
        self.world["hidden"]["pending_effects"] = remaining_effects

    def _apply_effect(self, kind: str) -> None:
        hidden = self.world["hidden"]
        if kind == "rollback_auth_deploy":
            hidden["bad_auth_deploy_active"] = False
            self.world["services"]["auth-api"]["version"] = hidden["stable_auth_version"]
            self._append_event("Rollback finished: auth-api is now serving the previous stable build.")
        elif kind == "rollback_billing_deploy":
            hidden["billing_regression_active"] = False
            self.world["services"]["billing-api"]["version"] = hidden["stable_billing_version"]
            self._append_event("Rollback finished: billing-api stopped emitting malformed queue payloads.")
        elif kind == "disable_shadow_reads":
            hidden["shadow_reads_enabled"] = False
            self.world["config"]["auth-api"]["shadow_reads"] = "off"
            self._append_event("Feature flag update propagated: shadow_reads is now disabled.")

    def _healthy_metrics(self, service: str) -> dict[str, float]:
        baseline = {
            "auth-api": {"error_rate": 0.02, "latency": 0.08, "cpu_load": 0.28, "memory_usage": 0.34, "saturation": 0.24},
            "billing-api": {"error_rate": 0.02, "latency": 0.09, "cpu_load": 0.24, "memory_usage": 0.32, "saturation": 0.22},
            "worker": {"error_rate": 0.01, "latency": 0.07, "cpu_load": 0.35, "memory_usage": 0.42, "saturation": 0.26},
            "main-db": {"error_rate": 0.01, "latency": 0.05, "cpu_load": 0.31, "memory_usage": 0.44, "saturation": 0.29},
            "cache": {"error_rate": 0.0, "latency": 0.02, "cpu_load": 0.18, "memory_usage": 0.29, "saturation": 0.16},
            "queue": {"error_rate": 0.0, "latency": 0.03, "cpu_load": 0.14, "memory_usage": 0.21, "saturation": 0.15},
        }
        return baseline[service].copy()

    def _set_service(self, service: str, *, status: str, **metrics: float) -> None:
        healthy = self._healthy_metrics(service)
        healthy.update(metrics)
        self.world["services"][service].update({"status": status, **healthy})

    def _scenario_tick(self) -> None:
        hidden = self.world["hidden"]
        task_id = self.task_id
        if task_id == "worker_memory_leak":
            if hidden.get("memory_leak_active", False):
                hidden["queue_depth"] = min(1400, hidden["queue_depth"] + 120)
            else:
                drain_rate = 220 + 80 * max(self.world["services"]["worker"]["replicas"] - 2, 0)
                hidden["queue_depth"] = max(0, hidden["queue_depth"] - drain_rate)
        elif task_id == "queue_backlog_regression":
            if hidden.get("billing_regression_active", False):
                hidden["queue_depth"] = min(1800, hidden["queue_depth"] + 140)
            else:
                drain_rate = 240 + 90 * max(self.world["services"]["worker"]["replicas"] - 2, 0)
                hidden["queue_depth"] = max(0, hidden["queue_depth"] - drain_rate)
        if hidden.get("temp_failover_risk", 0) > 0:
            hidden["temp_failover_risk"] = max(0, hidden["temp_failover_risk"] - 1)

    def _refresh_world(self) -> None:
        for service in self.world["services"]:
            self._set_service(service, status="RUNNING")
        dependency_health = self.world["dependency_health"]
        for edge in dependency_health:
            dependency_health[edge] = "healthy"
        self.world["smoke_tests"]["auth-api"] = "PASS /login in 84ms"
        self.world["smoke_tests"]["billing-api"] = "PASS /invoices in 92ms"
        self.world["smoke_tests"]["worker"] = "PASS background drain loop"

        task_id = self.task_id
        hidden = self.world["hidden"]
        if task_id == "bad_auth_deploy":
            if hidden.get("bad_auth_deploy_active", False):
                self._set_service(
                    "auth-api",
                    status="DEGRADED",
                    error_rate=0.42,
                    latency=1.32,
                    cpu_load=0.61,
                    memory_usage=0.56,
                    saturation=0.82,
                )
                self.world["smoke_tests"]["auth-api"] = "FAIL /login returns 500 from login serializer path"
        elif task_id == "worker_memory_leak":
            queue_depth = hidden["queue_depth"]
            if hidden.get("memory_leak_active", False):
                self._set_service(
                    "worker",
                    status="DEGRADED",
                    error_rate=0.24,
                    latency=0.84,
                    cpu_load=0.78,
                    memory_usage=0.98,
                    saturation=0.92,
                )
            queue_status = "DEGRADED" if queue_depth > 300 else "RUNNING"
            self._set_service(
                "queue",
                status=queue_status,
                latency=0.05,
                saturation=min(1.15, 0.18 + queue_depth / 1000.0),
            )
            if queue_depth > 120:
                self._set_service(
                    "auth-api",
                    status="DEGRADED",
                    error_rate=0.28 if queue_depth > 500 else 0.12,
                    latency=1.08 if queue_depth > 500 else 0.42,
                    cpu_load=0.44,
                    memory_usage=0.51,
                    saturation=0.71 if queue_depth > 500 else 0.44,
                )
                self.world["smoke_tests"]["auth-api"] = "FAIL /login waits on downstream queue-backed workflow"
        elif task_id == "db_pool_exhaustion":
            if not hidden.get("db_scaled", False):
                self._set_service(
                    "main-db",
                    status="DEGRADED",
                    error_rate=0.18,
                    latency=0.73,
                    cpu_load=0.66,
                    memory_usage=0.58,
                    saturation=0.98,
                )
                self._set_service(
                    "auth-api",
                    status="DEGRADED",
                    error_rate=0.31,
                    latency=1.45,
                    cpu_load=0.49,
                    memory_usage=0.47,
                    saturation=0.84,
                )
                dependency_health["auth-api->main-db"] = "degraded"
                self.world["smoke_tests"]["auth-api"] = "FAIL /login times out waiting for a DB pool slot"
            elif not hidden.get("connections_cleared", False):
                self._set_service(
                    "main-db",
                    status="RUNNING",
                    error_rate=0.06,
                    latency=0.2,
                    cpu_load=0.72,
                    memory_usage=0.62,
                    saturation=0.64,
                )
                self._set_service(
                    "auth-api",
                    status="DEGRADED",
                    error_rate=0.14,
                    latency=0.62,
                    cpu_load=0.38,
                    memory_usage=0.42,
                    saturation=0.51,
                )
                dependency_health["auth-api->main-db"] = "recovering"
                self.world["smoke_tests"]["auth-api"] = "FAIL /login still sees stale connection timeouts"
        elif task_id == "cache_flag_storm":
            if hidden.get("shadow_reads_enabled", False):
                self._set_service(
                    "cache",
                    status="DEGRADED",
                    error_rate=0.05,
                    latency=0.18,
                    cpu_load=0.42,
                    memory_usage=0.37,
                    saturation=0.71,
                )
                self._set_service(
                    "main-db",
                    status="DEGRADED",
                    error_rate=0.08,
                    latency=0.33,
                    cpu_load=0.54,
                    memory_usage=0.48,
                    saturation=0.57,
                )
                self._set_service(
                    "auth-api",
                    status="DEGRADED",
                    error_rate=0.21,
                    latency=0.92,
                    cpu_load=0.57,
                    memory_usage=0.51,
                    saturation=0.74,
                )
                dependency_health["auth-api->cache"] = "degraded"
                dependency_health["auth-api->main-db"] = "degraded"
                self.world["smoke_tests"]["auth-api"] = "FAIL /login fans out to extra reads and breaches latency SLO"
        elif task_id == "queue_backlog_regression":
            queue_depth = hidden["queue_depth"]
            if hidden.get("billing_regression_active", False):
                self._set_service(
                    "billing-api",
                    status="DEGRADED",
                    error_rate=0.27,
                    latency=1.01,
                    cpu_load=0.61,
                    memory_usage=0.47,
                    saturation=0.83,
                )
                self.world["smoke_tests"]["billing-api"] = "FAIL /invoices emits malformed queue events"
            if queue_depth > 120:
                self._set_service(
                    "queue",
                    status="DEGRADED" if queue_depth > 300 else "RUNNING",
                    latency=0.06,
                    saturation=min(1.2, 0.2 + queue_depth / 1000.0),
                )
                self._set_service(
                    "worker",
                    status="DEGRADED" if queue_depth > 400 else "RUNNING",
                    error_rate=0.16 if queue_depth > 400 else 0.05,
                    latency=0.64 if queue_depth > 400 else 0.18,
                    cpu_load=0.68 if queue_depth > 400 else 0.41,
                    memory_usage=0.73 if queue_depth > 400 else 0.46,
                    saturation=0.89 if queue_depth > 400 else 0.43,
                )
        elif task_id == "network_partition_failover":
            if hidden.get("temp_failover_risk", 0) > 0:
                self._set_service(
                    "main-db",
                    status="DEGRADED",
                    error_rate=0.18,
                    latency=0.56,
                    cpu_load=0.81,
                    memory_usage=0.61,
                    saturation=0.88,
                )
            if hidden.get("network_partition_active", False):
                self._set_service(
                    "auth-api",
                    status="DEGRADED",
                    error_rate=0.24,
                    latency=1.12,
                    cpu_load=0.46,
                    memory_usage=0.43,
                    saturation=0.77,
                )
                dependency_health["auth-api->main-db"] = "partitioned"
                self.world["smoke_tests"]["auth-api"] = "FAIL /login from zone-a times out on upstream DB path"

    def _investigation_coverage(self) -> float:
        relevant = set(self.world["hidden"]["relevant_evidence"])
        if not relevant:
            return 1.0
        discovered = set(self.world["hidden"]["evidence"])
        return len(relevant & discovered) / len(relevant)

    def _active_causes(self) -> list[str]:
        hidden = self.world["hidden"]
        active = []
        if hidden.get("bad_auth_deploy_active", False):
            active.append("bad_auth_deploy")
        if hidden.get("memory_leak_active", False):
            active.append("worker_memory_leak")
        if hidden.get("db_pool_exhausted", False):
            active.append("db_pool_exhaustion")
        if hidden.get("shadow_reads_enabled", False):
            active.append("shadow_reads_flag")
        if hidden.get("billing_regression_active", False):
            active.append("billing_queue_regression")
        if hidden.get("network_partition_active", False):
            active.append("network_partition")
        return active

    def _fixed_causes(self) -> list[str]:
        root_causes = set(self.world["hidden"]["root_causes"])
        active = set(self._active_causes())
        return sorted(root_causes - active)

    def _recovery_confirmed(self) -> bool:
        signals = self._build_global_signals()
        return signals["customer_error_rate"] <= 0.05 and signals["customer_latency"] <= 0.18

    def _is_resolved(self) -> bool:
        task_id = self.task_id
        hidden = self.world["hidden"]
        if task_id == "worker_memory_leak":
            return not hidden.get("memory_leak_active", False) and hidden["queue_depth"] <= 120 and self._recovery_confirmed()
        if task_id == "db_pool_exhaustion":
            return hidden.get("db_scaled", False) and hidden.get("connections_cleared", False) and self._recovery_confirmed()
        if task_id == "queue_backlog_regression":
            return not hidden.get("billing_regression_active", False) and hidden["queue_depth"] <= 120
        if task_id == "network_partition_failover":
            return not hidden.get("network_partition_active", False) and hidden.get("db_failed_over", False)
        return not self._active_causes() and self._recovery_confirmed()

    def _handle_investigation(self, action: IncidentAction, breakdown: RewardBreakdown) -> None:
        command = action.command
        target = action.target
        hidden = self.world["hidden"]
        marker = f"{command}:{target}"
        if marker not in hidden["investigations"]:
            hidden["investigations"].append(marker)
        else:
            self._mark_no_effect(breakdown, f"Repeated investigation {marker} did not add new evidence.")

        if command == "get_recent_deploys":
            deploy_line = "; ".join(self.world["deploy_history"][-2:]) if self.world["deploy_history"] else "No recent deploys recorded."
            self._append_event(f"Deploy history inspected: {deploy_line}")
            if self.task_id == "bad_auth_deploy":
                self._add_fact("Recent auth-api rollout lines up with the incident start.", "deploy_timeline", breakdown)
            elif self.task_id == "queue_backlog_regression":
                self._add_fact("Billing deploy happened shortly before queue failures started.", "billing_regression", breakdown)
            elif self.task_id == "cache_flag_storm":
                self._add_fact("Rollout history shows a feature flag change right before latency spiked.", "flag_shadow_reads", breakdown)
        elif command == "query_logs":
            log_lines = self.world["logs"].get(target, [])
            preview = " | ".join(log_lines[:2]) if log_lines else f"No suspicious logs for {target}."
            self._append_event(f"Logs inspected for {target}: {preview}")
            if self.task_id == "bad_auth_deploy" and target == "auth-api":
                self._add_fact("Auth logs tie the regression to the latest auth-api build.", "auth_regression_logs", breakdown)
            elif self.task_id == "worker_memory_leak" and target == "worker":
                self._add_fact("Worker logs show runaway memory growth and allocator churn.", "worker_memory_growth", breakdown)
            elif self.task_id == "worker_memory_leak" and target == "queue":
                self._add_fact("Queue logs confirm backlog growth outpacing workers.", "queue_backlog", breakdown)
            elif self.task_id == "db_pool_exhaustion" and target == "main-db":
                self._add_fact("DB logs show saturated pool slots and stale sessions.", "stale_connections", breakdown)
            elif self.task_id == "cache_flag_storm" and target in {"auth-api", "cache"}:
                self._add_fact("Logs show request fanout and cache misses spiking after the rollout.", "cache_miss_spike", breakdown)
            elif self.task_id == "queue_backlog_regression" and target == "billing-api":
                self._add_fact("Billing logs show malformed queue payloads after the last release.", "billing_regression", breakdown)
            elif self.task_id == "queue_backlog_regression" and target == "queue":
                self._add_fact("Queue logs show repeated poison-message retries.", "poison_messages", breakdown)
            elif self.task_id == "network_partition_failover" and target == "auth-api":
                self._add_fact("Auth logs show failures isolated to one zone, not all requests.", "zone_specific_outage", breakdown)
        elif command == "get_metrics":
            service = self.world["services"].get(target)
            if service is None:
                self._append_event(f"Metrics probe for {target} returned no data.")
            else:
                self._append_event(
                    f"Metrics for {target}: error_rate={service['error_rate']:.2f}, latency={service['latency']:.2f}, saturation={service['saturation']:.2f}"
                )
                if self.task_id == "worker_memory_leak" and target == "worker":
                    self._add_fact("Worker metrics confirm memory is pinned near the cgroup limit.", "worker_memory_growth", breakdown)
                elif self.task_id == "worker_memory_leak" and target == "queue":
                    self._add_fact("Queue depth metrics confirm backlog is compounding.", "queue_backlog", breakdown)
                elif self.task_id == "db_pool_exhaustion" and target == "main-db":
                    self._add_fact("DB metrics confirm pool saturation, not raw compute exhaustion.", "db_pool_saturation", breakdown)
                elif self.task_id == "cache_flag_storm" and target == "cache":
                    self._add_fact("Cache metrics show the hit rate collapsed after the rollout.", "cache_miss_spike", breakdown)
                elif self.task_id == "queue_backlog_regression" and target == "queue":
                    self._add_fact("Queue metrics show poison retries dominating throughput.", "poison_messages", breakdown)
        elif command == "inspect_config":
            config = self.world["config"].get(target, {})
            rendered = ", ".join(f"{key}={value}" for key, value in config.items()) if config else "no config found"
            self._append_event(f"Config inspected for {target}: {rendered}")
            if self.task_id == "cache_flag_storm" and target == "auth-api":
                self._add_fact("Auth config shows shadow_reads is enabled globally.", "flag_shadow_reads", breakdown)
            elif self.task_id == "db_pool_exhaustion" and target == "main-db":
                self._add_fact("DB config shows the pool limit is normal, so saturation is likely stale-session related.", "stale_connections", breakdown)
        elif command == "check_dependency":
            if target == "auth-api":
                health = self.world["dependency_health"].get("auth-api->main-db", "healthy")
                self._append_event(f"Dependency check for auth-api -> main-db: {health}")
                if self.task_id == "network_partition_failover":
                    self._add_fact("Dependency health shows a partition on the auth-api -> main-db path.", "dependency_partition", breakdown)
                elif self.task_id == "db_pool_exhaustion":
                    self._add_fact("Dependency health shows auth-api is blocked on the DB path, not cache.", "db_pool_saturation", breakdown)
            else:
                self._append_event(f"Dependency check for {target} returned healthy downstreams.")
        elif command == "run_smoke_test":
            result = self.world["smoke_tests"].get(target, "No smoke test configured")
            self._append_event(f"Smoke test for {target}: {result}")
            if "PASS" in result and self._is_resolved():
                breakdown.investigation += 0.05

    def _handle_mitigation(self, action: IncidentAction, breakdown: RewardBreakdown) -> None:
        command = action.command
        target = action.target
        hidden = self.world["hidden"]
        evidence = set(hidden["evidence"])

        if command == "post_status_update":
            if not hidden["communication_sent"]:
                hidden["communication_sent"] = True
                breakdown.communication += 0.05
                self._append_event("Status update posted with incident scope, mitigation, and next checkpoint.")
            else:
                self._mark_no_effect(breakdown, "Repeated status update added no new operational value.")
            return

        if command == "rollback_deploy" and target == "auth-api":
            if hidden.get("bad_auth_deploy_active", False):
                self._schedule_effect("rollback_auth_deploy", 1, "Rollback requested for auth-api; waiting for deployment propagation.")
                breakdown.mitigation += 0.35
                if "deploy_timeline" in evidence or "auth_regression_logs" in evidence:
                    breakdown.mitigation += 0.05
            else:
                self._mark_no_effect(breakdown, "Rollback requested for auth-api, but no active auth deploy regression remains.")
            return

        if command == "restart_service" and target == "worker":
            if hidden.get("memory_leak_active", False):
                hidden["memory_leak_active"] = False
                breakdown.mitigation += 0.35
                self._append_event("Worker restart completed; leaking process was replaced with a clean worker.")
            else:
                self._mark_no_effect(breakdown, "Worker restart completed, but it did not address the current incident.")
            return

        if command == "scale_service" and target == "worker":
            self.world["services"]["worker"]["replicas"] += 1
            if self.task_id in {"worker_memory_leak", "queue_backlog_regression"}:
                breakdown.mitigation += 0.15
                self._append_event("Worker fleet scaled out to increase queue drain capacity.")
            else:
                self._mark_no_effect(breakdown, "Worker fleet scaled out, but it was not the bottleneck.")
            return

        if command == "drain_queue" and target == "queue":
            queue_depth = hidden.get("queue_depth", 0)
            if queue_depth > 0:
                hidden["queue_depth"] = max(0, queue_depth - 500)
                breakdown.mitigation += 0.2 if queue_depth > 300 else 0.1
                self._append_event("Queue drain executed to flush the highest-priority backlog.")
            else:
                self._mark_no_effect(breakdown, "Queue drain executed, but the queue was already nominal.")
            return

        if command == "scale_service" and target == "main-db":
            if self.task_id == "db_pool_exhaustion" and not hidden.get("db_scaled", False):
                hidden["db_scaled"] = True
                self.world["services"]["main-db"]["replicas"] += 1
                breakdown.mitigation += 0.25
                self._append_event("Main DB scaled out to add more pool capacity.")
            else:
                self._mark_no_effect(breakdown, "DB scale request added capacity, but it did not improve the active incident.")
            return

        if command == "clear_connections" and target == "main-db":
            if self.task_id == "db_pool_exhaustion":
                if hidden.get("db_scaled", False):
                    hidden["connections_cleared"] = True
                    hidden["db_pool_exhausted"] = False
                    breakdown.mitigation += 0.35
                    self._append_event("Stale DB sessions were cleared after capacity recovery.")
                else:
                    self._mark_no_effect(
                        breakdown,
                        "Connection clear ran before DB capacity was scaled; stale sessions quickly re-formed.",
                    )
            else:
                self._mark_no_effect(breakdown, "Connection clear ran on main-db, but the incident was elsewhere.")
            return

        if command == "disable_feature_flag" and target == "auth-api":
            if self.task_id == "cache_flag_storm" and hidden.get("shadow_reads_enabled", False):
                self._schedule_effect("disable_shadow_reads", 1, "Feature flag disable requested; waiting for config rollout.")
                breakdown.mitigation += 0.35
                if "flag_shadow_reads" in evidence:
                    breakdown.mitigation += 0.05
            else:
                self._mark_no_effect(breakdown, "Feature flag change on auth-api did not match the active issue.")
            return

        if command == "rollback_deploy" and target == "billing-api":
            if self.task_id == "queue_backlog_regression" and hidden.get("billing_regression_active", False):
                self._schedule_effect("rollback_billing_deploy", 1, "Rollback requested for billing-api to stop malformed queue writes.")
                breakdown.mitigation += 0.35
                if "billing_regression" in evidence:
                    breakdown.mitigation += 0.05
            else:
                self._mark_no_effect(breakdown, "Rollback requested for billing-api, but there is no active billing regression.")
            return

        if command == "failover_db" and target == "main-db":
            if self.task_id == "network_partition_failover" and hidden.get("network_partition_active", False):
                if "dependency_partition" in evidence:
                    hidden["network_partition_active"] = False
                    hidden["db_failed_over"] = True
                    breakdown.mitigation += 0.45
                    self._append_event("DB failover completed; auth-api traffic moved to the healthy zone.")
                else:
                    hidden["unsafe_actions"] += 1
                    hidden["temp_failover_risk"] = 2
                    breakdown.safety -= 0.35
                    self._append_event("Blind DB failover triggered extra instability because the partition was not confirmed.")
            else:
                self._mark_no_effect(breakdown, "DB failover ran without an active zonal partition.", risky=True)
            return

        if command == "restart_service" and target == "auth-api":
            self._mark_no_effect(breakdown, "Auth-api restarted, but the unhealthy dependency or bad code path remained.")
            return

        self._mark_no_effect(breakdown, f"Action {command}({target}) had no meaningful corrective effect.", risky=command in self.HIGH_RISK_COMMANDS)

    def step(self, action: Any) -> str:
        """OpenEnv-compatible step returning a string observation payload."""
        if isinstance(action, IncidentAction):
            parsed_action = action
        elif isinstance(action, dict):
            parsed_action = IncidentAction(**action)
        else:
            parsed_action = IncidentAction(command=str(action), target="system")
        observation, reward, done, info = self.step_episode(parsed_action)
        return json.dumps(
            {
                "observation": observation.model_dump(),
                "reward": reward.model_dump(),
                "done": done,
                "info": info,
            }
        )

    def step_episode(self, action: IncidentAction):
        if self.done:
            info = dict(self.last_info)
            info["last_action_error"] = "episode_already_done"
            reward = RewardOutput(
                value=0.0,
                reason="episode_already_done",
                breakdown=RewardBreakdown(),
                done=True,
            )
            return self._get_obs(), reward, True, info

        self.step_count += 1
        breakdown = new_breakdown()
        marker = f"{action.command}({action.target})"
        self.world["hidden"]["action_history"].append(marker)

        self._advance_pending_effects()
        self._scenario_tick()

        if action.command in self.INVESTIGATION_COMMANDS:
            self._handle_investigation(action, breakdown)
        elif action.command in self.MITIGATION_COMMANDS:
            self._handle_mitigation(action, breakdown)
        else:
            self._mark_no_effect(breakdown, f"Unknown command '{action.command}' was ignored.")

        self._refresh_world()

        resolved = self._is_resolved()
        if resolved:
            breakdown.recovery += 1.0
            if self._recovery_confirmed():
                self._append_event("Customer-impact metrics returned to nominal levels.")

        done = resolved or self.step_count >= self.max_steps
        reward_value = total_reward(breakdown)
        reward_reason = "incident_resolved" if resolved else "progress_or_penalty"
        info = {
            "session_id": self.session_id,
            "task_id": self.task_id,
            "difficulty": self.world["difficulty"],
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "unsafe_actions": self.world["hidden"]["unsafe_actions"],
            "ineffective_actions": self.world["hidden"]["ineffective_actions"],
            "evidence_count": len(self.world["hidden"]["evidence"]),
            "active_causes": len(self._active_causes()),
            "resolved": resolved,
            "seed": self.seed,
        }

        self.done = done
        self.last_info = info
        self.last_reward = RewardOutput(
            value=reward_value,
            reason=reward_reason,
            breakdown=breakdown,
            done=done,
        )
        self.reward_history.append(self.last_reward)
        self.cumulative_reward = round(self.cumulative_reward + reward_value, 2)
        return self._get_obs(), self.last_reward, done, info

    def grade(self):
        metadata = SCENARIO_METADATA[self.task_id]
        resolved = self._is_resolved()
        root_cause_fixed = not self._active_causes()
        recovery_confirmed = self._recovery_confirmed()
        notes = []
        if resolved:
            notes.append("Incident resolved and customer-facing signals recovered.")
        elif root_cause_fixed:
            notes.append("Primary cause was fixed, but customer-facing recovery is incomplete.")
        else:
            notes.append("Active root cause remains unresolved.")
        if self.world["hidden"]["unsafe_actions"] > 0:
            notes.append("Unsafe actions were attempted during remediation.")
        if not self.world["hidden"]["communication_sent"]:
            notes.append("No stakeholder status update was posted.")
        return build_grade(
            task_id=self.task_id,
            scenario_family=metadata.name,
            resolved=resolved,
            root_cause_fixed=root_cause_fixed,
            recovery_confirmed=recovery_confirmed,
            steps_taken=self.step_count,
            max_steps=self.max_steps,
            unsafe_actions=self.world["hidden"]["unsafe_actions"],
            ineffective_actions=self.world["hidden"]["ineffective_actions"],
            investigation_coverage=self._investigation_coverage(),
            communication_sent=self.world["hidden"]["communication_sent"],
            reward_total=self.cumulative_reward,
            root_causes=list(self.world["hidden"]["root_causes"]),
            fixed_causes=self._fixed_causes(),
            notes=notes,
        )

    @property
    def state(self):
        return State(episode_id=self.session_id, step_count=self.step_count)

    @property
    def debug_state(self):
        hidden = self.world["hidden"]
        return EnvironmentState(
            task_id=self.task_id,
            difficulty=self.world["difficulty"],
            observation=self._get_obs(),
            hidden_state=HiddenIncidentState(
                root_causes=list(hidden["root_causes"]),
                active_causes=self._active_causes(),
                fixed_causes=self._fixed_causes(),
                evidence=list(hidden["evidence"]),
                dependencies={service: list(children) for service, children in self.world["dependencies"].items()},
                deploy_history=list(self.world["deploy_history"]),
                config={service: dict(values) for service, values in self.world["config"].items()},
                dependency_health=dict(self.world["dependency_health"]),
                pending_effects=[effect["detail"] for effect in hidden["pending_effects"]],
                service_graph=list(hidden["service_graph"]),
                unsafe_actions=hidden["unsafe_actions"],
                ineffective_actions=hidden["ineffective_actions"],
                communication_sent=hidden["communication_sent"],
                scenario_flags={
                    key: value
                    for key, value in hidden.items()
                    if key
                    not in {
                        "root_causes",
                        "relevant_evidence",
                        "evidence",
                        "discovered_facts",
                        "investigations",
                        "communication_sent",
                        "unsafe_actions",
                        "ineffective_actions",
                        "action_history",
                        "pending_effects",
                        "service_graph",
                    }
                },
            ),
            step_count=self.step_count,
            max_steps=self.max_steps,
            cumulative_reward=self.cumulative_reward,
            reward_history=self.reward_history,
            done=self.done,
            info=self.last_info,
        )

    @property
    def public_state(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "task_id": self.task_id,
            "difficulty": self.world["difficulty"],
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "done": self.done,
            "cumulative_reward": self.cumulative_reward,
            "observation": self._get_obs().model_dump(),
            "metrics": {
                "unsafe_actions": self.world["hidden"]["unsafe_actions"],
                "ineffective_actions": self.world["hidden"]["ineffective_actions"],
                "investigation_coverage": round(self._investigation_coverage(), 4),
                "communication_sent": self.world["hidden"]["communication_sent"],
            },
        }
