import random
from copy import deepcopy

from .models import EnvironmentState, IncidentAction, RewardOutput, SystemObservation, SystemState
from .tasks import TASKS


class DevOpsEnv:
    STEP_PENALTY = -0.05
    INCORRECT_ACTION_PENALTY = -0.2
    PARTIAL_REWARD = 0.5
    SUCCESS_REWARD = 1.0

    NOISE_LOGS = [
        "[INFO] node-exporter scrape completed successfully",
        "[INFO] kubelet health probe passed on node-01",
        "[INFO] cache hit ratio stable over the last minute",
        "[INFO] background cron job finished with exit code 0",
        "[INFO] disk utilization remains below 55%",
        "[INFO] TLS certificate check passed for internal services",
        "[INFO] synthetic canary passed from edge-us-east-1",
        "[INFO] cluster autoscaler observed no pending pods",
    ]

    MEDIUM_SIGNAL_LOGS = [
        "[WARN] worker-process RSS exceeded 95% of cgroup limit",
        "[ERROR] OOM score increasing for worker-process",
        "[WARN] auth-api queue depth rising while worker drains slowly",
        "[ERROR] kernel memory pressure event detected on node-01",
    ]

    HARD_SIGNAL_LOGS = [
        "[WARN] p95 latency breached service objective for auth-api",
        "[INFO] auth-api pod health checks still reporting healthy",
        "[INFO] database CPU nominal despite sustained request slowness",
        "[WARN] client retries increasing without corresponding application exceptions",
    ]

    def __init__(self):
        self.reset()

    def reset(self, task_id: str = "service_restart", seed: int | None = None):
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id '{task_id}'")

        self.step_count = 0
        self.task_id = task_id
        self.task = deepcopy(TASKS[task_id])
        self.done = False
        self.last_info = {}
        self.last_reward = RewardOutput(value=0.0, reason="environment_reset", done=False)
        self.random = random.Random(seed) if seed is not None else random.Random()
        self.seed = seed

        initial_state = self.task["initial_state"]
        hidden_state = initial_state["state"]

        self.service_status = deepcopy(initial_state["services"])
        self.metrics = deepcopy(initial_state["metrics"])
        self.logs = list(initial_state["logs"])

        self.hidden_state = SystemState(
            internal_error_rate=hidden_state["internal_error_rate"],
            is_database_locked=hidden_state["is_database_locked"],
            active_incidents=list(hidden_state["active_incidents"]),
        )

        self.required_sequence = list(self.task["required_sequence"])
        self.progress_index = 0
        self.partial_reward_given = False
        self.flakiness_rate = self.random.uniform(*self.task["flakiness_range"])

        self.leak_active = hidden_state["leak_active"]
        self.db_pool_exhausted = hidden_state["db_pool_exhausted"]
        self.db_scaled = hidden_state["db_scaled"]
        self.connections_cleared = hidden_state["connections_cleared"]

        self._refresh_state()
        self._append_telemetry_logs()
        return self._get_obs()

    def _get_obs(self):
        return SystemObservation(
            logs=self.logs[-5:],
            metrics=self.metrics.copy(),
            service_status=self.service_status.copy(),
            step_count=self.step_count,
        )

    def _current_expected_step(self):
        if self.progress_index >= len(self.required_sequence):
            return None
        return self.required_sequence[self.progress_index]

    def _is_correct_action(self, action: IncidentAction, expected_step: str | None):
        if expected_step == "restart_service":
            return action.command == "restart" and action.target == "auth-api"
        if expected_step == "kill_process":
            return action.command == "kill_process"
        if expected_step == "scale_db":
            return action.command in {"scale_db", "scale"} and action.target == "main-db"
        if expected_step == "clear_connections":
            return action.command == "clear_connections" and action.target == "main-db"
        return False

    def _matches_any_future_step(self, action: IncidentAction):
        for future_step in self.required_sequence[self.progress_index + 1 :]:
            if self._is_correct_action(action, future_step):
                return future_step
        return None

    def _apply_flakiness(self):
        return self.random.random() < self.flakiness_rate

    def _grant_partial_reward_if_needed(self):
        if len(self.required_sequence) > 1 and self.progress_index == 1 and not self.partial_reward_given:
            self.partial_reward_given = True
            return self.PARTIAL_REWARD
        return 0.0

    def _complete_expected_step(self, expected_step: str):
        if expected_step == "restart_service":
            self.service_status["auth-api"] = "RUNNING"
            self.logs.append("auth-api restarted successfully.")
        elif expected_step == "kill_process":
            self.leak_active = False
            self.service_status["auth-api"] = "DOWN"
            self.logs.append("Leaking worker process terminated. Service restart still required.")
        elif expected_step == "scale_db":
            self.db_scaled = True
            self.service_status["main-db"] = "RUNNING"
            self.logs.append("Database pool capacity scaled out. Existing stale connections still present.")
        elif expected_step == "clear_connections":
            self.connections_cleared = True
            self.db_pool_exhausted = False
            self.hidden_state.is_database_locked = False
            self.logs.append("Database connections cleared after capacity increase. Pool recovered.")

        self.progress_index += 1
        return self._grant_partial_reward_if_needed()

    def _handle_inspection_action(self, action: IncidentAction):
        if action.command in {"get_logs", "check_logs"}:
            self.logs.append("Recent logs inspected.")
            return 0.0, True, False
        if action.command in {"check_metrics", "get_metrics"}:
            self.logs.append(
                f"Metrics snapshot: error_rate={self.metrics['error_rate']:.2f}, latency={self.metrics['latency']:.2f}"
            )
            return 0.0, True, False
        return 0.0, False, False

    def _execute_action(self, action: IncidentAction):
        expected_step = self._current_expected_step()

        inspection_reward, handled, flaky_failure = self._handle_inspection_action(action)
        if handled:
            return inspection_reward, True, flaky_failure

        if self._is_correct_action(action, expected_step):
            if self._apply_flakiness():
                self.logs.append(
                    f"[WARN] Action '{action.command}' against '{action.target}' did not stick due to transient control-plane failure."
                )
                return 0.0, False, True

            reward = self._complete_expected_step(expected_step)
            return reward, True, False

        out_of_order_step = self._matches_any_future_step(action)
        if out_of_order_step:
            self.logs.append(
                f"[WARN] Action '{action.command}' was attempted out of order. '{expected_step}' must happen first."
            )
            return self.INCORRECT_ACTION_PENALTY, False, False

        self.logs.append(f"[WARN] Action '{action.command}' on '{action.target}' had no corrective effect.")
        return self.INCORRECT_ACTION_PENALTY, False, False

    def _set_nominal_metrics(self):
        nominal = self.task["nominal_metrics"]
        self.metrics["cpu_load"] = nominal["cpu_load"]
        self.metrics["memory_usage"] = nominal["memory_usage"]
        self.metrics["latency"] = nominal["latency"]
        self.metrics["error_rate"] = nominal["error_rate"]
        self.hidden_state.internal_error_rate = nominal["error_rate"]

    def _refresh_state(self):
        if self.task_id == "service_restart":
            if self.progress_index >= len(self.required_sequence):
                self._set_nominal_metrics()
                self.service_status["auth-api"] = "RUNNING"
                self.hidden_state.active_incidents = []
            else:
                self.service_status["auth-api"] = "DOWN"
                self.metrics["cpu_load"] = 0.22
                self.metrics["memory_usage"] = 0.33
                self.metrics["latency"] = 0.78
                self.metrics["error_rate"] = 0.91
                self.hidden_state.internal_error_rate = 0.91
                self.hidden_state.active_incidents = ["auth_api_process_crash"]

        elif self.task_id == "memory_leak":
            if self.progress_index >= len(self.required_sequence):
                self._set_nominal_metrics()
                self.service_status["auth-api"] = "RUNNING"
                self.hidden_state.active_incidents = []
            elif self.leak_active:
                self.service_status["auth-api"] = "DEGRADED"
                self.metrics["cpu_load"] = 0.64
                self.metrics["memory_usage"] = 0.97
                self.metrics["latency"] = 0.83
                self.metrics["error_rate"] = 0.42
                self.hidden_state.internal_error_rate = 0.42
                self.hidden_state.active_incidents = ["worker_memory_leak"]
            else:
                self.service_status["auth-api"] = "DOWN"
                self.metrics["cpu_load"] = 0.31
                self.metrics["memory_usage"] = 0.42
                self.metrics["latency"] = 0.28
                self.metrics["error_rate"] = 0.18
                self.hidden_state.internal_error_rate = 0.18
                self.hidden_state.active_incidents = ["service_restart_pending"]

        elif self.task_id == "db_connection_exhaustion":
            if self.progress_index >= len(self.required_sequence):
                self._set_nominal_metrics()
                self.hidden_state.is_database_locked = False
                self.service_status["main-db"] = "RUNNING"
                self.hidden_state.active_incidents = []
            elif not self.db_scaled:
                self.service_status["main-db"] = "DEGRADED"
                self.metrics["cpu_load"] = 0.46
                self.metrics["memory_usage"] = 0.58
                self.metrics["latency"] = 0.97
                self.metrics["error_rate"] = 0.37
                self.hidden_state.internal_error_rate = 0.37
                self.hidden_state.is_database_locked = True
                self.hidden_state.active_incidents = ["db_connection_pool_exhaustion"]
            else:
                self.service_status["main-db"] = "RUNNING"
                self.metrics["cpu_load"] = 0.41
                self.metrics["memory_usage"] = 0.54
                self.metrics["latency"] = 0.42
                self.metrics["error_rate"] = 0.14
                self.hidden_state.internal_error_rate = 0.14
                self.hidden_state.is_database_locked = True
                self.hidden_state.active_incidents = ["stale_connection_pool_state"]

    def _append_telemetry_logs(self):
        for _ in range(2):
            if self.random.random() < self.task["noise_level"]:
                self.logs.append(self.random.choice(self.NOISE_LOGS))
            else:
                if self.task_id == "memory_leak":
                    self.logs.append(self.random.choice(self.MEDIUM_SIGNAL_LOGS))
                elif self.task_id == "db_connection_exhaustion":
                    self.logs.append(self.random.choice(self.HARD_SIGNAL_LOGS))

    def _is_resolved(self):
        if self.task_id == "db_connection_exhaustion":
            return self.progress_index == 2
        return self.progress_index == len(self.required_sequence)

    def step(self, action: IncidentAction):
        self.step_count += 1
        reward = self.STEP_PENALTY
        max_steps = self.task.get("max_steps", 15)

        action_reward, action_successful, flaky_failure = self._execute_action(action)
        reward += action_reward

        self._refresh_state()
        self._append_telemetry_logs()

        resolved = self._is_resolved()
        if resolved:
            reward += self.SUCCESS_REWARD
            self.logs.append("[RECOVERY] error_rate and latency returned to nominal levels.")

        done = resolved or self.step_count >= max_steps
        reward_reason = "nominal_recovery" if resolved else "progress_or_penalty"
        info = {
            "task_id": self.task_id,
            "difficulty": self.task["difficulty"],
            "max_steps": max_steps,
            "required_sequence": self.required_sequence,
            "progress_index": self.progress_index,
            "current_required_action": self._current_expected_step(),
            "action_successful": action_successful,
            "flaky_action_failed": flaky_failure,
            "flakiness_rate": round(self.flakiness_rate, 2),
            "seed": self.seed,
        }
        reward_value = round(reward, 2)
        self.done = done
        self.last_info = info
        self.last_reward = RewardOutput(value=reward_value, reason=reward_reason, done=done)
        return self._get_obs(), reward_value, done, info

    @property
    def state(self):
        return EnvironmentState(
            task_id=self.task_id,
            difficulty=self.task["difficulty"],
            observation=self._get_obs(),
            hidden_state=self.hidden_state,
            progress_index=self.progress_index,
            required_sequence=self.required_sequence,
            flakiness_rate=round(self.flakiness_rate, 4),
            done=self.done,
            info=self.last_info,
        )
