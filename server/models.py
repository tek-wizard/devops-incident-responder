from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class IncidentAction(BaseModel):
    """Action requested by the agent."""

    session_id: Optional[str] = Field(
        default=None,
        description="Episode session identifier returned by reset().",
    )
    command: str = Field(
        description=(
            "Action command such as query_logs, get_metrics, get_recent_deploys, "
            "inspect_config, check_dependency, run_smoke_test, restart_service, "
            "rollback_deploy, disable_feature_flag, scale_service, clear_connections, "
            "drain_queue, failover_db, or post_status_update."
        )
    )
    target: str = Field(
        default="system",
        description="Action target such as auth-api, billing-api, worker, main-db, cache, queue, or system.",
    )
    params: Dict[str, str] = Field(default_factory=dict)


class ServiceSnapshot(BaseModel):
    status: str
    version: str
    replicas: int
    error_rate: float = Field(ge=0.0, le=1.0)
    latency: float = Field(ge=0.0)
    cpu_load: float = Field(ge=0.0, le=1.5)
    memory_usage: float = Field(ge=0.0, le=1.5)
    saturation: float = Field(ge=0.0, le=1.5)


class RewardBreakdown(BaseModel):
    step_penalty: float = 0.0
    investigation: float = 0.0
    mitigation: float = 0.0
    recovery: float = 0.0
    safety: float = 0.0
    communication: float = 0.0


class SystemObservation(BaseModel):
    """Agent-facing observation for the current step."""

    session_id: Optional[str] = None
    task_id: str
    difficulty: str
    incident_brief: str
    step_count: int
    remaining_steps: int
    global_signals: Dict[str, float]
    service_status: Dict[str, ServiceSnapshot]
    recent_events: List[str]
    discovered_facts: List[str]
    available_commands: List[str]


class HiddenIncidentState(BaseModel):
    """Debug / grader-facing state hidden from the acting agent."""

    root_causes: List[str]
    active_causes: List[str]
    fixed_causes: List[str]
    evidence: List[str]
    dependencies: Dict[str, List[str]]
    deploy_history: List[str]
    config: Dict[str, Dict[str, str]]
    dependency_health: Dict[str, str]
    pending_effects: List[str]
    service_graph: List[str]
    unsafe_actions: int
    ineffective_actions: int
    communication_sent: bool
    scenario_flags: Dict[str, Any]


class RewardOutput(BaseModel):
    """Reward payload returned by step()."""

    value: float
    reason: str
    breakdown: RewardBreakdown
    done: bool


class TaskGrade(BaseModel):
    """Episode grade used by baselines and evaluator."""

    task_id: str
    scenario_family: str
    score: float = Field(ge=0.0, le=1.0)
    resolved: bool
    root_cause_fixed: bool
    recovery_confirmed: bool
    steps_taken: int
    max_steps: int
    unsafe_actions: int
    ineffective_actions: int
    investigation_coverage: float = Field(ge=0.0, le=1.0)
    communication_sent: bool
    reward_total: float
    root_causes: List[str]
    fixed_causes: List[str]
    notes: List[str] = Field(default_factory=list)


class EnvironmentState(BaseModel):
    """Current full environment state."""

    task_id: str
    difficulty: str
    observation: SystemObservation
    hidden_state: HiddenIncidentState
    step_count: int
    max_steps: int
    cumulative_reward: float
    reward_history: List[RewardOutput]
    done: bool
    info: Dict[str, Any]
