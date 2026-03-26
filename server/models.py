from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

class IncidentAction(BaseModel):
    """What the agent can do."""
    command: str = Field(
        description="Action command such as restart_service, restart, kill_process, scale_db, clear_connections, get_logs, or check_metrics."
    )
    target: str = Field(description="Action target such as auth-api, main-db, or system.")
    params: Optional[Dict[str, str]] = None


class SystemObservation(BaseModel):
    """What the agent sees."""
    logs: List[str]
    metrics: Dict[str, float]
    service_status: Dict[str, str]
    step_count: int


class SystemState(BaseModel):
    """The 'hidden' truth the grader uses."""
    internal_error_rate: float
    is_database_locked: bool
    active_incidents: List[str]


class RewardOutput(BaseModel):
    """Typed reward payload returned by step()."""
    value: float
    reason: str
    done: bool


class EnvironmentState(BaseModel):
    """Current environment state for debugging and validation."""
    task_id: str
    difficulty: str
    observation: SystemObservation
    hidden_state: SystemState
    progress_index: int
    required_sequence: List[str]
    flakiness_rate: float
    done: bool
    info: Dict[str, Any]
