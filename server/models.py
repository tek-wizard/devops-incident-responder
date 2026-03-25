from pydantic import BaseModel
from typing import List, Dict, Optional

class IncidentAction(BaseModel):
    """What the agent can do."""
    command: str  # e.g., "restart", "scale", "get_logs", "kill_process"
    target: str   # e.g., "auth-api", "pid-1024", "main-db"
    params: Optional[Dict[str, str]] = None

class SystemObservation(BaseModel):
    """What the agent sees."""
    logs: List[str]
    metrics: Dict[str, float]  # cpu_load, memory_usage, error_rate, latency
    service_status: Dict[str, str] # {"auth-api": "DOWN", "db": "RUNNING"}
    step_count: int

class SystemState(BaseModel):
    """The 'hidden' truth the grader uses."""
    internal_error_rate: float
    is_database_locked: bool
    active_incidents: List[str]