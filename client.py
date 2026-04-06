import requests
from typing import Any, Dict

class DevOpsClient:
    """Standard client for the DevOps Incident Responder environment."""
    
    def __init__(self, url: str = "http://localhost:7860"):
        self.url = url.rstrip("/")

    def reset(self, task_id: str = "service_restart") -> Dict[str, Any]:
        response = requests.post(f"{self.url}/reset", params={"task_id": task_id})
        response.raise_for_status()
        return response.json()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(f"{self.url}/step", json=action)
        response.raise_for_status()
        return response.json()