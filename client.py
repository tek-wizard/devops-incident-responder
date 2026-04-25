from __future__ import annotations

from typing import Any, Dict

import requests
from openenv_core.client_types import StepResult
from openenv_core.http_env_client import HTTPEnvClient


ActionType = Dict[str, Any]
ObservationType = Dict[str, Any]
StateType = Dict[str, Any]


class DevOpsClient(HTTPEnvClient[ActionType, ObservationType]):
    """OpenEnv HTTP client without importing server internals."""

    def __init__(self, base_url: str = "http://localhost:7860", **kwargs: Any):
        super().__init__(base_url=base_url, **kwargs)
        self.session_id: str | None = None

    def _step_payload(self, action: ActionType) -> dict:
        payload = dict(action)
        payload.setdefault("session_id", self.session_id)
        return payload

    def _parse_result(self, payload: dict) -> StepResult[ObservationType]:
        if "observation" in payload:
            observation = payload["observation"]
            reward_payload = payload.get("reward", {})
            reward = reward_payload.get("value") if isinstance(reward_payload, dict) else reward_payload
            done = bool(payload.get("done", False))
        else:
            observation = payload
            reward = None
            done = False
        if isinstance(observation, dict):
            self.session_id = observation.get("session_id", self.session_id)
        return StepResult(observation=observation, reward=reward, done=done)

    def _parse_state(self, payload: dict) -> StateType:
        return payload

    def reset(self, task_id: str = "bad_auth_deploy", seed: int | None = None) -> StepResult[ObservationType]:
        response = self._http.post(
            f"{self._base}/reset",
            params={"task_id": task_id, "seed": seed},
            headers=self._headers,
            timeout=self._timeout,
        )
        response.raise_for_status()
        return self._parse_result(response.json())

    def grader(self) -> Dict[str, Any]:
        response = requests.get(
            f"{self._base}/grader",
            params={"session_id": self.session_id},
            headers=self._headers,
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()
