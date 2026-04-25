from __future__ import annotations

import json
import unittest

from openenv_core import Environment
from openenv_core.env_server.types import State

from server.environment import DevOpsEnv
from server.models import IncidentAction


class OpenEnvComplianceTests(unittest.TestCase):
    def test_environment_inherits_openenv_base(self) -> None:
        self.assertTrue(issubclass(DevOpsEnv, Environment))

    def test_reset_accepts_kwargs_and_returns_string(self) -> None:
        env = DevOpsEnv()
        observation = env.reset(task_id="bad_auth_deploy", seed=101, extra_column="ignored")
        self.assertIsInstance(observation, str)
        self.assertEqual(json.loads(observation)["task_id"], "bad_auth_deploy")

    def test_step_returns_string_observation_payload(self) -> None:
        env = DevOpsEnv()
        env.reset(task_id="bad_auth_deploy", seed=101)
        payload = env.step(IncidentAction(command="query_logs", target="auth-api"))
        self.assertIsInstance(payload, str)
        self.assertIn("observation", json.loads(payload))

    def test_state_is_openenv_state(self) -> None:
        env = DevOpsEnv(session_id="compliance")
        self.assertIsInstance(env.state, State)
        self.assertEqual(env.state.episode_id, "compliance")


if __name__ == "__main__":
    unittest.main()
