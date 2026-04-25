from __future__ import annotations

import unittest

from server.environment import DevOpsEnv
from server.models import IncidentAction


class RewardTests(unittest.TestCase):
    def test_resolving_incident_yields_positive_reward(self) -> None:
        env = DevOpsEnv()
        env.reset_episode(task_id="bad_auth_deploy", seed=101)
        env.step_episode(IncidentAction(command="get_recent_deploys", target="system"))
        env.step_episode(IncidentAction(command="query_logs", target="auth-api"))
        _, reward, done, _ = env.step_episode(IncidentAction(command="rollback_deploy", target="auth-api"))
        self.assertFalse(done)
        _, reward, done, _ = env.step_episode(IncidentAction(command="run_smoke_test", target="auth-api"))
        self.assertTrue(done)
        self.assertGreater(reward.value, 0.5)

    def test_redundant_investigation_is_penalized(self) -> None:
        env = DevOpsEnv()
        env.reset_episode(task_id="bad_auth_deploy", seed=101)
        env.step_episode(IncidentAction(command="query_logs", target="auth-api"))
        _, reward, _, info = env.step_episode(IncidentAction(command="query_logs", target="auth-api"))
        self.assertLess(reward.value, 0.0)
        self.assertGreaterEqual(info["ineffective_actions"], 1)


if __name__ == "__main__":
    unittest.main()
