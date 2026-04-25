from __future__ import annotations

import unittest

from server.session_store import EnvSessionStore


class SessionStoreTests(unittest.TestCase):
    def test_sessions_are_isolated(self) -> None:
        store = EnvSessionStore()
        first_id, first_env = store.reset(task_id="bad_auth_deploy", seed=101)
        second_id, second_env = store.reset(task_id="db_pool_exhaustion", seed=201)
        self.assertNotEqual(first_id, second_id)
        self.assertEqual(first_env.task_id, "bad_auth_deploy")
        self.assertEqual(second_env.task_id, "db_pool_exhaustion")


if __name__ == "__main__":
    unittest.main()
