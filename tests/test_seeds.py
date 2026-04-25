from __future__ import annotations

import unittest

from server.scenarios import build_scenario


class ScenarioSeedTests(unittest.TestCase):
    def test_same_seed_produces_same_deploy_history(self) -> None:
        first = build_scenario("bad_auth_deploy", seed=101)
        second = build_scenario("bad_auth_deploy", seed=101)
        self.assertEqual(first["deploy_history"], second["deploy_history"])
        self.assertEqual(first["services"]["auth-api"]["version"], second["services"]["auth-api"]["version"])


if __name__ == "__main__":
    unittest.main()
