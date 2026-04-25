from __future__ import annotations

from uuid import uuid4

from .environment import DevOpsEnv


class SessionNotFoundError(KeyError):
    pass


class EnvSessionStore:
    def __init__(self):
        self._sessions: dict[str, DevOpsEnv] = {}
        self.default_session_id: str | None = None

    def reset(self, *, task_id: str, seed: int | None = None, session_id: str | None = None) -> tuple[str, DevOpsEnv]:
        resolved_id = session_id or str(uuid4())
        env = self._sessions.get(resolved_id)
        if env is None:
            env = DevOpsEnv(session_id=resolved_id)
            self._sessions[resolved_id] = env
        env.session_id = resolved_id
        env.reset_episode(task_id=task_id, seed=seed)
        self.default_session_id = resolved_id
        return resolved_id, env

    def get(self, session_id: str | None = None) -> DevOpsEnv:
        resolved_id = session_id or self.default_session_id
        if resolved_id is None or resolved_id not in self._sessions:
            raise SessionNotFoundError("No active session found. Call /reset first.")
        self.default_session_id = resolved_id
        return self._sessions[resolved_id]
