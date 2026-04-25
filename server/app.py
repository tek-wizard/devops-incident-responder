from __future__ import annotations

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool

from .models import EnvironmentState, IncidentAction, RewardOutput, SystemObservation, TaskGrade
from .scenarios import DEFAULT_TASK_ID, TASKS
from .seed_sets import SEED_SETS
from .session_store import EnvSessionStore, SessionNotFoundError

app = FastAPI(title="AI Incident Commander")
store = EnvSessionStore()


@app.get("/")
async def root():
    return {"message": "AI Incident Commander OpenEnv is running", "status": "200"}


@app.get("/health")
async def health():
    return {"status": "ok"}


def _normalize_action(action: IncidentAction) -> IncidentAction:
    command_aliases = {
        "restart": "restart_service",
        "scale": "scale_service",
        "get_logs": "query_logs",
        "check_metrics": "get_metrics",
    }
    target_aliases = {
        "system": "system",
        "main-db": "main-db",
        "db": "main-db",
    }
    normalized_command = command_aliases.get(action.command, action.command)
    normalized_target = target_aliases.get(action.target, action.target)
    return IncidentAction(
        session_id=action.session_id,
        command=normalized_command,
        target=normalized_target,
        params=action.params,
    )


def _get_env(session_id: str | None = None):
    try:
        return store.get(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/reset")
async def reset(task_id: str = DEFAULT_TASK_ID, seed: int | None = None, session_id: str | None = None):
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail=f"Unknown task_id '{task_id}'")
    resolved_id, env = store.reset(task_id=task_id, seed=seed, session_id=session_id)
    env.session_id = resolved_id
    return env._get_obs()


@app.post("/step")
async def step(action: dict):
    raw_action = action.get("action", action)
    normalized_action = _normalize_action(IncidentAction(**raw_action))
    env = _get_env(normalized_action.session_id)
    observation, reward, done, info = env.step_episode(normalized_action)
    return {
        "observation": observation,
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/tasks")
async def get_tasks():
    return {
        "tasks": [
            {
                "id": task["id"],
                "name": task["name"],
                "difficulty": task["difficulty"],
                "description": task["description"],
                "incident_brief": task["incident_brief"],
                "max_steps": task["max_steps"],
                "root_causes": task["root_causes"],
                "relevant_evidence": task["relevant_evidence"],
            }
            for task in TASKS.values()
        ],
        "seed_sets": SEED_SETS,
        "action_schema": IncidentAction.model_json_schema(),
        "observation_schema": SystemObservation.model_json_schema(),
        "reward_schema": RewardOutput.model_json_schema(),
        "state_schema": EnvironmentState.model_json_schema(),
        "grader_schema": TaskGrade.model_json_schema(),
    }


@app.get("/grader")
async def get_grader(session_id: str | None = None):
    env = _get_env(session_id)
    return env.grade()


@app.get("/state")
async def get_state(session_id: str | None = None):
    env = _get_env(session_id)
    return env.public_state


@app.post("/baseline")
async def run_baseline():
    from scripts.baseline import run_all_tasks

    results = await run_in_threadpool(run_all_tasks)
    return {"results": results}


def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
