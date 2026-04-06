from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool

from .models import EnvironmentState, IncidentAction, RewardOutput, SystemObservation, TaskGrade
from .environment import DevOpsEnv
from .tasks import TASKS
import uvicorn

app = FastAPI(title="DevOps Incident Responder")

env = DevOpsEnv()

@app.get("/")
async def root():
    return {"message": "DevOps Incident Responder OpenEnv is running", "status": "200"}


@app.get("/health")
async def health():
    return {"status": "ok"}


def _normalize_action(action: IncidentAction) -> IncidentAction:
    command_aliases = {
        "restart_service": "restart",
        "scale": "scale_db",
    }
    normalized_command = command_aliases.get(action.command, action.command)
    return IncidentAction(
        command=normalized_command,
        target=action.target,
        params=action.params,
    )


@app.post("/reset")
async def reset(task_id: str = "service_restart", seed: int | None = None):
    """Resets the environment to a specific task state."""
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail=f"Unknown task_id '{task_id}'")
    observation = env.reset(task_id=task_id, seed=seed)
    return observation

@app.post("/step")
async def step(action: IncidentAction):
    """Executes one action and returns the new observation and reward."""
    normalized_action = _normalize_action(action)
    observation, reward, done, info = env.step(normalized_action)
    return {
        "observation": observation,
        "reward": reward,
        "done": done,
        "info": info,
    }

@app.get("/tasks")
async def get_tasks():
    """MANDATORY: Returns list of tasks and the action schema."""
    return {
        "tasks": [
            {
                "id": task["id"],
                "name": task["name"],
                "difficulty": task["difficulty"],
                "noise_level": task["noise_level"],
                "required_sequence": task["required_sequence"],
            }
            for task in TASKS.values()
        ],
        "action_schema": IncidentAction.model_json_schema(),
        "observation_schema": SystemObservation.model_json_schema(),
        "reward_schema": RewardOutput.model_json_schema(),
        "state_schema": EnvironmentState.model_json_schema(),
        "grader_schema": TaskGrade.model_json_schema(),
    }

@app.get("/grader")
async def get_grader():
    """MANDATORY: Returns the final score (0.0-1.0) after an episode."""
    return env.grade()


@app.get("/state")
async def get_state():
    """Returns the current environment state for debugging and validation."""
    return env.state


@app.post("/baseline")
async def run_baseline():
    """MANDATORY: Triggers the baseline inference script."""
    from scripts.baseline import run_all_tasks

    results = await run_in_threadpool(run_all_tasks)
    return {"results": results}

def main():
    """The entry point for the OpenEnv validator and deployment."""
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
