import json
import os
import re
from typing import Any

import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

# INTERNAL CONFIG
ENV_URL = os.getenv("DEVOPS_ENV_BASE_URL", "http://localhost:7860")
DEFAULT_SEED = int(os.getenv("BASELINE_SEED", "7"))

def _build_client() -> OpenAI | None:
    if not API_KEY:
        print("Error: API Key (HF_TOKEN or OPENAI_API_KEY) not found.")
        return None
    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

def _logs_contain(observation: dict[str, Any], needle: str) -> bool:
    return any(needle in line for line in observation.get("logs", []))

def _latest_warn(observation: dict[str, Any]) -> str | None:
    warn_lines = [line for line in observation.get("logs", []) if "[WARN]" in line]
    return warn_lines[-1] if warn_lines else None


def _extract_grade(grader_response: dict[str, Any]) -> float:
    if "score" in grader_response:
        return float(grader_response["score"])
    raise ValueError(f"Unexpected grader response: {grader_response}")

def _mock_action(
    observation: dict[str, Any],
    task_id: str,
    history: list[dict[str, str]],
    pending_retry: dict[str, str] | None = None,
) -> dict[str, str]:
    """Fallback logic if the LLM fails or returns garbage."""
    if pending_retry is not None:
        return pending_retry

    metrics = observation.get("metrics", {})
    logs = observation.get("logs", [])
    logs_blob = "\n".join(logs)

    if task_id == "service_restart":
        return {"command": "restart_service", "target": "auth-api"}

    if task_id == "memory_leak":
        if _logs_contain(observation, "Leaking worker process terminated"):
            return {"command": "restart_service", "target": "auth-api"}
        if "worker-process" in logs_blob or metrics.get("memory_usage", 0.0) > 0.85:
            return {"command": "kill_process", "target": "auth-api"}

    if task_id == "db_connection_exhaustion":
        if _logs_contain(observation, "Database pool capacity scaled out"):
            return {"command": "clear_connections", "target": "main-db"}
        if metrics.get("latency", 0.0) > 0.40:
            return {"command": "scale_db", "target": "main-db"}

    return {"command": "get_logs", "target": "system"}

def _llm_action(
    client: OpenAI,
    observation: dict[str, Any],
    task_id: str,
    history: list[dict[str, str]],
    pending_retry: dict[str, str] | None = None,
) -> dict[str, str]:
    if pending_retry is not None:
        return pending_retry

    prompt = f"""
You are a senior SRE responding to a DevOps incident.

Choose exactly one action:
- command: restart_service | kill_process | scale_db | clear_connections | get_logs | check_metrics
- target: auth-api | main-db | system

Task id: {task_id}
History: {json.dumps(history[-4:])}
Observation: {json.dumps(observation)}

Rules:
- Solve the incident in the minimum number of steps.
- Use 'get_logs' or 'check_metrics' if the current state is unclear.
- Think like an SRE:
  - prefer diagnosing the bottleneck before applying a fix
  - infrastructure capacity issues should usually be addressed before clearing downstream symptoms
  - if an action fails due to a transient control-plane issue, retry the same corrective action
  - avoid unrelated restarts when the logs and metrics point to a database or worker bottleneck
- Return ONLY valid JSON.
- Do NOT include explanations, text, or markdown.

Valid format:
{{"command": "restart_service", "target": "auth-api"}}
""".strip()

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = response.choices[0].message.content.strip()

        # Robust Parsing
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            content = match.group(0)
        
        parsed = json.loads(content)
        return {
            "command": parsed.get("command", "get_logs"),
            "target": parsed.get("target", "system"),
        }
    except Exception as e:
        print(f"LLM Action Error: {e}. Falling back to mock policy.")
        return _mock_action(observation, task_id, history, pending_retry)

def run_task(task_id: str, seed: int = DEFAULT_SEED, client: OpenAI | None = None) -> float:
    print(f"Starting task: {task_id}")
    observation = requests.post(
        f"{ENV_URL}/reset",
        params={"task_id": task_id, "seed": seed},
        timeout=15,
    ).json()
    
    history = []
    pending_retry = None

    for step in range(20):
        if client:
            action = _llm_action(client, observation, task_id, history, pending_retry)
        else:
            action = _mock_action(observation, task_id, history, pending_retry)
            
        print(f"  Step {step}: {action['command']} -> {action['target']}")

        history.append(action)
        step_response = requests.post(f"{ENV_URL}/step", json=action, timeout=15).json()
        observation = step_response["observation"]
        done = step_response["done"]

        # Transient Failure Retry Logic
        warn_line = _latest_warn(observation)
        if warn_line and "did not stick due to transient control-plane failure" in warn_line:
            pending_retry = action
        else:
            pending_retry = None

        if done:
            break

    final_score = _extract_grade(requests.get(f"{ENV_URL}/grader", timeout=15).json())
    print(f"Completed task: {task_id} score={final_score}")
    return final_score

def main():
    client = _build_client()
    try:
        task_response = requests.get(f"{ENV_URL}/tasks", timeout=15).json()
        task_ids = [task["id"] for task in task_response["tasks"]]
    except Exception as e:
        print(f"Could not fetch tasks: {e}")
        task_ids = ["service_restart", "memory_leak", "db_connection_exhaustion"]

    results = []
    for tid in task_ids:
        score = run_task(tid, seed=DEFAULT_SEED, client=client)
        results.append({"task_id": tid, "score": score})

    print("\nFINAL BASELINE SCORES:")
    print(json.dumps(results, indent=2))
    return results

if __name__ == "__main__":
    main()
