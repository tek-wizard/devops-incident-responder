from __future__ import annotations

import atexit
import json
import os
import random
import re
import socket
import subprocess
import sys
import time
from typing import Any

import requests

from scripts.heuristic_policy import choose_action as heuristic_action
from scripts.random_policy import choose_action as random_action

try:
    from openai import OpenAI
except Exception as exc:  # pragma: no cover - depends on runtime packaging
    OpenAI = None  # type: ignore[assignment]
    OPENAI_IMPORT_ERROR: Exception | None = exc
else:
    OPENAI_IMPORT_ERROR = None

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b:groq")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

BENCHMARK = "ai-incident-commander"
DEFAULT_SEED = int(os.getenv("BASELINE_SEED", "201"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "15"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "20"))
BASELINE_POLICY = os.getenv("BASELINE_POLICY", "heuristic").lower()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def log_warn(message: str) -> None:
    print(f"[WARN] {message}", file=sys.stderr, flush=True)


def _build_client() -> OpenAI | None:
    if BASELINE_POLICY != "llm":
        return None
    if OPENAI_IMPORT_ERROR is not None:
        log_warn(
            f"OpenAI SDK import failed ({OPENAI_IMPORT_ERROR.__class__.__name__}: {OPENAI_IMPORT_ERROR}). "
            "Falling back to heuristic policy."
        )
        return None
    if not API_KEY:
        log_warn("No HF_TOKEN or OPENAI_API_KEY available. Falling back to heuristic policy.")
        return None
    try:
        return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception as exc:
        log_warn(f"OpenAI client initialization failed ({exc.__class__.__name__}: {exc}). Falling back to heuristic.")
        return None


def _llm_action(client: OpenAI, observation: dict[str, Any], history: list[dict[str, str]]) -> dict[str, str]:
    prompt = f"""
You are acting as an incident commander in a simulated production environment.

Choose exactly one action as JSON with keys command and target.

Observation:
{json.dumps(observation)}

Recent history:
{json.dumps(history[-6:])}

Rules:
- Prefer investigating before risky mitigations.
- Roll back suspected deploys only when logs or deploy history support it.
- Use failover_db only if dependency checks indicate a partition or DB path isolation.
- Return JSON only.
""".strip()

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = (response.choices[0].message.content or "").strip()
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            content = match.group(0)
        parsed = json.loads(content)
        return {"command": parsed.get("command", "query_logs"), "target": parsed.get("target", "auth-api")}
    except Exception:
        return heuristic_action(observation, history)


def _select_action(
    observation: dict[str, Any],
    history: list[dict[str, str]],
    rng: random.Random,
    client: OpenAI | None,
) -> dict[str, str]:
    if client is not None:
        return _llm_action(client, observation, history)
    if BASELINE_POLICY == "random":
        return random_action(observation, history, rng=rng)
    return heuristic_action(observation, history, rng=rng)


def _format_action(action: dict[str, str]) -> str:
    return f"{action['command']}({action['target']})"


def _find_available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _wait_for_env(base_url: str, timeout_seconds: float = 30.0) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            response = requests.get(f"{base_url}/tasks", timeout=2)
            if response.status_code == 200:
                return
        except Exception as exc:
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"Environment did not become ready at {base_url}: {last_error}")


def _stop_process(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _resolve_env_url() -> tuple[str, subprocess.Popen[bytes] | None]:
    external_url = os.getenv("DEVOPS_ENV_BASE_URL")
    if external_url:
        return external_url.rstrip("/"), None

    port = _find_available_port()
    base_url = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "server.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_env(base_url)
    except Exception:
        _stop_process(proc)
        raise
    atexit.register(_stop_process, proc)
    return base_url, proc


def _get_tasks(env_url: str) -> list[str]:
    response = requests.get(f"{env_url}/tasks", timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    return [task["id"] for task in payload["tasks"]]


def run_task(task_id: str, env_url: str, seed: int, client: OpenAI | None) -> dict[str, Any]:
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    history: list[dict[str, str]] = []
    rng = random.Random(seed)

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME if client is not None else BASELINE_POLICY)

    try:
        reset_response = requests.post(
            f"{env_url}/reset",
            params={"task_id": task_id, "seed": seed},
            timeout=REQUEST_TIMEOUT,
        )
        reset_response.raise_for_status()
        observation = reset_response.json()
        session_id = observation.get("session_id")

        for step_number in range(1, MAX_STEPS + 1):
            action = _select_action(observation, history, rng, client)
            action["session_id"] = session_id
            history.append({"command": action["command"], "target": action["target"]})

            step_response = requests.post(f"{env_url}/step", json=action, timeout=REQUEST_TIMEOUT)
            step_response.raise_for_status()
            payload = step_response.json()

            observation = payload["observation"]
            reward = float(payload["reward"]["value"])
            done = bool(payload.get("done", False))
            error = payload.get("info", {}).get("last_action_error")

            rewards.append(reward)
            steps_taken = step_number
            log_step(step=step_number, action=_format_action(action), reward=reward, done=done, error=error)

            if done:
                break

        grader_response = requests.get(
            f"{env_url}/grader",
            params={"session_id": session_id},
            timeout=REQUEST_TIMEOUT,
        )
        grader_response.raise_for_status()
        grade_payload = grader_response.json()
        score = float(grade_payload["score"])
        success = bool(grade_payload.get("resolved", False))
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task_id": task_id, "score": score, "success": success}


def main() -> list[dict[str, Any]]:
    client = _build_client()
    env_url, proc = _resolve_env_url()
    try:
        task_ids = _get_tasks(env_url)
        return [run_task(task_id=task_id, env_url=env_url, seed=DEFAULT_SEED, client=client) for task_id in task_ids]
    finally:
        _stop_process(proc)


if __name__ == "__main__":
    main()
