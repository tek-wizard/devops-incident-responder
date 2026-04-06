---
title: DevOps Incident Responder
colorFrom: blue
colorTo: green
sdk: docker
tags:
  - openenv
  - devops
  - evaluation
pinned: false
---

# DevOps Incident Responder

DevOps Incident Responder is an OpenEnv-compatible environment for evaluating agents on a real SRE workflow: incident triage and remediation in a small production microservice stack. The agent reads recent logs, service health, and system metrics, then issues operational actions such as restarting a crashed API, killing a leaking worker, scaling a degraded database, or clearing stale connections.

This is not a game or toy domain. It models a task a human on-call engineer actually performs: diagnosing outages under noisy telemetry and occasional control-plane flakiness.

## Motivation

Most agent benchmarks over-index on browsing or synthetic planning tasks. This environment targets operational reliability work instead:

- reading noisy but meaningful observability signals
- choosing safe remediation actions
- respecting causal ordering in multi-step recoveries
- handling transient failure without thrashing

## Environment Overview

The simulated system contains three services:

- `auth-api`: customer-facing API
- `main-db`: primary database
- `cache`: healthy supporting service

Each episode loads one incident scenario, exposes the current observable system state, and ends when the incident is resolved or the task step budget is exhausted.

## OpenEnv Interface

The environment implements the required OpenEnv surface:

- `POST /reset?task_id=<id>&seed=<int>` returns the initial typed observation
- `POST /step` accepts a typed action and returns `observation`, typed `reward`, `done`, and `info`
- `GET /state` returns the full typed environment state
- `GET /tasks` lists tasks plus JSON schemas for action, observation, reward, state, and grader
- `GET /grader` returns the deterministic typed grade for the current task run

Metadata is defined in [openenv.yaml](/home/prateeksingh/Desktop/devops_responder/openenv.yaml).

## Action Space

Typed action model: `IncidentAction`

Fields:

- `command: str`
- `target: str`
- `params: Optional[Dict[str, str]]`

Supported commands:

- `restart_service` or `restart`
- `kill_process`
- `scale_db` or `scale`
- `clear_connections`
- `get_logs`
- `check_metrics`

Supported targets:

- `auth-api`
- `main-db`
- `system`

## Observation Space

Typed observation model: `SystemObservation`

Fields:

- `logs: List[str]`
- `metrics: Dict[str, float]`
- `service_status: Dict[str, str]`
- `step_count: int`

Example metrics:

- `cpu_load`
- `memory_usage`
- `latency`
- `error_rate`

## Reward Model

Typed reward model: `RewardOutput`

Fields:

- `value: float`
- `reason: str`
- `done: bool`

Reward shaping is dense rather than binary:

- `-0.05` base step penalty on every action
- `+0.50` partial progress when the first required remediation step in a multi-step task succeeds
- `+1.00` final recovery bonus when the incident is fully resolved
- `-0.20` penalty for incorrect or out-of-order actions
- `0.00` net action reward for inspections such as `get_logs` and `check_metrics` before the step penalty

This gives agents trajectory-level signal instead of only terminal success/failure.

The same trajectory is also exposed to the grader through `cumulative_reward`, so inefficiency and mistakes remain visible in the final task score.

## State Model

Typed state model: `EnvironmentState`

Fields include:

- current observation
- hidden system state used by graders
- required sequence
- completed steps
- cumulative reward
- reward history
- flakiness rate
- episode completion flag

## Tasks

The environment ships with three deterministic, programmatic task graders and a clear difficulty progression.

### 1. `service_restart` (easy)

Objective:
Restart a crashed `auth-api` service.

Expected remediation:

- `restart_service` on `auth-api`

Why it is easy:

- single-step recovery
- no telemetry noise
- only transient control-plane flakiness

### 2. `memory_leak` (medium)

Objective:
Terminate the leaking worker process and then restart `auth-api`.

Expected remediation:

- `kill_process` on `auth-api`
- `restart_service` on `auth-api`

Why it is medium:

- requires root-cause action before restart
- mixes signal and noise in logs
- transient flakiness can force retries

### 3. `db_connection_exhaustion` (hard)

Objective:
Scale the degraded database before clearing stale connections.

Expected remediation:

- `scale_db` on `main-db`
- `clear_connections` on `main-db`

Why it is hard:

- strict causal ordering
- high telemetry noise
- apparent service health can hide backend saturation

## Graders

Each task has a dedicated deterministic grader function:

- `grade_service_restart`
- `grade_memory_leak`
- `grade_db_connection_exhaustion`

All graders return a typed `TaskGrade` object with:

- `score` in `0.0–1.0`
- `resolved`
- `steps_taken`
- `progress_ratio`
- `reward_total`
- `expected_sequence`
- `completed_steps`
- `notes`

Scoring is deterministic and reproducible for a fixed seed. The final score is penalty-driven and based primarily on `env.cumulative_reward`, not on a large weighted success bonus.

Grading rules:

- final score is derived from `cumulative_reward` with a task-specific normalization factor
- every `STEP_PENALTY` and `INCORRECT_ACTION_PENALTY` directly lowers the final grade
- unresolved tasks are capped at `0.40`
- resolved tasks retain visible score differences based on retries, diagnostics, and mistakes
- the hard task is intentionally calibrated so a successful run with a few extra steps lands well below a perfect score

This makes the hard task scientifically useful: a weak baseline can honestly score low, while stronger agents still have room to improve.

## Baseline Inference

The required root-level script is [inference.py](/home/prateeksingh/Desktop/devops_responder/inference.py). It:

- uses the OpenAI Python client for all LLM calls
- reads `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`
- uses an observation-driven prompting policy when model credentials are present
- falls back to a deterministic heuristic policy only if no model credentials are present
- runs all tasks with a fixed default seed

The primary benchmark path is observation-driven and uses the OpenAI client with the configured model. The fallback path is a deterministic, task-aware heuristic so local validation can still complete without external model access.

Required environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Optional environment variables:

- `DEVOPS_ENV_BASE_URL`
- `BASELINE_SEED`
- `OPENAI_API_KEY`

## Baseline Scores

Two baseline modes are available:

- LLM baseline: observation-driven policy using the OpenAI client and the configured model
- fallback baseline: deterministic heuristic policy used only when no API credentials are set

Documented reproducible baseline with `BASELINE_SEED=7`:

| Task | Score |
|---|---:|
| `service_restart` | `0.850` |
| `memory_leak` | `0.933` |
| `db_connection_exhaustion` | `0.700` |

These results come from the deterministic fallback policy and are reproducible for the same seed. When `HF_TOKEN` is set, the script still uses the OpenAI client for the primary evaluation path.

The hard task remains intentionally non-perfect even under the deterministic policy, which helps preserve headroom for stronger agents.

## Setup

### Local Python

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t devops-incident-responder .
docker run --rm -p 7860:7860 devops-incident-responder
```

## Usage

### Reset an episode

```bash
curl -X POST "http://localhost:7860/reset?task_id=memory_leak&seed=7"
```

### Take a step

```bash
curl -X POST "http://localhost:7860/step" \
  -H "Content-Type: application/json" \
  -d '{"command":"kill_process","target":"auth-api"}'
```

### Inspect state and grade

```bash
curl "http://localhost:7860/state"
curl "http://localhost:7860/grader"
```

### Run the baseline

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="openai/gpt-oss-120b:groq"
export HF_TOKEN="<your-token>"
python3 inference.py
```

To run the deterministic fallback policy without an LLM, omit `HF_TOKEN`.

## Hugging Face Spaces

This repository is designed for a Docker-based Hugging Face Space and the README frontmatter includes the `openenv` tag required by the hackathon.

Use the Space runtime URL for validator pings, not the repository page URL. The validator expects a `https://<space-name>.hf.space` endpoint that responds to `POST /reset`.

## Validation Checklist

Before submission, verify:

```bash
docker build -t devops-incident-responder .
docker run --rm -p 7860:7860 devops-incident-responder
python3 inference.py
openenv validate
./val.sh https://<your-space>.hf.space .
```

## Project Structure

```text
.
├── Dockerfile
├── inference.py
├── openenv.yaml
├── README.md
├── requirements.txt
├── scripts/
│   └── baseline.py
└── server/
    ├── app.py
    ├── environment.py
    ├── models.py
    └── tasks.py
```
