---
title: DevOps Incident Responder
colorFrom: blue
colorTo: green
sdk: docker
tags:
  - openenv
  - devops
  - evaluation
  - sre
pinned: false
---

# DevOps Incident Responder

DevOps Incident Responder is a real-world OpenEnv benchmark for incident triage and remediation in a small production microservice stack. The agent reads logs, service health, and system metrics, then chooses operational actions such as restarting a crashed API, killing a leaking worker, scaling a degraded database, or clearing stale connections.

This environment is built for the kind of work human on-call engineers actually do: diagnose noisy telemetry, pick safe remediation steps, recover service quickly, and avoid thrashing the system with incorrect actions.

## Why This Benchmark Matters

Many agent benchmarks test browsing, chat, or synthetic workflows. Real production operations are different:

- the signal is incomplete and noisy
- the correct fix often depends on action ordering
- transient control-plane failure makes retry logic important
- good agents should get partial credit for progress, not just a binary pass/fail

This benchmark targets that gap with a compact but realistic SRE loop.

## Round 1 Fit

This submission is designed to match the hackathon checklist directly:

- real-world task simulation: on-call DevOps / SRE incident response
- full OpenEnv interface: typed `Action`, `Observation`, `Reward`, `State`, and grader models
- required API surface: `reset()`, `step()`, `state()`
- metadata in [`openenv.yaml`](openenv.yaml)
- three graded tasks with easy -> medium -> hard progression
- dense reward shaping with partial progress and penalties
- root-level [`inference.py`](inference.py) using the OpenAI client
- Docker-based deployment for Hugging Face Spaces
- reproducible validator and usage instructions

## Environment Overview

The simulated system contains three services:

| Service | Role | Typical Failure Mode |
|---|---|---|
| `auth-api` | Customer-facing API | Crash, degraded request handling |
| `main-db` | Primary database | Pool exhaustion, stale connections |
| `cache` | Supporting service | Healthy background dependency |

Each episode loads one incident scenario. The agent sees only observable state and must recover the system before the step budget expires.

## OpenEnv Interface

The environment exposes the required OpenEnv surface:

- `POST /reset?task_id=<id>&seed=<int>` -> typed initial observation
- `POST /step` -> typed `observation`, typed `reward`, `done`, `info`
- `GET /state` -> typed full environment state
- `GET /tasks` -> task metadata plus schemas
- `GET /grader` -> deterministic typed grade for the current run
- `POST /baseline` -> triggers the packaged baseline runner

Implementation entrypoints:

- metadata: [`openenv.yaml`](openenv.yaml)
- API app: [`server/app.py`](server/app.py)
- environment logic: [`server/environment.py`](server/environment.py)
- task definitions and graders: [`server/tasks.py`](server/tasks.py)

## Action Space

Typed action model: `IncidentAction`

| Field | Type | Meaning |
|---|---|---|
| `command` | `str` | The operation the agent wants to execute |
| `target` | `str` | The service or system target |
| `params` | `Optional[Dict[str, str]]` | Reserved for future control parameters |

Supported commands:

- `restart_service` / `restart`
- `kill_process`
- `scale_db` / `scale`
- `clear_connections`
- `get_logs`
- `check_metrics`

Supported targets:

- `auth-api`
- `main-db`
- `system`

Design note:

- the action space is intentionally narrow and operationally safe
- wrong targets are penalized
- out-of-order actions are penalized
- actions after episode completion no longer generate reward

## Observation Space

Typed observation model: `SystemObservation`

| Field | Type | Meaning |
|---|---|---|
| `logs` | `List[str]` | Most recent operational logs |
| `metrics` | `Dict[str, float]` | Observable system metrics |
| `service_status` | `Dict[str, str]` | Current service health labels |
| `step_count` | `int` | Number of steps taken in the episode |

Example metrics:

- `cpu_load`
- `memory_usage`
- `latency`
- `error_rate`

## State Space

Typed state model: `EnvironmentState`

The state endpoint includes the observation plus hidden grader-facing state:

- current observation
- active incidents
- internal error state
- required remediation sequence
- completed steps
- reward history
- cumulative reward
- flakiness rate
- done flag
- last step info

This separation lets agents act from partial observability while graders still score deterministically.

## Reward Design

Typed reward model: `RewardOutput`

| Signal | Value | Purpose |
|---|---:|---|
| base step cost | `-0.05` | discourages looping and unnecessary actions |
| partial progress | `+0.50` | rewards correct first-step remediation in multi-step tasks |
| final recovery bonus | `+1.00` | rewards successful incident resolution |
| incorrect or out-of-order action | `-0.20` | penalizes unsafe or irrelevant behavior |
| inspection actions | `0.00` before step cost | allows diagnosis without hidden bonus hacking |

Why this works:

- agents get trajectory-level signal instead of a sparse terminal-only reward
- graders inherit the same economics through `cumulative_reward`
- successful but inefficient runs score lower than clean runs
- poor action ordering is visibly punished

## Task Suite

The benchmark ships with three deterministic graded tasks.

| Task | Difficulty | Objective | Expected Sequence | Why It Is Non-Trivial | Max Steps |
|---|---|---|---|---|---:|
| `service_restart` | easy | Recover a crashed API | `restart_service(auth-api)` | transient control-plane flakiness can require retry | `5` |
| `memory_leak` | medium | Terminate the leaking worker, then restart the API | `kill_process(auth-api)` -> `restart_service(auth-api)` | correct root-cause action must happen before restart | `10` |
| `db_connection_exhaustion` | hard | Scale the database first, then clear stale connections | `scale_db(main-db)` -> `clear_connections(main-db)` | noisy telemetry, strict causal ordering, backend saturation hidden behind apparently healthy app pods | `15` |

## Graders

Each task has a dedicated programmatic grader:

- `grade_service_restart`
- `grade_memory_leak`
- `grade_db_connection_exhaustion`

All graders return a typed `TaskGrade` with:

- `score` in `[0.0, 1.0]`
- `resolved`
- `steps_taken`
- `progress_ratio`
- `reward_total`
- `expected_sequence`
- `completed_steps`
- `notes`

Scoring properties:

- deterministic for a fixed code snapshot and seed
- penalty-driven rather than inflated by a giant success constant
- unresolved tasks are capped
- successful but sloppy runs remain distinguishable from strong runs

## Robustness And Exploit Resistance

The environment includes several protections that matter for benchmarking quality:

- wrong-target actions do not receive progress credit
- future-step actions attempted out of order are penalized
- the same fixed seed reproduces flakiness patterns deterministically
- `step()` after `done` returns `0.0` reward and reports `episode_already_done`
- inspection actions provide information without letting agents farm reward

These guardrails make the benchmark more reliable under agentic evaluation and reduce trivial exploit surfaces.

## Baseline Inference

The required root-level baseline script is [`inference.py`](inference.py).

It:

- uses the OpenAI Python client for all model calls
- reads the required hackathon variables `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`
- also accepts `OPENAI_API_KEY` as a local compatibility fallback
- can run against a local server or an already deployed runtime via `DEVOPS_ENV_BASE_URL`
- emits the required `[START]`, `[STEP]`, and `[END]` structured stdout lines

Primary environment variables:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="openai/gpt-oss-120b:groq"
export HF_TOKEN="<your-token>"
```

Optional environment variables:

```bash
export DEVOPS_ENV_BASE_URL="https://tek-wizard-devops-incident-responder.hf.space"
export BASELINE_SEED="8"
export REQUEST_TIMEOUT_SECONDS="15"
export MAX_STEPS="20"
```

## Reference Baselines

Scores are deterministic for a fixed code snapshot, runtime, seed, and model configuration.

### Local Reference Run

Command:

```bash
export BASELINE_SEED="8"
python3 inference.py
```

Observed output with `MODEL_NAME="openai/gpt-oss-120b:groq"`:

| Task | Score | Steps | Reward Trace |
|---|---:|---:|---|
| `service_restart` | `0.95` | `1` | `0.95` |
| `memory_leak` | `0.77` | `3` | `-0.25,0.45,0.95` |
| `db_connection_exhaustion` | `0.65` | `4` | `-0.05,0.45,-0.05,0.95` |

### Deployed Space Reference Run

Runtime URL:

- `https://tek-wizard-devops-incident-responder.hf.space`

Command:

```bash
export DEVOPS_ENV_BASE_URL="https://tek-wizard-devops-incident-responder.hf.space"
export BASELINE_SEED="7"
python3 inference.py
```

Observed output with `MODEL_NAME="openai/gpt-oss-120b:groq"`:

| Task | Score | Steps | Reward Trace |
|---|---:|---:|---|
| `service_restart` | `0.85` | `3` | `-0.05,-0.05,0.95` |
| `memory_leak` | `0.70` | `5` | `-0.25,-0.05,-0.05,0.45,0.95` |
| `db_connection_exhaustion` | `0.40` | `6` | `-0.25,-0.05,-0.05,0.45,-0.25,0.95` |

If the Space is redeployed from a newer commit, rerun the command above and refresh this table so the README always matches the currently live build.

These published runs demonstrate that:

- the easy task can sometimes resolve immediately if the remediation sticks
- the medium and hard tasks remain meaningfully imperfect
- flakiness and noisy telemetry create measurable headroom for stronger agents

## Example Structured Output

```text
[START] task=service_restart env=devops-incident-responder model=openai/gpt-oss-120b:groq
[STEP] step=1 action=restart_service(auth-api) reward=0.95 done=true error=null
[END] success=true steps=1 score=0.95 rewards=0.95
```

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

## API Usage

### Reset an Episode

```bash
curl -X POST "http://localhost:7860/reset?task_id=memory_leak&seed=7"
```

### Take a Step

```bash
curl -X POST "http://localhost:7860/step" \
  -H "Content-Type: application/json" \
  -d '{"command":"kill_process","target":"auth-api"}'
```

### Inspect State And Grade

```bash
curl "http://localhost:7860/state"
curl "http://localhost:7860/grader"
curl "http://localhost:7860/tasks"
```

## Hugging Face Space

This repository is configured for a Docker-based Hugging Face Space and is tagged with `openenv` in the README frontmatter.

Use the runtime URL for validator pings:

- Space page: `https://huggingface.co/spaces/tek-wizard/devops-incident-responder`
- runtime URL: `https://tek-wizard-devops-incident-responder.hf.space`

The validator checks the runtime URL, not the repository page URL.

## Validation

Recommended pre-submission flow:

```bash
docker build -t devops-incident-responder .
python3 inference.py
openenv validate
./val.sh https://tek-wizard-devops-incident-responder.hf.space .
```

The local validator script checks:

- Space reachability and `POST /reset`
- Docker build success
- `openenv validate`

## Project Structure

```text
.
├── Dockerfile
├── inference.py
├── openenv.yaml
├── README.md
├── requirements.txt
├── val.sh
├── scripts/
│   └── baseline.py
└── server/
    ├── app.py
    ├── environment.py
    ├── models.py
    └── tasks.py
```

## Summary

DevOps Incident Responder is a compact but practical benchmark for evaluating whether an agent can do real operational work:

- diagnose incidents from noisy observability data
- respect causal remediation order
- recover service under transient failure
- earn partial credit for meaningful progress
- expose deterministic, typed graders for fair evaluation

That combination makes it a strong fit for Round 1 and a useful benchmark for the broader agent-evaluation community.
