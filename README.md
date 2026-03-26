---
title: DevOps Incident Responder
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# DevOps Incident Responder  
A High-Fidelity SRE Simulation Environment for OpenEnv  

DevOps Incident Responder is a production-grade simulation designed for the Meta PyTorch OpenEnv Hackathon. It challenges AI agents to act as Site Reliability Engineers (SREs) in a microservices ecosystem, moving beyond simple "if-this-then-that" scripts to complex, stateful causal reasoning.

---

## Why This Environment Wins  

### 1. Real-World Utility (30% Weight)  
Unlike toy environments or games, this simulates actual operator workflows. Agents must triage logs, interpret metrics (CPU, Latency, Error Rates), and execute corrective actions on a distributed system.  

It models "Ghost in the Machine" scenarios where a service reports "Healthy" while actually failing due to backend saturation.

---

### 2. Sophisticated Reward Shaping  
We move beyond binary "Success/Failure" rewards. The environment provides a dense reward signal that guides the agent through the "Valley of Disappointment" during multi-step fixes.

| Event               | Reward | Purpose |
|--------------------|--------|--------|
| Success Recovery   | +1.00  | Full resolution of the incident |
| Partial Progress   | +0.50  | Intermediate causal steps |
| Inspection Action  | 0.00   | Free exploration of logs/metrics |
| Step Penalty       | -0.05  | Penalizes inefficiency |
| Flaky Failure      | -0.05  | Forces resilience under uncertainty |
| Incorrect Action   | -0.20  | Penalizes wrong or out-of-order fixes |

---

### 3. Causal & Stateful Logic  
The "Hard" task (`db_connection_exhaustion`) features a strict causal dependency.

**The Logic:**  
Clearing connection pools on a saturated database is ineffective.  

**The Challenge:**  
The agent must scale the database before clearing connections.  

This enforces true system understanding rather than keyword matching.

---

### 4. High-Entropy Noise & Flakiness  

To simulate real-world distributed systems, we introduce controlled uncertainty:

- **Telemetry Noise (~80%)**  
  Real systems are noisy. Agents must filter hundreds of irrelevant "Healthy" signals to detect critical failures.

- **Control-Plane Flakiness (~10–12%)**  
  Even correct actions may randomly fail due to simulated transient issues (e.g., network hiccups, cloud delays).

  👉 This is intentional and critical for evaluation:  
  - Weak agents assume the action is wrong and change strategy  
  - Strong agents recognize transient failure and retry  

  **Result:**  
  We evaluate *persistence and decision-making under uncertainty*, not just correctness.

  > “We simulate real-world unreliability so agents learn to retry intelligently—not panic and switch strategies.”

---

## 📊 Task Difficulty Progression  

| Task ID                     | Difficulty | Primary Challenge      | Causal Chain        |
|---------------------------|------------|------------------------|---------------------|
| service_restart           | Easy       | Process Crash          | Restart             |
| memory_leak               | Medium     | Zombie Processes       | Kill ➔ Restart      |
| db_connection_exhaustion  | Hard       | Resource Saturation    | Scale ➔ Clear       |

---

## 🛠️ Spec Compliance & Architecture  

This project is fully compliant with the OpenEnv Specification:

- **Typed Models:** Pydantic-based `Observation`, `Action`, `RewardOutput`, `EnvironmentState`  
- **Full API Surface:** `/reset`, `/step`, `/tasks`, `/grader`, `/state`  
- **Reproducibility:** Deterministic runs via fixed seeding  

### Project Structure
```
├── server/
│ ├── app.py # FastAPI Interface
│ ├── environment.py # Core RL Logic & Reward Engine
│ ├── tasks.py # Task Definitions
│ └── models.py # Pydantic Models
├── scripts/
│ └── baseline.py # OpenAI-Compatible Agent
├── openenv.yaml # Metadata
└── Dockerfile # Deployment
```

## 📝 Evaluation Logic  

The `/grader` endpoint computes the final score based on cumulative reward:

- **0.85 – 1.0:** Optimal resolution with minimal retries  
- **0.6 – 0.8:** Correct but inefficient  
- **< 0.5:** Failed to resolve within step limit  

---
