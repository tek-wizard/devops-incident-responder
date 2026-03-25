---
title: DevOps Incident Responder
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# DevOps Incident Responder

An OpenEnv environment designed for the Meta PyTorch OpenEnv Hackathon. This environment simulates a production microservices cluster where an AI agent acts as an SRE to resolve complex, cascading failures.

## Environment Features
- **Hierarchical Tasks**: Progression from simple process restarts to complex database dependency resolution.
- **Stateful Dependencies**: High-level tasks require specific sequences (e.g., scaling capacity before clearing connection pools).
- **Signal vs. Noise**: Logs include realistic system telemetry noise (up to 80% on Hard mode) to test an agent's reasoning capabilities.
- **Non-Deterministic Failures**: 10-20% flakiness in actions to force agents to verify system state post-action.

## Task Difficulty
- **Easy**: `service_restart` (API Process Crash)
- **Medium**: `memory_leak` (Requires Kill -> Restart sequence)
- **Hard**: `db_connection_exhaustion` (Requires Scale -> Clear sequence)

## Getting Started
The environment is exposed via a FastAPI server on port 7860. It follows the standard OpenEnv specification with `/reset`, `/step`, `/tasks`, and `/grader` endpoints.