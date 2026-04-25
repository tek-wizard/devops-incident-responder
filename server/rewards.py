from __future__ import annotations

from .models import RewardBreakdown, TaskGrade


def new_breakdown(*, step_penalty: float = -0.05) -> RewardBreakdown:
    return RewardBreakdown(step_penalty=step_penalty)


def total_reward(breakdown: RewardBreakdown) -> float:
    total = (
        breakdown.step_penalty
        + breakdown.investigation
        + breakdown.mitigation
        + breakdown.recovery
        + breakdown.safety
        + breakdown.communication
    )
    return round(total, 2)


def build_grade(
    *,
    task_id: str,
    scenario_family: str,
    resolved: bool,
    root_cause_fixed: bool,
    recovery_confirmed: bool,
    steps_taken: int,
    max_steps: int,
    unsafe_actions: int,
    ineffective_actions: int,
    investigation_coverage: float,
    communication_sent: bool,
    reward_total: float,
    root_causes: list[str],
    fixed_causes: list[str],
    notes: list[str],
) -> TaskGrade:
    efficiency = max(0.0, 1.0 - (steps_taken / max(max_steps, 1)))
    safety = max(0.0, 1.0 - min(1.0, unsafe_actions * 0.25 + ineffective_actions * 0.08))
    score = (
        0.42 * float(resolved)
        + 0.2 * float(root_cause_fixed)
        + 0.15 * investigation_coverage
        + 0.13 * safety
        + 0.05 * efficiency
        + 0.05 * float(communication_sent)
    )
    score = round(max(0.0, min(1.0, score)), 4)
    return TaskGrade(
        task_id=task_id,
        scenario_family=scenario_family,
        score=score,
        resolved=resolved,
        root_cause_fixed=root_cause_fixed,
        recovery_confirmed=recovery_confirmed,
        steps_taken=steps_taken,
        max_steps=max_steps,
        unsafe_actions=unsafe_actions,
        ineffective_actions=ineffective_actions,
        investigation_coverage=round(investigation_coverage, 4),
        communication_sent=communication_sent,
        reward_total=round(reward_total, 2),
        root_causes=root_causes,
        fixed_causes=fixed_causes,
        notes=notes,
    )
