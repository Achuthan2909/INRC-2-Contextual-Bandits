"""Repair loop runner — iteratively applies repair strategies to improve a schedule."""
from __future__ import annotations

import random as _random
from typing import Any, Callable

from schedule.representation import Schedule
from schedule.penalty import compute_penalty, PenaltyResult
from repair_level.greedy_init import generate_initial_schedule
from repair_level.repairs.base import RepairStrategy


# Type alias for the selector callback
Selector = Callable[
    [list[RepairStrategy], list[tuple[RepairStrategy, dict[str, Any]]]],
    tuple[RepairStrategy, dict[str, Any]],
]


def random_selector(
    strategies: list[RepairStrategy],
    candidates: list[tuple[RepairStrategy, dict[str, Any]]],
) -> tuple[RepairStrategy, dict[str, Any]]:
    """Pick a random (strategy, violation) pair from the active list."""
    return _random.choice(candidates)


def run_repairs(
    scenario: dict[str, Any],
    history: dict[str, Any],
    week_data_list: list[dict[str, Any]],
    strategies: list[RepairStrategy],
    selector: Selector = random_selector,
    num_rounds: int = 500,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run the repair loop and return a results dict.

    Returns
    -------
    dict with keys:
        schedule         : final Schedule object
        initial_penalty  : int
        final_penalty    : int
        penalty_trajectory : list[int]  (penalty after each round)
        rounds_run       : int
        total_attempted  : int
        total_succeeded  : int
        strategy_counts  : dict[str, int]  (selections per strategy name)
        hard_violations  : dict[str, int]  (final hard constraint counts)
    """
    if seed is not None:
        _random.seed(seed)

    schedule = generate_initial_schedule(scenario, history, week_data_list)

    initial_result = compute_penalty(schedule, scenario, week_data_list, history)
    initial_penalty = initial_result.total

    penalty_trajectory: list[int] = [initial_penalty]
    total_attempted = 0
    total_succeeded = 0
    strategy_counts: dict[str, int] = {s.name: 0 for s in strategies}

    for round_idx in range(num_rounds):
        # collect all active violations across all strategies
        candidates: list[tuple[RepairStrategy, dict[str, Any]]] = []
        for strategy in strategies:
            violations = strategy.find_violations(
                schedule, scenario, week_data_list,
            )
            for v in violations:
                candidates.append((strategy, v))

        if not candidates:
            break  # nothing left to repair

        # select one (strategy, violation) to attempt
        chosen_strategy, chosen_violation = selector(strategies, candidates)
        strategy_counts[chosen_strategy.name] += 1
        total_attempted += 1

        success = chosen_strategy.apply(schedule, chosen_violation, scenario)
        if success:
            total_succeeded += 1

        result = compute_penalty(schedule, scenario, week_data_list, history)
        penalty_trajectory.append(result.total)

    final_result = compute_penalty(schedule, scenario, week_data_list, history)

    return {
        "schedule": schedule,
        "initial_penalty": initial_penalty,
        "final_penalty": final_result.total,
        "penalty_trajectory": penalty_trajectory,
        "rounds_run": len(penalty_trajectory) - 1,
        "total_attempted": total_attempted,
        "total_succeeded": total_succeeded,
        "strategy_counts": strategy_counts,
        "hard_violations": final_result.hard,
    }
