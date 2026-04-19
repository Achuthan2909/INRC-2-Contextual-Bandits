"""Repair loop runner — iteratively applies repair strategies to improve a schedule."""
from __future__ import annotations

import random as _random
from dataclasses import dataclass
import time
from typing import Any, Callable

from schedule.representation import Schedule
from schedule.penalty import compute_penalty, PenaltyResult
from repair_level.repairs.base import RepairStrategy


# Type alias for the selector callback
Selector = Callable[
    [list[RepairStrategy], list[tuple[RepairStrategy, dict[str, Any]]]],
    tuple[RepairStrategy, dict[str, Any]],
]

@dataclass(frozen=True)
class RunResult:
    schedule: Schedule
    initial_penalty: int
    final_penalty: int
    penalty_trajectory: list[int]
    rounds_run: int
    total_attempted: int
    total_succeeded: int
    strategy_counts: dict[str, int]
    hard_violations: dict[str, int]
    timings_s: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation (excludes the Schedule object)."""
        out: dict[str, Any] = {
            "initial_penalty": self.initial_penalty,
            "final_penalty": self.final_penalty,
            "penalty_trajectory": list(self.penalty_trajectory),
            "rounds_run": self.rounds_run,
            "total_attempted": self.total_attempted,
            "total_succeeded": self.total_succeeded,
            "strategy_counts": dict(self.strategy_counts),
            "hard_violations": dict(self.hard_violations),
        }
        if self.timings_s is not None:
            out["timings_s"] = self.timings_s
        return out


def random_selector(
    strategies: list[RepairStrategy],
    candidates: list[tuple[RepairStrategy, dict[str, Any]]],
) -> tuple[RepairStrategy, dict[str, Any]]:
    """Pick a random strategy uniformly, then a random violation for it.

    Note: `candidates` is a list of (strategy, violation) pairs. Sampling
    directly from that list biases selection toward strategies that expose
    more violations; instead we sample strategies uniformly first.
    """
    by_strategy: dict[RepairStrategy, list[dict[str, Any]]] = {}
    for strategy, violation in candidates:
        by_strategy.setdefault(strategy, []).append(violation)

    chosen_strategy = _random.choice(list(by_strategy.keys()))
    chosen_violation = _random.choice(by_strategy[chosen_strategy])
    return chosen_strategy, chosen_violation


def run_repairs(
    scenario: dict[str, Any],
    history: dict[str, Any],
    week_data_list: list[dict[str, Any]],
    strategies: list[RepairStrategy],
    schedule: Schedule,
    selector: Selector = random_selector,
    num_rounds: int = 500,
    seed: int | None = None,
    collect_timings: bool = False,
    lazy_find_violations: bool = False,
) -> RunResult:
    """Run the repair loop on an existing schedule (caller builds init schedule)."""
    # Avoid mutating the global RNG state; keep reproducibility local to this run.
    rng = _random.Random(seed) if seed is not None else _random.Random()
    if selector is random_selector:
        # Bind the local RNG into the default selector without changing its signature.
        def selector(  # type: ignore[no-redef]
            strategies: list[RepairStrategy],
            candidates: list[tuple[RepairStrategy, dict[str, Any]]],
        ) -> tuple[RepairStrategy, dict[str, Any]]:
            by_strategy: dict[RepairStrategy, list[dict[str, Any]]] = {}
            for strategy, violation in candidates:
                by_strategy.setdefault(strategy, []).append(violation)

            chosen_strategy = rng.choice(list(by_strategy.keys()))
            chosen_violation = rng.choice(by_strategy[chosen_strategy])
            return chosen_strategy, chosen_violation

    timings_s: dict[str, Any] | None = {} if collect_timings else None
    per_strategy_find_s: dict[str, float] | None = (
        {s.name: 0.0 for s in strategies} if collect_timings else None
    )

    t0 = time.perf_counter() if collect_timings else 0.0
    initial_result = compute_penalty(schedule, scenario, week_data_list, history)
    if collect_timings and timings_s is not None:
        timings_s["compute_penalty_initial"] = time.perf_counter() - t0
    initial_penalty = initial_result.total
    current_result: PenaltyResult = initial_result

    penalty_trajectory: list[int] = [initial_penalty]
    total_attempted = 0
    total_succeeded = 0
    strategy_counts: dict[str, int] = {s.name: 0 for s in strategies}
    strat_by_name: dict[str, RepairStrategy] = {s.name: s for s in strategies}

    for round_idx in range(num_rounds):
        chosen_strategy: RepairStrategy | None = None
        chosen_violation: dict[str, Any] | None = None

        if not lazy_find_violations:
            # Eager mode: collect all active violations across all strategies.
            candidates: list[tuple[RepairStrategy, dict[str, Any]]] = []
            if collect_timings:
                t_scan0 = time.perf_counter()
            for strategy in strategies:
                t_s0 = time.perf_counter() if collect_timings else 0.0
                violations = strategy.find_violations(schedule, scenario, week_data_list)
                if collect_timings and per_strategy_find_s is not None:
                    per_strategy_find_s[strategy.name] += time.perf_counter() - t_s0
                for v in violations:
                    candidates.append((strategy, v))
            if collect_timings and timings_s is not None:
                timings_s["scan_candidates_total"] = timings_s.get("scan_candidates_total", 0.0) + (
                    time.perf_counter() - t_scan0
                )

            if not candidates:
                break  # nothing left to repair

            # select one (strategy, violation) to attempt
            if collect_timings:
                t_sel0 = time.perf_counter()
            chosen_strategy, chosen_violation = selector(strategies, candidates)
            if collect_timings and timings_s is not None:
                timings_s["selector_total"] = timings_s.get("selector_total", 0.0) + (
                    time.perf_counter() - t_sel0
                )
        else:
            # Lazy mode: pick a strategy name first, then only scan that strategy.
            # If it has no violations, temporarily remove it from the active set and try again.
            empty_names: set[str] = set()
            all_names = list(strat_by_name.keys())

            sel_acc = 0.0
            scan_acc = 0.0

            while True:
                remaining = [n for n in all_names if n not in empty_names]
                if not remaining:
                    break

                t_pick0 = time.perf_counter() if collect_timings else 0.0
                pick_fn = getattr(selector, "_pick", None)
                if callable(pick_fn):
                    picked_name = pick_fn(remaining)
                else:
                    picked_name = rng.choice(remaining)
                if collect_timings:
                    sel_acc += time.perf_counter() - t_pick0

                strat = strat_by_name[picked_name]
                t_s0 = time.perf_counter() if collect_timings else 0.0
                violations = strat.find_violations(schedule, scenario, week_data_list)
                if collect_timings and per_strategy_find_s is not None:
                    dt = time.perf_counter() - t_s0
                    per_strategy_find_s[strat.name] += dt
                    scan_acc += dt

                if not violations:
                    empty_names.add(picked_name)
                    continue

                chosen_strategy = strat
                chosen_violation = rng.choice(violations)
                break

            if collect_timings and timings_s is not None:
                timings_s["selector_total"] = timings_s.get("selector_total", 0.0) + (
                    sel_acc
                )
                timings_s["scan_candidates_total"] = timings_s.get("scan_candidates_total", 0.0) + (
                    scan_acc
                )

            if chosen_strategy is None or chosen_violation is None:
                break  # nothing left to repair

        strategy_counts[chosen_strategy.name] += 1
        total_attempted += 1

        penalty_before = penalty_trajectory[-1]

        if collect_timings:
            t_apply0 = time.perf_counter()
        success = chosen_strategy.apply(schedule, chosen_violation, scenario)
        if collect_timings and timings_s is not None:
            timings_s["apply_total"] = timings_s.get("apply_total", 0.0) + (
                time.perf_counter() - t_apply0
            )
        if success:
            total_succeeded += 1
            if collect_timings:
                t_pen0 = time.perf_counter()
            current_result = compute_penalty(schedule, scenario, week_data_list, history)
            if collect_timings and timings_s is not None:
                timings_s["compute_penalty_total"] = timings_s.get("compute_penalty_total", 0.0) + (
                    time.perf_counter() - t_pen0
                )
            penalty_after = current_result.total
        else:
            # If the strategy reports no change, avoid an expensive full recompute.
            penalty_after = penalty_before

        penalty_trajectory.append(penalty_after)

        # Feed per-round reward back to stateful selectors (e.g. UCB1).
        # Stateless selectors (plain functions) simply won't expose ``update``.
        update = getattr(selector, "update", None)
        if callable(update):
            update(chosen_strategy.name, penalty_before - penalty_after)

    return RunResult(
        schedule=schedule,
        initial_penalty=initial_penalty,
        final_penalty=penalty_trajectory[-1],
        penalty_trajectory=penalty_trajectory,
        rounds_run=len(penalty_trajectory) - 1,
        total_attempted=total_attempted,
        total_succeeded=total_succeeded,
        strategy_counts=strategy_counts,
        hard_violations=current_result.hard,
        timings_s=None
        if not collect_timings
        else {
            **(timings_s or {}),
            **(
                {"find_violations_by_strategy": per_strategy_find_s}
                if per_strategy_find_s is not None
                else {}
            ),
        },
    )
