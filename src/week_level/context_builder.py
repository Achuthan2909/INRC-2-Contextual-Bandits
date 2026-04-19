"""Context builder for the week-level contextual bandit.

Emits a 6-dim feature vector per week, each feature normalized to roughly
[0, 1] so that LinUCB weights transfer across instances of different sizes.
Every feature is chosen to discriminate one arm's winning regime from the
others — add or drop features only when a new arm's strength needs a new
signal.
"""
from __future__ import annotations

from typing import Any

import numpy as np


REQ_DAY_KEYS = [
    "requirementOnMonday",
    "requirementOnTuesday",
    "requirementOnWednesday",
    "requirementOnThursday",
    "requirementOnFriday",
    "requirementOnSaturday",
    "requirementOnSunday",
]

FEATURE_LABELS = [
    "coverage_slack",            # tight demand => coverage_first wins
    "mean_fatigue_ratio",        # tired nurses => fatigue_aware wins
    "weekend_spread_ratio",      # uneven weekends => weekend_balancing wins
    "request_density",           # many shift-off requests => preference arm wins
    "max_assignment_saturation", # some nurse near cap => lookahead arm wins
    "week_position",             # horizon position — always useful
]


def _total_min_demand(week_data: dict[str, Any]) -> int:
    total = 0
    for req in week_data["requirements"]:
        for day_key in REQ_DAY_KEYS:
            total += req[day_key]["minimum"]
    return total


def build_context(
    scenario: dict[str, Any],
    history: dict[str, Any],
    week_data: dict[str, Any],
    week_idx: int,
    total_weeks: int,
) -> tuple[np.ndarray, list[str]]:
    contracts = {c["id"]: c for c in scenario["contracts"]}
    nurses = scenario["nurses"]
    num_nurses = len(nurses)
    nurse_contract = {n["id"]: contracts[n["contract"]] for n in nurses}
    hist_by_nurse = {h["nurse"]: h for h in history["nurseHistory"]}

    # 1. Coverage tightness. Fraction of nurse-capacity the week's minimum
    # demand consumes. 0 => very loose demand, 1 => demand saturates every
    # nurse every day. (Higher = tighter = coverage_first arm more useful.)
    total_min = _total_min_demand(week_data)
    capacity = num_nurses * 7
    coverage_slack = float(np.clip(total_min / capacity, 0.0, 1.0))

    # 2. Mean fatigue ratio. Each nurse's current consecutive working days
    # divided by their contract's maximum. Averaged across nurses.
    fatigue_ratios = []
    for nurse in nurses:
        nid = nurse["id"]
        c = nurse_contract[nid]
        cap = c["maximumNumberOfConsecutiveWorkingDays"]
        if cap > 0:
            fatigue_ratios.append(
                hist_by_nurse[nid]["numberOfConsecutiveWorkingDays"] / cap
            )
    mean_fatigue_ratio = float(np.clip(np.mean(fatigue_ratios), 0.0, 1.0))

    # 3. Weekend spread. Std of per-nurse weekends-worked-so-far, normalized
    # by the maximum allowed weekends across contracts. High std => some
    # nurses are overworking weekends; weekend_balancing arm is valuable.
    weekends_worked = np.array([
        hist_by_nurse[n["id"]]["numberOfWorkingWeekends"] for n in nurses
    ])
    max_weekend_cap = max(
        c["maximumNumberOfWorkingWeekends"] for c in contracts.values()
    )
    if max_weekend_cap > 0:
        weekend_spread_ratio = float(np.clip(
            weekends_worked.std() / max_weekend_cap, 0.0, 1.0,
        ))
    else:
        weekend_spread_ratio = 0.0

    # 4. Request density. Shift-off requests per nurse this week.
    num_requests = len(week_data.get("shiftOffRequests", []))
    request_density = float(np.clip(num_requests / num_nurses, 0.0, 1.0))

    # 5. Assignment saturation. For each nurse, prior_assignments /
    # contract_max. Take the max across nurses — if even one is near
    # cap, the lookahead arm matters.
    saturations = []
    for nurse in nurses:
        nid = nurse["id"]
        cap = nurse_contract[nid]["maximumNumberOfAssignments"]
        if cap > 0:
            saturations.append(
                hist_by_nurse[nid]["numberOfAssignments"] / cap
            )
    max_assignment_saturation = float(np.clip(
        max(saturations) if saturations else 0.0, 0.0, 1.0,
    ))

    # 6. Week position. Progress through the horizon — weight constraints
    # like s6/s7 bite harder in later weeks.
    if total_weeks > 1:
        week_position = week_idx / (total_weeks - 1)
    else:
        week_position = 0.0

    features = np.array([
        coverage_slack,
        mean_fatigue_ratio,
        weekend_spread_ratio,
        request_density,
        max_assignment_saturation,
        week_position,
    ], dtype=float)

    return features, list(FEATURE_LABELS)
