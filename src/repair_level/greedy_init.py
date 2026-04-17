"""Greedy initial schedule generator for INRC-II.

Two-stage fill:
  Stage 1 — fill *minimum* coverage (H2) so the schedule starts feasible.
  Stage 2 — fill toward *optimal* coverage (S1) using remaining capacity.

Both stages respect hard constraints H1/H3/H4.
"""
from __future__ import annotations

from typing import Any

from schedule.representation import Schedule, REQ_DAY_KEYS


def generate_initial_schedule(
    scenario: dict[str, Any],
    history: dict[str, Any],
    week_data_list: list[dict[str, Any]],
) -> Schedule:
    nurse_ids = [n["id"] for n in scenario["nurses"]]
    num_weeks = len(week_data_list)
    schedule = Schedule(num_weeks, nurse_ids)

    # --- precompute lookups ---
    nurse_skills: dict[str, set[str]] = {
        n["id"]: set(n["skills"]) for n in scenario["nurses"]
    }

    # forbidden successors: preceding_shift -> set of shifts that cannot follow
    forbidden: dict[str, set[str]] = {}
    for entry in scenario["forbiddenShiftTypeSuccessions"]:
        succ = entry["succeedingShiftTypes"]
        if succ:
            forbidden[entry["precedingShiftType"]] = set(succ)

    # previous-day shift from history (for day 0)
    hist_map = {h["nurse"]: h for h in history["nurseHistory"]}
    prev_shift: dict[str, str | None] = {}
    for nid in nurse_ids:
        last = hist_map[nid]["lastAssignedShiftType"]
        prev_shift[nid] = last if last not in (None, "None") else None

    # track assignments per nurse for tie-breaking
    assignment_count: dict[str, int] = {nid: 0 for nid in nurse_ids}

    # --- Stage 1: fill minimum coverage (H2) ---
    _fill_pass(
        schedule=schedule,
        week_data_list=week_data_list,
        nurse_ids=nurse_ids,
        nurse_skills=nurse_skills,
        forbidden=forbidden,
        prev_shift=prev_shift,
        assignment_count=assignment_count,
        target="minimum",
    )

    # --- Stage 2: fill toward optimal coverage (S1) ---
    _fill_pass(
        schedule=schedule,
        week_data_list=week_data_list,
        nurse_ids=nurse_ids,
        nurse_skills=nurse_skills,
        forbidden=forbidden,
        prev_shift=prev_shift,
        assignment_count=assignment_count,
        target="optimal",
    )

    return schedule


def _fill_pass(
    schedule: Schedule,
    week_data_list: list[dict[str, Any]],
    nurse_ids: list[str],
    nurse_skills: dict[str, set[str]],
    forbidden: dict[str, set[str]],
    prev_shift: dict[str, str | None],
    assignment_count: dict[str, int],
    target: str,
) -> None:
    """Fill coverage up to *target* ('minimum' or 'optimal') for each slot."""
    if target not in ("minimum", "optimal"):
        raise ValueError("target must be 'minimum' or 'optimal'")

    for week_idx, wd in enumerate(week_data_list):
        for day_in_week, day_key in enumerate(REQ_DAY_KEYS):
            global_day = week_idx * 7 + day_in_week

            for req in wd["requirements"]:
                demand = req[day_key][target]
                if demand == 0:
                    continue

                shift_type = req["shiftType"]
                skill = req["skill"]
                current = schedule.coverage(global_day, shift_type, skill)
                needed = demand - current

                for _ in range(needed):
                    best_nurse = _pick_nurse(
                        nurse_ids,
                        schedule,
                        global_day,
                        shift_type,
                        skill,
                        nurse_skills,
                        forbidden,
                        prev_shift,
                        assignment_count,
                    )
                    if best_nurse is None:
                        break  # no feasible nurse — leave slot unfilled
                    schedule.add_assignment(best_nurse, global_day, shift_type, skill)
                    assignment_count[best_nurse] += 1

            # update prev_shift for the next day
            for nid in nurse_ids:
                s = schedule.shift(nid, global_day)
                if s is not None:
                    prev_shift[nid] = s
                elif prev_shift.get(nid) is not None:
                    # nurse had a day off, clear the succession chain
                    prev_shift[nid] = None


def _pick_nurse(
    nurse_ids: list[str],
    schedule: Schedule,
    global_day: int,
    shift_type: str,
    skill: str,
    nurse_skills: dict[str, set[str]],
    forbidden: dict[str, set[str]],
    prev_shift: dict[str, str | None],
    assignment_count: dict[str, int],
) -> str | None:
    """Pick the best feasible nurse for (global_day, shift_type, skill).

    Feasibility: H1 (not already working), H4 (has skill), H3 (no forbidden
    succession from yesterday).  Tie-break: fewest total assignments.
    """
    best: str | None = None
    best_count = float("inf")

    for nid in nurse_ids:
        # H1: not already assigned this day
        if schedule.is_working(nid, global_day):
            continue
        # H4: has required skill
        if skill not in nurse_skills[nid]:
            continue
        # H3: no forbidden succession from yesterday
        ps = prev_shift.get(nid)
        if ps is not None and ps in forbidden and shift_type in forbidden[ps]:
            continue

        # tie-break: fewest assignments
        c = assignment_count[nid]
        if c < best_count:
            best = nid
            best_count = c

    return best
