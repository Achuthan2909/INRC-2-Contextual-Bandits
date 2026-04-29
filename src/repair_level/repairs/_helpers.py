"""Shared helpers for repair strategies.

Pure functions only — no classes.  Strategies should use these so that
H3 / H4 / H2 / run-length logic stays consistent across the repair family.
"""
from __future__ import annotations

from typing import Any

from schedule.representation import Schedule, REQ_DAY_KEYS


# ---------------------------------------------------------------------------
# Scenario-level lookup builders
# ---------------------------------------------------------------------------

def build_forbidden_map(scenario: dict[str, Any]) -> dict[str, set[str]]:
    """Return {preceding_shift: {forbidden_successor_shifts}}."""
    forbidden: dict[str, set[str]] = {}
    for entry in scenario["forbiddenShiftTypeSuccessions"]:
        succ = entry["succeedingShiftTypes"]
        if succ:
            forbidden[entry["precedingShiftType"]] = set(succ)
    return forbidden


def build_nurse_skills(scenario: dict[str, Any]) -> dict[str, set[str]]:
    """Return {nurse_id: {skills}}."""
    return {n["id"]: set(n["skills"]) for n in scenario["nurses"]}


def build_contract_map(scenario: dict[str, Any]) -> dict[str, dict]:
    """Return {contract_id: contract_dict}."""
    return {c["id"]: c for c in scenario["contracts"]}


def build_shift_type_map(scenario: dict[str, Any]) -> dict[str, dict]:
    """Return {shift_type_id: shift_type_dict}."""
    return {s["id"]: s for s in scenario["shiftTypes"]}


def build_nurse_contract(scenario: dict[str, Any]) -> dict[str, dict]:
    """Return {nurse_id: contract_dict}."""
    contract_map = build_contract_map(scenario)
    return {n["id"]: contract_map[n["contract"]] for n in scenario["nurses"]}


def build_history_map(initial_history: dict[str, Any]) -> dict[str, dict]:
    """Return {nurse_id: history_entry}."""
    return {h["nurse"]: h for h in initial_history["nurseHistory"]}


def build_minimum_lookup(
    week_data_list: list[dict[str, Any]],
) -> dict[tuple[int, str, str], int]:
    """Return {(global_day, shift_type, skill): minimum}.

    Only includes slots with minimum > 0.
    """
    out: dict[tuple[int, str, str], int] = {}
    for week_idx, wd in enumerate(week_data_list):
        for req in wd["requirements"]:
            shift_type = req["shiftType"]
            skill = req["skill"]
            for day_in_week, day_key in enumerate(REQ_DAY_KEYS):
                minimum = req[day_key]["minimum"]
                if minimum == 0:
                    continue
                out[(week_idx * 7 + day_in_week, shift_type, skill)] = minimum
    return out


def build_optimal_lookup(
    week_data_list: list[dict[str, Any]],
) -> dict[tuple[int, str, str], int]:
    """Return {(global_day, shift_type, skill): optimal}."""
    out: dict[tuple[int, str, str], int] = {}
    for week_idx, wd in enumerate(week_data_list):
        for req in wd["requirements"]:
            shift_type = req["shiftType"]
            skill = req["skill"]
            for day_in_week, day_key in enumerate(REQ_DAY_KEYS):
                optimal = req[day_key]["optimal"]
                if optimal == 0:
                    continue
                out[(week_idx * 7 + day_in_week, shift_type, skill)] = optimal
    return out


# ---------------------------------------------------------------------------
# Succession (H3) checks
# ---------------------------------------------------------------------------

def prev_shift_on_day(
    schedule: Schedule,
    initial_history: dict[str, Any],
    nurse_id: str,
    global_day: int,
) -> str | None:
    """Return the nurse's shift on day (global_day - 1).

    For global_day == 0, returns the last shift from history (``None`` if
    the history entry has no preceding shift).
    """
    if global_day > 0:
        return schedule.shift(nurse_id, global_day - 1)
    hist_map = build_history_map(initial_history)
    last = hist_map[nurse_id]["lastAssignedShiftType"]
    return last if last not in (None, "None") else None


def next_shift_on_day(
    schedule: Schedule,
    nurse_id: str,
    global_day: int,
) -> str | None:
    """Return the nurse's shift on day (global_day + 1), or None past horizon."""
    if global_day >= schedule.num_days - 1:
        return None
    return schedule.shift(nurse_id, global_day + 1)


def h3_ok_for_assignment(
    schedule: Schedule,
    initial_history: dict[str, Any],
    forbidden: dict[str, set[str]],
    nurse_id: str,
    global_day: int,
    new_shift_type: str,
) -> bool:
    """True iff assigning *new_shift_type* to *nurse_id* on *global_day*
    would not create an H3 violation with the previous or next day."""
    prev = prev_shift_on_day(schedule, initial_history, nurse_id, global_day)
    if (prev is not None
            and prev in forbidden
            and new_shift_type in forbidden[prev]):
        return False
    nxt = next_shift_on_day(schedule, nurse_id, global_day)
    if (nxt is not None
            and new_shift_type in forbidden
            and nxt in forbidden[new_shift_type]):
        return False
    return True


def h3_ok_for_removal(
    schedule: Schedule,
    initial_history: dict[str, Any],
    forbidden: dict[str, set[str]],
    nurse_id: str,
    global_day: int,
) -> bool:
    """Removing an assignment never introduces an H3 violation.

    Kept for API symmetry — always returns True.
    """
    _ = (schedule, initial_history, forbidden, nurse_id, global_day)
    return True


# ---------------------------------------------------------------------------
# Consecutive-run helpers
# ---------------------------------------------------------------------------

def _runs_from_predicate(
    schedule: Schedule,
    nurse_id: str,
    predicate,
) -> list[tuple[int, int]]:
    """Return inclusive (start, end) runs of days where *predicate(day)* is true."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for d in range(schedule.num_days):
        if predicate(d):
            if start is None:
                start = d
        else:
            if start is not None:
                runs.append((start, d - 1))
                start = None
    if start is not None:
        runs.append((start, schedule.num_days - 1))
    return runs


def consecutive_runs_work(
    schedule: Schedule, nurse_id: str,
) -> list[tuple[int, int]]:
    """Return inclusive (start_day, end_day) runs of working days."""
    return _runs_from_predicate(
        schedule, nurse_id, lambda d: schedule.is_working(nurse_id, d),
    )


def consecutive_runs_off(
    schedule: Schedule, nurse_id: str,
) -> list[tuple[int, int]]:
    """Return inclusive (start_day, end_day) runs of off-days."""
    return _runs_from_predicate(
        schedule, nurse_id, lambda d: not schedule.is_working(nurse_id, d),
    )


def consecutive_runs_shift(
    schedule: Schedule, nurse_id: str, shift_type: str,
) -> list[tuple[int, int]]:
    """Return inclusive (start_day, end_day) runs of the given shift type."""
    return _runs_from_predicate(
        schedule, nurse_id, lambda d: schedule.shift(nurse_id, d) == shift_type,
    )


# ---------------------------------------------------------------------------
# Coverage / minimum safety
# ---------------------------------------------------------------------------

def would_break_minimum(
    schedule: Schedule,
    minimum_lookup: dict[tuple[int, str, str], int],
    global_day: int,
    shift_type: str,
    skill: str,
) -> bool:
    """Return True iff removing one nurse from (day, shift, skill) would
    drop coverage below the slot's minimum requirement."""
    minimum = minimum_lookup.get((global_day, shift_type, skill), 0)
    if minimum == 0:
        return False
    return schedule.coverage(global_day, shift_type, skill) - 1 < minimum


# ---------------------------------------------------------------------------
# Nurse-level aggregates
# ---------------------------------------------------------------------------

def total_assignments(schedule: Schedule, nurse_id: str) -> int:
    """Return the number of days the nurse is working in the horizon."""
    return sum(
        1 for d in range(schedule.num_days) if schedule.is_working(nurse_id, d)
    )


def working_weekend_indices(
    schedule: Schedule, nurse_id: str,
) -> list[int]:
    """Return week indices where the nurse works Sat or Sun."""
    out: list[int] = []
    for w in range(schedule.num_weeks):
        if (schedule.is_working(nurse_id, w * 7 + 5)
                or schedule.is_working(nurse_id, w * 7 + 6)):
            out.append(w)
    return out
