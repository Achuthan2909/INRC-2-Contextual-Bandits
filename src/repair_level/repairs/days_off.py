"""Repairs for S3 consecutive-days-off violations (max and min)."""
from __future__ import annotations

from typing import Any

from schedule.representation import Schedule
from repair_level.repairs.base import RepairStrategy
from repair_level.repairs._helpers import (
    build_forbidden_map,
    build_nurse_skills,
    build_nurse_contract,
    build_minimum_lookup,
    build_optimal_lookup,
    consecutive_runs_off,
    h3_ok_for_assignment,
    total_assignments,
)


def _would_exceed_max_work(
    schedule: Schedule,
    nurse_contract: dict[str, dict],
    nurse_id: str,
    global_day: int,
) -> bool:
    limit = nurse_contract[nurse_id]["maximumNumberOfConsecutiveWorkingDays"]
    run_len = 1
    d = global_day - 1
    while d >= 0 and schedule.is_working(nurse_id, d):
        run_len += 1
        d -= 1
    d = global_day + 1
    while d < schedule.num_days and schedule.is_working(nurse_id, d):
        run_len += 1
        d += 1
    return run_len > limit


def _best_slot_for_nurse(
    schedule: Schedule,
    initial_history: dict[str, Any],
    forbidden: dict[str, set[str]],
    nurse_skills: dict[str, set[str]],
    optimal_lookup: dict[tuple[int, str, str], int],
    nurse_id: str,
    global_day: int,
) -> tuple[str, str] | None:
    """Pick the (shift, skill) slot with the highest coverage need for which
    the nurse is skilled and H3 is satisfied."""
    best: tuple[str, str] | None = None
    best_need = 0
    for (gd, shift, skill), opt in optimal_lookup.items():
        if gd != global_day:
            continue
        if skill not in nurse_skills[nurse_id]:
            continue
        need = opt - schedule.coverage(gd, shift, skill)
        if need <= 0:
            continue
        if not h3_ok_for_assignment(
            schedule, initial_history, forbidden,
            nurse_id, global_day, shift,
        ):
            continue
        if need > best_need:
            best_need = need
            best = (shift, skill)
    return best


class BreakLongRestMid(RepairStrategy):
    """Break an over-length off-run by assigning the nurse to an understaffed
    shift on the middle day of the rest streak."""

    name = "break_long_rest_mid"

    def __init__(
        self,
        scenario: dict[str, Any],
        initial_history: dict[str, Any],
        week_data_list: list[dict[str, Any]],
    ):
        self._initial_history = initial_history
        self._nurse_skills = build_nurse_skills(scenario)
        self._nurse_contract = build_nurse_contract(scenario)
        self._nurse_ids = [n["id"] for n in scenario["nurses"]]
        self._forbidden = build_forbidden_map(scenario)
        self._optimal_lookup = build_optimal_lookup(week_data_list)

    def find_violations(
        self,
        schedule: Schedule,
        scenario: dict[str, Any],
        week_data_list: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        violations: list[dict[str, Any]] = []
        for nid in self._nurse_ids:
            limit = self._nurse_contract[nid][
                "maximumNumberOfConsecutiveDaysOff"
            ]
            for (start, end) in consecutive_runs_off(schedule, nid):
                length = end - start + 1
                if length <= limit:
                    continue
                violations.append({
                    "type": "long_rest_mid",
                    "nurse_id": nid,
                    "start_day": start,
                    "end_day": end,
                    "length": length,
                    "limit": limit,
                    "target_day": (start + end) // 2,
                })
        return violations

    def apply(
        self,
        schedule: Schedule,
        violation: dict[str, Any],
        scenario: dict[str, Any],
    ) -> bool:
        nid = violation["nurse_id"]
        target = violation["target_day"]
        if schedule.is_working(nid, target):
            return False
        if _would_exceed_max_work(
            schedule, self._nurse_contract, nid, target,
        ):
            return False
        slot = _best_slot_for_nurse(
            schedule, self._initial_history, self._forbidden,
            self._nurse_skills, self._optimal_lookup, nid, target,
        )
        if slot is None:
            return False
        shift, skill = slot
        schedule.add_assignment(nid, target, shift, skill)
        return True


class BreakLongRestBoundary(RepairStrategy):
    """Break an over-length off-run by assigning the nurse on the first or
    last day of the rest streak, preferring the boundary with higher
    coverage need."""

    name = "break_long_rest_boundary"

    def __init__(
        self,
        scenario: dict[str, Any],
        initial_history: dict[str, Any],
        week_data_list: list[dict[str, Any]],
    ):
        self._initial_history = initial_history
        self._nurse_skills = build_nurse_skills(scenario)
        self._nurse_contract = build_nurse_contract(scenario)
        self._nurse_ids = [n["id"] for n in scenario["nurses"]]
        self._forbidden = build_forbidden_map(scenario)
        self._optimal_lookup = build_optimal_lookup(week_data_list)

    def find_violations(
        self,
        schedule: Schedule,
        scenario: dict[str, Any],
        week_data_list: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        violations: list[dict[str, Any]] = []
        for nid in self._nurse_ids:
            limit = self._nurse_contract[nid][
                "maximumNumberOfConsecutiveDaysOff"
            ]
            for (start, end) in consecutive_runs_off(schedule, nid):
                length = end - start + 1
                if length <= limit:
                    continue
                violations.append({
                    "type": "long_rest_boundary",
                    "nurse_id": nid,
                    "start_day": start,
                    "end_day": end,
                    "length": length,
                    "limit": limit,
                })
        return violations

    def _candidate_score(
        self, schedule: Schedule, nurse_id: str, global_day: int,
    ) -> tuple[int, tuple[str, str] | None]:
        """Return (best_need, best_slot) for the nurse on global_day."""
        if schedule.is_working(nurse_id, global_day):
            return (0, None)
        if _would_exceed_max_work(
            schedule, self._nurse_contract, nurse_id, global_day,
        ):
            return (0, None)
        best: tuple[str, str] | None = None
        best_need = 0
        for (gd, shift, skill), opt in self._optimal_lookup.items():
            if gd != global_day:
                continue
            if skill not in self._nurse_skills[nurse_id]:
                continue
            need = opt - schedule.coverage(gd, shift, skill)
            if need <= 0:
                continue
            if not h3_ok_for_assignment(
                schedule, self._initial_history, self._forbidden,
                nurse_id, global_day, shift,
            ):
                continue
            if need > best_need:
                best_need = need
                best = (shift, skill)
        return (best_need, best)

    def apply(
        self,
        schedule: Schedule,
        violation: dict[str, Any],
        scenario: dict[str, Any],
    ) -> bool:
        nid = violation["nurse_id"]
        start = violation["start_day"]
        end = violation["end_day"]
        start_score, start_slot = self._candidate_score(schedule, nid, start)
        end_score, end_slot = self._candidate_score(schedule, nid, end)

        if start_slot is not None and start_score >= end_score:
            shift, skill = start_slot
            schedule.add_assignment(nid, start, shift, skill)
            return True
        if end_slot is not None:
            shift, skill = end_slot
            schedule.add_assignment(nid, end, shift, skill)
            return True
        return False


class ExtendShortRest(RepairStrategy):
    """Extend a completed short off-run by removing the nurse from an
    adjacent working day and finding a replacement for that slot.

    A rest run is "completed" when both neighbours are working days within
    the horizon."""

    name = "extend_short_rest"

    def __init__(
        self,
        scenario: dict[str, Any],
        initial_history: dict[str, Any],
        week_data_list: list[dict[str, Any]],
    ):
        self._initial_history = initial_history
        self._nurse_skills = build_nurse_skills(scenario)
        self._nurse_contract = build_nurse_contract(scenario)
        self._nurse_ids = [n["id"] for n in scenario["nurses"]]
        self._forbidden = build_forbidden_map(scenario)
        self._minimum_lookup = build_minimum_lookup(week_data_list)
        self._optimal_lookup = build_optimal_lookup(week_data_list)

    def find_violations(
        self,
        schedule: Schedule,
        scenario: dict[str, Any],
        week_data_list: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        violations: list[dict[str, Any]] = []
        last = schedule.num_days - 1
        for nid in self._nurse_ids:
            min_limit = self._nurse_contract[nid][
                "minimumNumberOfConsecutiveDaysOff"
            ]
            for (start, end) in consecutive_runs_off(schedule, nid):
                length = end - start + 1
                if length >= min_limit:
                    continue
                if start <= 0 or end >= last:
                    continue
                if not (schedule.is_working(nid, start - 1)
                        and schedule.is_working(nid, end + 1)):
                    continue
                violations.append({
                    "type": "short_rest",
                    "nurse_id": nid,
                    "start_day": start,
                    "end_day": end,
                    "length": length,
                    "min_limit": min_limit,
                    "adjacent_work_days": [start - 1, end + 1],
                })
        return violations

    def _pick_replacement(
        self,
        schedule: Schedule,
        global_day: int,
        shift_type: str,
        skill: str,
    ) -> str | None:
        best: str | None = None
        best_count = float("inf")
        for nid in self._nurse_ids:
            if schedule.is_working(nid, global_day):
                continue
            if skill not in self._nurse_skills[nid]:
                continue
            if not h3_ok_for_assignment(
                schedule, self._initial_history, self._forbidden,
                nid, global_day, shift_type,
            ):
                continue
            if _would_exceed_max_work(
                schedule, self._nurse_contract, nid, global_day,
            ):
                continue
            cnt = total_assignments(schedule, nid)
            if cnt < best_count:
                best = nid
                best_count = cnt
        return best

    def apply(
        self,
        schedule: Schedule,
        violation: dict[str, Any],
        scenario: dict[str, Any],
    ) -> bool:
        nid = violation["nurse_id"]
        adjacents = violation["adjacent_work_days"]

        # Rank adjacent working days by how much "surplus" their slot has —
        # largest surplus (current - optimal) first: removal there costs the
        # least coverage.
        def surplus(d: int) -> int:
            a = schedule.get(nid, d)
            if a is None:
                return -999
            shift, skill = a
            opt = self._optimal_lookup.get((d, shift, skill), 0)
            return schedule.coverage(d, shift, skill) - opt

        for adj in sorted(adjacents, key=surplus, reverse=True):
            a = schedule.get(nid, adj)
            if a is None:
                continue
            shift_type, skill = a
            replacement = self._pick_replacement(
                schedule, adj, shift_type, skill,
            )
            if replacement is None:
                continue
            schedule.remove_assignment(nid, adj)
            schedule.add_assignment(replacement, adj, shift_type, skill)
            return True
        return False
