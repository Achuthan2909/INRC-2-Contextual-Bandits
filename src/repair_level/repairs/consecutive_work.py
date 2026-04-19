"""Repairs for S2 consecutive-working-day violations (max and min)."""
from __future__ import annotations

from typing import Any

from schedule.representation import Schedule, REQ_DAY_KEYS
from repair_level.repairs.base import RepairStrategy
from repair_level.repairs._helpers import (
    build_forbidden_map,
    build_nurse_skills,
    build_nurse_contract,
    build_minimum_lookup,
    build_optimal_lookup,
    consecutive_runs_work,
    h3_ok_for_assignment,
    total_assignments,
    would_break_minimum,
)


def _pick_replacement(
    schedule: Schedule,
    initial_history: dict[str, Any],
    forbidden: dict[str, set[str]],
    nurse_skills: dict[str, set[str]],
    nurse_contract: dict[str, dict],
    nurse_ids: list[str],
    global_day: int,
    shift_type: str,
    skill: str,
) -> str | None:
    """Pick a nurse off on *global_day*, skilled, H3-safe, and not at
    the consecutive-working-days cap after assignment."""
    best: str | None = None
    best_count = float("inf")
    for nid in nurse_ids:
        if schedule.is_working(nid, global_day):
            continue
        if skill not in nurse_skills[nid]:
            continue
        if not h3_ok_for_assignment(
            schedule, initial_history, forbidden,
            nid, global_day, shift_type,
        ):
            continue
        if _would_exceed_max_consec(schedule, nurse_contract, nid, global_day):
            continue
        cnt = total_assignments(schedule, nid)
        if cnt < best_count:
            best = nid
            best_count = cnt
    return best


def _would_exceed_max_consec(
    schedule: Schedule,
    nurse_contract: dict[str, dict],
    nurse_id: str,
    global_day: int,
) -> bool:
    """True iff assigning the nurse on global_day would create an in-horizon
    working run longer than the contract max."""
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


class _BreakLongWorkStreakBase(RepairStrategy):
    """Shared scaffolding for break-long-work-streak strategies."""

    target_kind: str = "mid"  # "mid" | "start" | "end"

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

    def _target_day(self, start: int, end: int) -> int:
        if self.target_kind == "start":
            return start
        if self.target_kind == "end":
            return end
        return (start + end) // 2

    def find_violations(
        self,
        schedule: Schedule,
        scenario: dict[str, Any],
        week_data_list: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        violations: list[dict[str, Any]] = []
        for nid in self._nurse_ids:
            limit = self._nurse_contract[nid][
                "maximumNumberOfConsecutiveWorkingDays"
            ]
            for (start, end) in consecutive_runs_work(schedule, nid):
                length = end - start + 1
                if length <= limit:
                    continue
                violations.append({
                    "type": "long_work_streak",
                    "nurse_id": nid,
                    "start_day": start,
                    "end_day": end,
                    "length": length,
                    "limit": limit,
                    "target_day": self._target_day(start, end),
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
        assignment = schedule.get(nid, target)
        if assignment is None:
            return False
        shift_type, skill = assignment

        # A replacement fills the vacated (shift, skill) slot, so coverage
        # stays flat and H2 is trivially preserved.
        replacement = _pick_replacement(
            schedule, self._initial_history, self._forbidden,
            self._nurse_skills, self._nurse_contract, self._nurse_ids,
            target, shift_type, skill,
        )
        if replacement is None:
            return False

        schedule.remove_assignment(nid, target)
        schedule.add_assignment(replacement, target, shift_type, skill)
        return True


class BreakLongWorkStreakMid(_BreakLongWorkStreakBase):
    """Break an over-length working run by swapping the nurse out on the
    middle day of the streak and slotting in a feasible replacement."""

    name = "break_long_work_streak_mid"
    target_kind = "mid"


class BreakLongWorkStreakEnd(_BreakLongWorkStreakBase):
    """Break an over-length working run by swapping the nurse out on the
    last day of the streak and slotting in a feasible replacement."""

    name = "break_long_work_streak_end"
    target_kind = "end"


class BreakLongWorkStreakStart(_BreakLongWorkStreakBase):
    """Break an over-length working run by swapping the nurse out on the
    first day of the streak and slotting in a feasible replacement."""

    name = "break_long_work_streak_start"
    target_kind = "start"


class ExtendShortWorkStreak(RepairStrategy):
    """Extend a completed short working run by assigning the nurse to an
    understaffed adjacent off-day.

    A run is "completed" when it is bounded by off-days on both sides
    within the horizon (history boundary treated as off for day 0)."""

    name = "extend_short_work_streak"

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
        # For each global_day, list of (shift, skill, optimal) slots.
        self._slots_by_day: dict[int, list[tuple[str, str, int]]] = {}
        for (gd, st, sk), opt in self._optimal_lookup.items():
            self._slots_by_day.setdefault(gd, []).append((st, sk, opt))

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
                "minimumNumberOfConsecutiveWorkingDays"
            ]
            for (start, end) in consecutive_runs_work(schedule, nid):
                length = end - start + 1
                if length >= min_limit:
                    continue
                # Completed: bounded by off-days on both sides.
                left_off = start > 0
                right_off = end < last
                if not (left_off and right_off):
                    continue
                violations.append({
                    "type": "short_work_streak",
                    "nurse_id": nid,
                    "start_day": start,
                    "end_day": end,
                    "length": length,
                    "min_limit": min_limit,
                    "adjacent_days": [start - 1, end + 1],
                })
        return violations

    def apply(
        self,
        schedule: Schedule,
        violation: dict[str, Any],
        scenario: dict[str, Any],
    ) -> bool:
        nid = violation["nurse_id"]
        # Score each adjacent day by its highest coverage need (optimal - current)
        # across slots the nurse can fill.
        best_day: int | None = None
        best_need = 0
        best_slot: tuple[str, str] | None = None
        for adj in violation["adjacent_days"]:
            if adj < 0 or adj >= schedule.num_days:
                continue
            if schedule.is_working(nid, adj):
                continue
            for (st, sk, opt) in self._slots_by_day.get(adj, []):
                if sk not in self._nurse_skills[nid]:
                    continue
                need = opt - schedule.coverage(adj, st, sk)
                if need <= 0:
                    continue
                if not h3_ok_for_assignment(
                    schedule, self._initial_history, self._forbidden,
                    nid, adj, st,
                ):
                    continue
                if _would_exceed_max_consec(
                    schedule, self._nurse_contract, nid, adj,
                ):
                    continue
                if need > best_need:
                    best_need = need
                    best_day = adj
                    best_slot = (st, sk)

        if best_day is None or best_slot is None:
            return False

        st, sk = best_slot
        schedule.add_assignment(nid, best_day, st, sk)
        return True
