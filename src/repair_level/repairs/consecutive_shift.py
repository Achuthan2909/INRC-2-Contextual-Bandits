"""Repairs for S2 consecutive-same-shift violations (max and min)."""
from __future__ import annotations

from typing import Any

from schedule.representation import Schedule
from repair_level.repairs.base import RepairStrategy
from repair_level.repairs._helpers import (
    build_forbidden_map,
    build_nurse_skills,
    build_nurse_contract,
    build_shift_type_map,
    build_optimal_lookup,
    consecutive_runs_shift,
    h3_ok_for_assignment,
)


def _h3_ok_after_swap(
    schedule: Schedule,
    initial_history: dict[str, Any],
    forbidden: dict[str, set[str]],
    nurse_id: str,
    global_day: int,
    new_shift: str,
) -> bool:
    """H3 check using current schedule — callers must have already mutated if
    they want a post-swap view; here we simulate by temporarily ignoring the
    nurse's own entry on global_day since it'll be overwritten."""
    # prev check
    prev: str | None
    if global_day > 0:
        prev = schedule.shift(nurse_id, global_day - 1)
    else:
        hist = next(
            h for h in initial_history["nurseHistory"] if h["nurse"] == nurse_id
        )
        last = hist["lastAssignedShiftType"]
        prev = last if last not in (None, "None") else None
    if (prev is not None
            and prev in forbidden
            and new_shift in forbidden[prev]):
        return False
    if global_day < schedule.num_days - 1:
        nxt = schedule.shift(nurse_id, global_day + 1)
        if (nxt is not None
                and new_shift in forbidden
                and nxt in forbidden[new_shift]):
            return False
    return True


class BreakLongSameShiftStreak(RepairStrategy):
    """Break an over-length same-shift run by swapping the streaking nurse's
    shift with another nurse's (different-shift) assignment on the same day."""

    name = "break_long_same_shift_streak"

    def __init__(
        self,
        scenario: dict[str, Any],
        initial_history: dict[str, Any],
        week_data_list: list[dict[str, Any]],
    ):
        self._initial_history = initial_history
        self._nurse_skills = build_nurse_skills(scenario)
        self._nurse_ids = [n["id"] for n in scenario["nurses"]]
        self._forbidden = build_forbidden_map(scenario)
        self._shift_type_map = build_shift_type_map(scenario)

    def find_violations(
        self,
        schedule: Schedule,
        scenario: dict[str, Any],
        week_data_list: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        violations: list[dict[str, Any]] = []
        for nid in self._nurse_ids:
            for shift_id, st in self._shift_type_map.items():
                limit = st["maximumNumberOfConsecutiveAssignments"]
                for (start, end) in consecutive_runs_shift(
                    schedule, nid, shift_id,
                ):
                    length = end - start + 1
                    if length <= limit:
                        continue
                    violations.append({
                        "type": "long_shift_streak",
                        "nurse_id": nid,
                        "shift_type": shift_id,
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
        gd = violation["target_day"]
        a = schedule.get(nid, gd)
        if a is None:
            return False
        shift_a, skill_a = a

        # Find partner nurse on same day with a different shift type.
        for partner in self._nurse_ids:
            if partner == nid:
                continue
            b = schedule.get(partner, gd)
            if b is None:
                continue
            shift_b, skill_b = b
            if shift_b == shift_a:
                continue
            # Skill compatibility (H4): each must take the other's skill.
            if skill_b not in self._nurse_skills[nid]:
                continue
            if skill_a not in self._nurse_skills[partner]:
                continue
            # H3 on both nurses after the swap.
            if not _h3_ok_after_swap(
                schedule, self._initial_history, self._forbidden,
                nid, gd, shift_b,
            ):
                continue
            if not _h3_ok_after_swap(
                schedule, self._initial_history, self._forbidden,
                partner, gd, shift_a,
            ):
                continue

            schedule.remove_assignment(nid, gd)
            schedule.remove_assignment(partner, gd)
            schedule.add_assignment(nid, gd, shift_b, skill_b)
            schedule.add_assignment(partner, gd, shift_a, skill_a)
            return True

        return False


class ExtendShortSameShiftStreak(RepairStrategy):
    """Extend a short completed same-shift run by converting an adjacent-day
    assignment (or off-day) to the run's shift type.

    Runs at the end of horizon are skipped — their min-violations are
    deferred by the penalty calculator.
    """

    name = "extend_short_same_shift_streak"

    def __init__(
        self,
        scenario: dict[str, Any],
        initial_history: dict[str, Any],
        week_data_list: list[dict[str, Any]],
    ):
        self._initial_history = initial_history
        self._nurse_skills = build_nurse_skills(scenario)
        self._nurse_ids = [n["id"] for n in scenario["nurses"]]
        self._forbidden = build_forbidden_map(scenario)
        self._nurse_contract = build_nurse_contract(scenario)
        self._shift_type_map = build_shift_type_map(scenario)
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
            for shift_id, st in self._shift_type_map.items():
                min_limit = st["minimumNumberOfConsecutiveAssignments"]
                for (start, end) in consecutive_runs_shift(
                    schedule, nid, shift_id,
                ):
                    length = end - start + 1
                    if length >= min_limit:
                        continue
                    # Run must end before the horizon's last day for its min
                    # violation to be penalised.
                    if end >= last:
                        continue
                    violations.append({
                        "type": "short_shift_streak",
                        "nurse_id": nid,
                        "shift_type": shift_id,
                        "start_day": start,
                        "end_day": end,
                        "length": length,
                        "min_limit": min_limit,
                    })
        return violations

    def _would_exceed_max_work(
        self, schedule: Schedule, nurse_id: str, global_day: int,
    ) -> bool:
        limit = self._nurse_contract[nurse_id][
            "maximumNumberOfConsecutiveWorkingDays"
        ]
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

    def _try_swap_to_shift(
        self,
        schedule: Schedule,
        nurse_id: str,
        global_day: int,
        target_shift: str,
    ) -> bool:
        """Swap nurse_id's assignment on global_day with a partner who is
        currently working *target_shift* on that day."""
        a = schedule.get(nurse_id, global_day)
        if a is None:
            return False
        shift_a, skill_a = a
        for partner in self._nurse_ids:
            if partner == nurse_id:
                continue
            b = schedule.get(partner, global_day)
            if b is None:
                continue
            shift_b, skill_b = b
            if shift_b != target_shift:
                continue
            if skill_b not in self._nurse_skills[nurse_id]:
                continue
            if skill_a not in self._nurse_skills[partner]:
                continue
            if not _h3_ok_after_swap(
                schedule, self._initial_history, self._forbidden,
                nurse_id, global_day, shift_b,
            ):
                continue
            if not _h3_ok_after_swap(
                schedule, self._initial_history, self._forbidden,
                partner, global_day, shift_a,
            ):
                continue
            schedule.remove_assignment(nurse_id, global_day)
            schedule.remove_assignment(partner, global_day)
            schedule.add_assignment(nurse_id, global_day, shift_b, skill_b)
            schedule.add_assignment(partner, global_day, shift_a, skill_a)
            return True
        return False

    def _try_fill_off_day(
        self,
        schedule: Schedule,
        nurse_id: str,
        global_day: int,
        target_shift: str,
    ) -> bool:
        """If nurse_id is off on global_day, assign them to target_shift at
        a skill slot that's below optimal coverage."""
        if schedule.is_working(nurse_id, global_day):
            return False
        if self._would_exceed_max_work(schedule, nurse_id, global_day):
            return False
        if not h3_ok_for_assignment(
            schedule, self._initial_history, self._forbidden,
            nurse_id, global_day, target_shift,
        ):
            return False
        # Find a skill slot with coverage need.
        best_skill: str | None = None
        best_need = 0
        for skill in self._nurse_skills[nurse_id]:
            optimal = self._optimal_lookup.get(
                (global_day, target_shift, skill), 0,
            )
            if optimal == 0:
                continue
            need = optimal - schedule.coverage(global_day, target_shift, skill)
            if need > best_need:
                best_need = need
                best_skill = skill
        if best_skill is None:
            return False
        schedule.add_assignment(nurse_id, global_day, target_shift, best_skill)
        return True

    def apply(
        self,
        schedule: Schedule,
        violation: dict[str, Any],
        scenario: dict[str, Any],
    ) -> bool:
        nid = violation["nurse_id"]
        target_shift = violation["shift_type"]
        start = violation["start_day"]
        end = violation["end_day"]

        for adj in (start - 1, end + 1):
            if adj < 0 or adj >= schedule.num_days:
                continue
            if schedule.is_working(nid, adj):
                if schedule.shift(nid, adj) == target_shift:
                    continue  # already same shift, no extension needed
                if self._try_swap_to_shift(schedule, nid, adj, target_shift):
                    return True
            else:
                if self._try_fill_off_day(schedule, nid, adj, target_shift):
                    return True
        return False
