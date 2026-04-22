"""Preference repair (S4): move nurses off shifts they requested not to work."""
from __future__ import annotations

from typing import Any

from schedule.representation import Schedule, DAY_NAMES_FULL
from repair_level.repairs.base import RepairStrategy
from repair_level.repairs._helpers import (
    build_forbidden_map,
    build_nurse_skills,
    build_nurse_contract,
    build_minimum_lookup,
    build_optimal_lookup,
    h3_ok_for_assignment,
    total_assignments,
    would_break_minimum,
)


def _build_preference_lookups(
    week_data_list: list[dict[str, Any]],
) -> tuple[set[tuple[int, str]], set[tuple[int, str, str]]]:
    """Return (any_day_off, shift_off):
      any_day_off: set of (global_day, nurse_id) who requested day off.
      shift_off:   set of (global_day, nurse_id, shift_type) for specific.
    """
    any_day_off: set[tuple[int, str]] = set()
    shift_off: set[tuple[int, str, str]] = set()
    for week_idx, wd in enumerate(week_data_list):
        for req in wd.get("shiftOffRequests", []):
            nurse_id = req["nurse"]
            day_in_week = DAY_NAMES_FULL.index(req["day"])
            gd = week_idx * 7 + day_in_week
            shift_type = req["shiftType"]
            if shift_type == "Any":
                any_day_off.add((gd, nurse_id))
            else:
                shift_off.add((gd, nurse_id, shift_type))
    return any_day_off, shift_off


def _has_preference_conflict(
    any_day_off: set[tuple[int, str]],
    shift_off: set[tuple[int, str, str]],
    nurse_id: str,
    global_day: int,
    shift_type: str,
) -> bool:
    if (global_day, nurse_id) in any_day_off:
        return True
    if (global_day, nurse_id, shift_type) in shift_off:
        return True
    return False


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


class SwapOffUnwantedShift(RepairStrategy):
    """If a nurse is assigned to a shift they requested off, try to swap in
    a replacement.  If no replacement is available and the request is for a
    specific shift (not 'Any'), optionally move the nurse to a different
    shift on the same day that has coverage need and no preference conflict."""

    name = "swap_off_unwanted_shift"

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
        self._any_day_off, self._shift_off = _build_preference_lookups(
            week_data_list,
        )

    def find_violations(
        self,
        schedule: Schedule,
        scenario: dict[str, Any],
        week_data_list: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        violations: list[dict[str, Any]] = []
        for (gd, nid) in self._any_day_off:
            if schedule.is_working(nid, gd):
                a = schedule.get(nid, gd)
                assert a is not None
                violations.append({
                    "type": "preference_conflict",
                    "nurse_id": nid,
                    "global_day": gd,
                    "shift_type": a[0],
                    "skill": a[1],
                    "request": "any",
                })
        for (gd, nid, shift_type) in self._shift_off:
            if schedule.shift(nid, gd) == shift_type:
                a = schedule.get(nid, gd)
                assert a is not None
                violations.append({
                    "type": "preference_conflict",
                    "nurse_id": nid,
                    "global_day": gd,
                    "shift_type": shift_type,
                    "skill": a[1],
                    "request": "shift",
                })
        return violations

    def _pick_replacement(
        self,
        schedule: Schedule,
        exclude_id: str,
        global_day: int,
        shift_type: str,
        skill: str,
    ) -> str | None:
        best: str | None = None
        best_count = float("inf")
        for nid in self._nurse_ids:
            if nid == exclude_id:
                continue
            if schedule.is_working(nid, global_day):
                continue
            if skill not in self._nurse_skills[nid]:
                continue
            if _has_preference_conflict(
                self._any_day_off, self._shift_off,
                nid, global_day, shift_type,
            ):
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

    def _try_move_to_other_shift(
        self,
        schedule: Schedule,
        nurse_id: str,
        global_day: int,
        unwanted_shift: str,
        current_skill: str,
    ) -> bool:
        """Move nurse to a different shift on the same day that has coverage
        need, no preference conflict, and H3 safety."""
        # Removing from current slot must not break H2.
        if would_break_minimum(
            schedule, self._minimum_lookup,
            global_day, unwanted_shift, current_skill,
        ):
            return False

        best_shift: str | None = None
        best_skill: str | None = None
        best_need = 0
        for (gd, shift, skill), opt in self._optimal_lookup.items():
            if gd != global_day:
                continue
            if shift == unwanted_shift:
                continue
            if skill not in self._nurse_skills[nurse_id]:
                continue
            if _has_preference_conflict(
                self._any_day_off, self._shift_off,
                nurse_id, global_day, shift,
            ):
                continue
            need = opt - schedule.coverage(gd, shift, skill)
            if need <= 0:
                continue
            # H3 check for the new shift — simulate removal of old shift
            # by temporarily pretending nurse is off, which is fine because
            # h3_ok_for_assignment reads from neighbour days only.
            if not h3_ok_for_assignment(
                schedule, self._initial_history, self._forbidden,
                nurse_id, global_day, shift,
            ):
                continue
            if need > best_need:
                best_need = need
                best_shift = shift
                best_skill = skill

        if best_shift is None or best_skill is None:
            return False

        schedule.remove_assignment(nurse_id, global_day)
        schedule.add_assignment(nurse_id, global_day, best_shift, best_skill)
        return True

    def apply(
        self,
        schedule: Schedule,
        violation: dict[str, Any],
        scenario: dict[str, Any],
    ) -> bool:
        nid = violation["nurse_id"]
        gd = violation["global_day"]
        shift_type = violation["shift_type"]
        skill = violation["skill"]

        a = schedule.get(nid, gd)
        if a is None or a[0] != shift_type:
            return False

        replacement = self._pick_replacement(
            schedule, nid, gd, shift_type, skill,
        )
        if replacement is not None:
            schedule.remove_assignment(nid, gd)
            schedule.add_assignment(replacement, gd, shift_type, skill)
            return True

        # No replacement.  If request was for a specific shift, try to move
        # the nurse to another shift on the same day.
        if violation.get("request") == "shift":
            return self._try_move_to_other_shift(
                schedule, nid, gd, shift_type, skill,
            )
        return False
