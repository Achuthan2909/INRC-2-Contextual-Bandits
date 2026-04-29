"""Coverage repairs — fix under-staffed (day, shift, skill) slots."""
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
    h3_ok_for_assignment,
    total_assignments,
    would_break_minimum,
)


class PullOffDayNurse(RepairStrategy):
    """Find (day, shift, skill) slots below optimal coverage and assign a
    resting nurse who can feasibly fill the slot."""

    name = "pull_off_day_nurse"

    def __init__(self, scenario: dict[str, Any]):
        # precompute lookups once
        self._nurse_skills: dict[str, set[str]] = {
            n["id"]: set(n["skills"]) for n in scenario["nurses"]
        }
        self._nurse_ids: list[str] = [n["id"] for n in scenario["nurses"]]

        self._forbidden: dict[str, set[str]] = {}
        for entry in scenario["forbiddenShiftTypeSuccessions"]:
            succ = entry["succeedingShiftTypes"]
            if succ:
                self._forbidden[entry["precedingShiftType"]] = set(succ)

    def find_violations(
        self,
        schedule: Schedule,
        scenario: dict[str, Any],
        week_data_list: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        violations: list[dict[str, Any]] = []
        for week_idx, wd in enumerate(week_data_list):
            for req in wd["requirements"]:
                shift_type = req["shiftType"]
                skill = req["skill"]
                for day_in_week, day_key in enumerate(REQ_DAY_KEYS):
                    optimal = req[day_key]["optimal"]
                    if optimal == 0:
                        continue
                    gd = week_idx * 7 + day_in_week
                    count = schedule.coverage(gd, shift_type, skill)
                    if count < optimal:
                        violations.append({
                            "type": "coverage",
                            "global_day": gd,
                            "shift_type": shift_type,
                            "skill": skill,
                            "current": count,
                            "optimal": optimal,
                        })
        return violations

    def apply(
        self,
        schedule: Schedule,
        violation: dict[str, Any],
        scenario: dict[str, Any],
    ) -> bool:
        gd = violation["global_day"]
        shift_type = violation["shift_type"]
        skill = violation["skill"]

        best_nurse: str | None = None
        best_count = float("inf")

        for nid in self._nurse_ids:
            # must not already be working this day (H1)
            if schedule.is_working(nid, gd):
                continue
            # must have the skill (H4)
            if skill not in self._nurse_skills[nid]:
                continue
            # must not create forbidden succession with previous day (H3)
            if gd > 0:
                prev = schedule.shift(nid, gd - 1)
                if (prev is not None
                        and prev in self._forbidden
                        and shift_type in self._forbidden[prev]):
                    continue
            # must not create forbidden succession INTO next day (H3)
            if gd < schedule.num_days - 1:
                nxt = schedule.shift(nid, gd + 1)
                if (nxt is not None
                        and shift_type in self._forbidden
                        and nxt in self._forbidden[shift_type]):
                    continue

            # tie-break: fewest total assignments
            cnt = sum(1 for d in range(schedule.num_days)
                      if schedule.is_working(nid, d))
            if cnt < best_count:
                best_nurse = nid
                best_count = cnt

        if best_nurse is None:
            return False

        schedule.add_assignment(best_nurse, gd, shift_type, skill)
        return True


class MoveSameDaySurplusToDeficit(RepairStrategy):
    """Move one nurse from an over-staffed (shift, skill) slot on day *d*
    to an under-staffed slot on the same day.

    Preconditions: removing does not drop surplus slot below its minimum;
    adding preserves H3 on both sides; nurse possesses the deficit skill.
    """

    name = "move_same_day_surplus_to_deficit"

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
        self._minimum_lookup = build_minimum_lookup(week_data_list)
        self._optimal_lookup = build_optimal_lookup(week_data_list)

    def find_violations(
        self,
        schedule: Schedule,
        scenario: dict[str, Any],
        week_data_list: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        # Group optimal slots per day to find same-day deficit/surplus pairs.
        by_day: dict[int, dict[str, list]] = {}
        for (gd, shift_type, skill), optimal in self._optimal_lookup.items():
            count = schedule.coverage(gd, shift_type, skill)
            entry = by_day.setdefault(gd, {"deficit": [], "surplus": []})
            if count < optimal:
                entry["deficit"].append((shift_type, skill, optimal, count))
            elif count > optimal:
                entry["surplus"].append((shift_type, skill, optimal, count))

        violations: list[dict[str, Any]] = []
        for gd, entry in by_day.items():
            if not entry["deficit"] or not entry["surplus"]:
                continue
            safe_surplus = [
                (st, sk, opt, cnt)
                for (st, sk, opt, cnt) in entry["surplus"]
                if not would_break_minimum(
                    schedule, self._minimum_lookup, gd, st, sk,
                )
            ]
            if not safe_surplus:
                continue
            for (dshift, dskill, _opt, cur) in entry["deficit"]:
                violations.append({
                    "type": "same_day_move",
                    "global_day": gd,
                    "deficit_shift": dshift,
                    "deficit_skill": dskill,
                    "current": cur,
                    "surplus_slots": safe_surplus,
                })
        return violations

    def apply(
        self,
        schedule: Schedule,
        violation: dict[str, Any],
        scenario: dict[str, Any],
    ) -> bool:
        gd = violation["global_day"]
        dshift = violation["deficit_shift"]
        dskill = violation["deficit_skill"]
        # Prefer surplus slot with highest overage (count - optimal).
        surplus_slots = sorted(
            violation["surplus_slots"],
            key=lambda s: s[3] - s[2],
            reverse=True,
        )

        for (sshift, sskill, _opt, _cnt) in surplus_slots:
            # Verify live state — earlier rounds may have altered coverage.
            if would_break_minimum(
                schedule, self._minimum_lookup, gd, sshift, sskill,
            ):
                continue

            best_nurse: str | None = None
            best_count = float("inf")
            for nid in self._nurse_ids:
                if schedule.get(nid, gd) != (sshift, sskill):
                    continue
                if dskill not in self._nurse_skills[nid]:
                    continue
                if not h3_ok_for_assignment(
                    schedule, self._initial_history, self._forbidden,
                    nid, gd, dshift,
                ):
                    continue
                cnt = total_assignments(schedule, nid)
                if cnt < best_count:
                    best_nurse = nid
                    best_count = cnt

            if best_nurse is None:
                continue

            schedule.remove_assignment(best_nurse, gd)
            schedule.add_assignment(best_nurse, gd, dshift, dskill)
            return True

        return False


class PullFromAdjacentDaySurplus(RepairStrategy):
    """Add (do not move) a nurse working an over-staffed (shift, skill) on an
    adjacent day, filling their off-day that lies below optimal coverage."""

    name = "pull_from_adjacent_day_surplus"

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
        self._optimal_lookup = build_optimal_lookup(week_data_list)

    def find_violations(
        self,
        schedule: Schedule,
        scenario: dict[str, Any],
        week_data_list: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        violations: list[dict[str, Any]] = []
        for (gd, shift_type, skill), optimal in self._optimal_lookup.items():
            count = schedule.coverage(gd, shift_type, skill)
            if count >= optimal:
                continue
            for adj in (gd - 1, gd + 1):
                if adj < 0 or adj >= schedule.num_days:
                    continue
                adj_optimal = self._optimal_lookup.get(
                    (adj, shift_type, skill), 0,
                )
                adj_count = schedule.coverage(adj, shift_type, skill)
                if adj_optimal > 0 and adj_count > adj_optimal:
                    violations.append({
                        "type": "adjacent_day_pull",
                        "global_day": gd,
                        "adjacent_day": adj,
                        "shift_type": shift_type,
                        "skill": skill,
                        "current": count,
                        "optimal": optimal,
                    })
        return violations

    def _would_exceed_max_consec(
        self, schedule: Schedule, nurse_id: str, global_day: int,
    ) -> bool:
        """True iff assigning nurse on global_day would create a working run
        longer than the contract's consecutive-working-days max within the
        horizon (history boundary handled conservatively — ignored here)."""
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

    def apply(
        self,
        schedule: Schedule,
        violation: dict[str, Any],
        scenario: dict[str, Any],
    ) -> bool:
        gd = violation["global_day"]
        adj = violation["adjacent_day"]
        shift_type = violation["shift_type"]
        skill = violation["skill"]

        best_nurse: str | None = None
        best_count = float("inf")
        for nid in self._nurse_ids:
            if schedule.is_working(nid, gd):
                continue
            if schedule.get(nid, adj) != (shift_type, skill):
                continue
            if skill not in self._nurse_skills[nid]:
                continue
            if not h3_ok_for_assignment(
                schedule, self._initial_history, self._forbidden,
                nid, gd, shift_type,
            ):
                continue
            if self._would_exceed_max_consec(schedule, nid, gd):
                continue
            cnt = total_assignments(schedule, nid)
            if cnt < best_count:
                best_nurse = nid
                best_count = cnt

        if best_nurse is None:
            return False

        schedule.add_assignment(best_nurse, gd, shift_type, skill)
        return True


class SkillReassignment(RepairStrategy):
    """Relabel one nurse's skill on the same (day, shift): swap a surplus
    skill for a deficit skill, without moving the nurse."""

    name = "skill_reassignment"

    def __init__(
        self,
        scenario: dict[str, Any],
        initial_history: dict[str, Any],
        week_data_list: list[dict[str, Any]],
    ):
        self._initial_history = initial_history
        self._nurse_skills = build_nurse_skills(scenario)
        self._nurse_ids = [n["id"] for n in scenario["nurses"]]
        self._minimum_lookup = build_minimum_lookup(week_data_list)
        self._optimal_lookup = build_optimal_lookup(week_data_list)

    def find_violations(
        self,
        schedule: Schedule,
        scenario: dict[str, Any],
        week_data_list: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        # Group optimal slots by (gd, shift) to see skill alternatives.
        by_day_shift: dict[tuple[int, str], list[tuple[str, int]]] = {}
        for (gd, shift_type, skill), optimal in self._optimal_lookup.items():
            by_day_shift.setdefault((gd, shift_type), []).append(
                (skill, optimal),
            )

        violations: list[dict[str, Any]] = []
        for (gd, shift_type), skills in by_day_shift.items():
            for (skill_def, optimal_def) in skills:
                if schedule.coverage(gd, shift_type, skill_def) >= optimal_def:
                    continue
                for (skill_sur, optimal_sur) in skills:
                    if skill_sur == skill_def:
                        continue
                    if schedule.coverage(gd, shift_type, skill_sur) <= optimal_sur:
                        continue
                    if would_break_minimum(
                        schedule, self._minimum_lookup,
                        gd, shift_type, skill_sur,
                    ):
                        continue
                    violations.append({
                        "type": "skill_reassignment",
                        "global_day": gd,
                        "shift_type": shift_type,
                        "skill_def": skill_def,
                        "skill_sur": skill_sur,
                    })
        return violations

    def apply(
        self,
        schedule: Schedule,
        violation: dict[str, Any],
        scenario: dict[str, Any],
    ) -> bool:
        gd = violation["global_day"]
        shift_type = violation["shift_type"]
        skill_def = violation["skill_def"]
        skill_sur = violation["skill_sur"]

        if would_break_minimum(
            schedule, self._minimum_lookup, gd, shift_type, skill_sur,
        ):
            return False

        best_nurse: str | None = None
        best_count = float("inf")
        for nid in self._nurse_ids:
            if schedule.get(nid, gd) != (shift_type, skill_sur):
                continue
            if skill_def not in self._nurse_skills[nid]:
                continue
            cnt = total_assignments(schedule, nid)
            if cnt < best_count:
                best_nurse = nid
                best_count = cnt

        if best_nurse is None:
            return False

        schedule.remove_assignment(best_nurse, gd)
        schedule.add_assignment(best_nurse, gd, shift_type, skill_def)
        return True
