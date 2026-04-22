"""Weekend-related repairs: overload (S7) and incomplete-weekend (S5)."""
from __future__ import annotations

from typing import Any

from schedule.representation import Schedule
from repair_level.repairs.base import RepairStrategy
from repair_level.repairs._helpers import (
    build_forbidden_map,
    build_nurse_skills,
    build_nurse_contract,
    build_history_map,
    build_minimum_lookup,
    build_optimal_lookup,
    h3_ok_for_assignment,
    total_assignments,
    working_weekend_indices,
)


def _prior_working_weekends(
    history_map: dict[str, dict], nurse_id: str,
) -> int:
    return history_map[nurse_id]["numberOfWorkingWeekends"]


def _total_working_weekends(
    schedule: Schedule,
    history_map: dict[str, dict],
    nurse_id: str,
) -> int:
    return (
        _prior_working_weekends(history_map, nurse_id)
        + len(working_weekend_indices(schedule, nurse_id))
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


class ReduceOverloadedWeekends(RepairStrategy):
    """For nurses whose (prior + current) working weekends exceed the
    contract max, swap one of their weekend-day assignments with a less-
    loaded off-day nurse."""

    name = "reduce_overloaded_weekends"

    def __init__(
        self,
        scenario: dict[str, Any],
        initial_history: dict[str, Any],
        week_data_list: list[dict[str, Any]],
    ):
        self._initial_history = initial_history
        self._history_map = build_history_map(initial_history)
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
        for nid in self._nurse_ids:
            max_we = self._nurse_contract[nid][
                "maximumNumberOfWorkingWeekends"
            ]
            total_we = _total_working_weekends(
                schedule, self._history_map, nid,
            )
            if total_we <= max_we:
                continue
            working_weeks = working_weekend_indices(schedule, nid)
            violations.append({
                "type": "overloaded_weekends",
                "nurse_id": nid,
                "total_working_weekends": total_we,
                "limit": max_we,
                "candidate_weeks": working_weeks,
            })
        return violations

    def _replacement_feasible(
        self,
        schedule: Schedule,
        candidate_id: str,
        global_day: int,
        shift_type: str,
        skill: str,
    ) -> bool:
        if schedule.is_working(candidate_id, global_day):
            return False
        if skill not in self._nurse_skills[candidate_id]:
            return False
        if not h3_ok_for_assignment(
            schedule, self._initial_history, self._forbidden,
            candidate_id, global_day, shift_type,
        ):
            return False
        if _would_exceed_max_work(
            schedule, self._nurse_contract, candidate_id, global_day,
        ):
            return False
        # Weekend cap check on the candidate.
        cand_total = _total_working_weekends(
            schedule, self._history_map, candidate_id,
        )
        cand_max = self._nurse_contract[candidate_id][
            "maximumNumberOfWorkingWeekends"
        ]
        # Candidate adding this day may or may not tick a new weekend — only
        # counts if they weren't already working the other weekend day.
        week_idx = global_day // 7
        other_day = week_idx * 7 + (6 if global_day % 7 == 5 else 5)
        already_we = schedule.is_working(candidate_id, other_day)
        projected = cand_total + (0 if already_we else 1)
        if projected > cand_max:
            return False
        return True

    def apply(
        self,
        schedule: Schedule,
        violation: dict[str, Any],
        scenario: dict[str, Any],
    ) -> bool:
        nid = violation["nurse_id"]
        nurse_total = _total_working_weekends(
            schedule, self._history_map, nid,
        )

        # Gather candidate (day, shift, skill, surplus) across all their
        # worked weekend days, preferring highest-surplus slot (least
        # coverage damage if replacement falls through).
        day_options: list[tuple[int, str, str, int]] = []
        for w in violation["candidate_weeks"]:
            for d_off in (5, 6):
                gd = w * 7 + d_off
                a = schedule.get(nid, gd)
                if a is None:
                    continue
                shift, skill = a
                opt = self._optimal_lookup.get((gd, shift, skill), 0)
                surplus = schedule.coverage(gd, shift, skill) - opt
                day_options.append((gd, shift, skill, surplus))

        day_options.sort(key=lambda t: t[3], reverse=True)

        for (gd, shift, skill, _) in day_options:
            # Find a replacement with fewer working weekends.
            best_cand: str | None = None
            best_we = nurse_total  # must be strictly less
            best_assignments = float("inf")
            for cand in self._nurse_ids:
                if cand == nid:
                    continue
                cand_we = _total_working_weekends(
                    schedule, self._history_map, cand,
                )
                if cand_we >= nurse_total:
                    continue
                if not self._replacement_feasible(
                    schedule, cand, gd, shift, skill,
                ):
                    continue
                cnt = total_assignments(schedule, cand)
                if (cand_we < best_we
                        or (cand_we == best_we and cnt < best_assignments)):
                    best_cand = cand
                    best_we = cand_we
                    best_assignments = cnt
            if best_cand is None:
                continue

            schedule.remove_assignment(nid, gd)
            schedule.add_assignment(best_cand, gd, shift, skill)
            return True

        return False


class FixIncompleteWeekend(RepairStrategy):
    """For contracts requiring complete weekends, repair weeks where the
    nurse works exactly one of Sat/Sun by assigning them the missing day
    (preferred) or, failing that, removing them from the worked day."""

    name = "fix_incomplete_weekend"

    def __init__(
        self,
        scenario: dict[str, Any],
        initial_history: dict[str, Any],
        week_data_list: list[dict[str, Any]],
    ):
        self._initial_history = initial_history
        self._history_map = build_history_map(initial_history)
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
        for nid in self._nurse_ids:
            contract = self._nurse_contract[nid]
            if not contract.get("completeWeekends", 0):
                continue
            for w in range(schedule.num_weeks):
                sat = w * 7 + 5
                sun = w * 7 + 6
                works_sat = schedule.is_working(nid, sat)
                works_sun = schedule.is_working(nid, sun)
                if works_sat == works_sun:
                    continue
                violations.append({
                    "type": "incomplete_weekend",
                    "nurse_id": nid,
                    "week_idx": w,
                    "works": "saturday_only" if works_sat else "sunday_only",
                })
        return violations

    def _try_fill(
        self,
        schedule: Schedule,
        nurse_id: str,
        target_day: int,
        preferred_shift: str | None,
    ) -> bool:
        """Try to assign nurse_id on target_day, preferring preferred_shift
        when coverage needs it, otherwise falling back to any understaffed
        slot the nurse can fill."""
        if schedule.is_working(nurse_id, target_day):
            return False
        if _would_exceed_max_work(
            schedule, self._nurse_contract, nurse_id, target_day,
        ):
            return False

        candidates: list[tuple[int, str, str]] = []  # (need, shift, skill)
        for (gd, shift, skill), opt in self._optimal_lookup.items():
            if gd != target_day:
                continue
            if skill not in self._nurse_skills[nurse_id]:
                continue
            need = opt - schedule.coverage(gd, shift, skill)
            if need <= 0:
                continue
            if not h3_ok_for_assignment(
                schedule, self._initial_history, self._forbidden,
                nurse_id, target_day, shift,
            ):
                continue
            # Prefer the preferred_shift by boosting its score.
            priority = need + (1000 if shift == preferred_shift else 0)
            candidates.append((priority, shift, skill))

        if not candidates:
            return False
        candidates.sort(reverse=True)
        _, shift, skill = candidates[0]
        schedule.add_assignment(nurse_id, target_day, shift, skill)
        return True

    def _try_remove_worked_day(
        self,
        schedule: Schedule,
        nurse_id: str,
        worked_day: int,
    ) -> bool:
        """Remove nurse from worked_day and pull in a replacement so that
        coverage (and H2) stays intact."""
        a = schedule.get(nurse_id, worked_day)
        if a is None:
            return False
        shift_type, skill = a

        best_rep: str | None = None
        best_count = float("inf")
        for rep in self._nurse_ids:
            if rep == nurse_id:
                continue
            if schedule.is_working(rep, worked_day):
                continue
            if skill not in self._nurse_skills[rep]:
                continue
            if not h3_ok_for_assignment(
                schedule, self._initial_history, self._forbidden,
                rep, worked_day, shift_type,
            ):
                continue
            if _would_exceed_max_work(
                schedule, self._nurse_contract, rep, worked_day,
            ):
                continue
            cnt = total_assignments(schedule, rep)
            if cnt < best_count:
                best_rep = rep
                best_count = cnt

        if best_rep is None:
            return False

        schedule.remove_assignment(nurse_id, worked_day)
        schedule.add_assignment(best_rep, worked_day, shift_type, skill)
        return True

    def apply(
        self,
        schedule: Schedule,
        violation: dict[str, Any],
        scenario: dict[str, Any],
    ) -> bool:
        nid = violation["nurse_id"]
        w = violation["week_idx"]
        sat = w * 7 + 5
        sun = w * 7 + 6

        if violation["works"] == "saturday_only":
            worked, missing = sat, sun
        else:
            worked, missing = sun, sat

        worked_shift = schedule.shift(nid, worked)
        # Try to add the missing day.
        if self._try_fill(schedule, nid, missing, worked_shift):
            return True
        # Fallback: remove the worked day (with replacement to preserve H2).
        return self._try_remove_worked_day(schedule, nid, worked)
