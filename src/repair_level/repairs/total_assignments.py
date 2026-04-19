"""Total-assignments repair (S6): rebalance nurses above their contract max."""
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
    would_break_minimum,
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


class RebalanceOverworkedNurse(RepairStrategy):
    """For nurses whose (prior + assigned) exceed their contract max,
    remove them from a working day chosen to minimise coverage damage,
    preferring to find a replacement (so coverage stays intact) but
    accepting a bare removal when the slot still has currant > minimum."""

    name = "rebalance_overworked_nurse"

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

    def _total_assigned(self, schedule: Schedule, nurse_id: str) -> int:
        return (
            self._history_map[nurse_id]["numberOfAssignments"]
            + total_assignments(schedule, nurse_id)
        )

    def find_violations(
        self,
        schedule: Schedule,
        scenario: dict[str, Any],
        week_data_list: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        violations: list[dict[str, Any]] = []
        for nid in self._nurse_ids:
            limit = self._nurse_contract[nid][
                "maximumNumberOfAssignments"
            ]
            total = self._total_assigned(schedule, nid)
            if total <= limit:
                continue
            violations.append({
                "type": "overworked",
                "nurse_id": nid,
                "total": total,
                "limit": limit,
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
            # Replacement must be under their own max total assignments.
            rep_total = self._total_assigned(schedule, nid)
            rep_limit = self._nurse_contract[nid][
                "maximumNumberOfAssignments"
            ]
            if rep_total + 1 > rep_limit:
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

        # Rank this nurse's working days by (current - optimal) — highest
        # surplus (least damage) first.
        day_options: list[tuple[int, int, str, str]] = []
        for d in range(schedule.num_days):
            a = schedule.get(nid, d)
            if a is None:
                continue
            shift, skill = a
            opt = self._optimal_lookup.get((d, shift, skill), 0)
            surplus = schedule.coverage(d, shift, skill) - opt
            day_options.append((surplus, d, shift, skill))

        day_options.sort(reverse=True)

        for (surplus, d, shift, skill) in day_options:
            # Prefer days with surplus >= 0 (current >= optimal).
            if surplus < 0:
                # Still try replacement-only fixes for deficit slots.
                pass
            replacement = self._pick_replacement(schedule, d, shift, skill)
            if replacement is not None:
                schedule.remove_assignment(nid, d)
                schedule.add_assignment(replacement, d, shift, skill)
                return True
            # No replacement — only remove if H2 stays intact.
            if would_break_minimum(
                schedule, self._minimum_lookup, d, shift, skill,
            ):
                continue
            # Accept the coverage penalty only if current > minimum.
            schedule.remove_assignment(nid, d)
            return True

        return False
