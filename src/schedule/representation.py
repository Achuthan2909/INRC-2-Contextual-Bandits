"""Schedule representation for INRC-II nurse rostering solutions."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

SHORT_DAY_MAP = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}

REQ_DAY_KEYS = [
    "requirementOnMonday",
    "requirementOnTuesday",
    "requirementOnWednesday",
    "requirementOnThursday",
    "requirementOnFriday",
    "requirementOnSaturday",
    "requirementOnSunday",
]

DAY_NAMES_FULL = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
]


class Schedule:
    """Multi-week assignment state for one INRC-II instance.

    Global day indexing: week 0 Mon=0 .. Sun=6, week 1 Mon=7 .. Sun=13, etc.
    """

    def __init__(self, num_weeks: int, nurse_ids: list[str]):
        self.num_weeks = num_weeks
        self.num_days = num_weeks * 7
        self.nurse_ids = nurse_ids
        # nurse_id -> {global_day: (shift_type, skill)}
        self._by_nurse: dict[str, dict[int, tuple[str, str]]] = {
            nid: {} for nid in nurse_ids
        }
        # (global_day, shift_type, skill) -> count
        self._coverage: dict[tuple[int, str, str], int] = defaultdict(int)

    def add_assignment(
        self, nurse_id: str, global_day: int, shift_type: str, skill: str
    ):
        self._by_nurse[nurse_id][global_day] = (shift_type, skill)
        self._coverage[(global_day, shift_type, skill)] += 1

    def remove_assignment(self, nurse_id: str, global_day: int):
        prev = self._by_nurse[nurse_id].pop(global_day, None)
        if prev is not None:
            key = (global_day, prev[0], prev[1])
            self._coverage[key] -= 1

    def get(self, nurse_id: str, global_day: int) -> tuple[str, str] | None:
        """Returns (shift_type, skill) or None if nurse is not working."""
        return self._by_nurse.get(nurse_id, {}).get(global_day)

    def shift(self, nurse_id: str, global_day: int) -> str | None:
        a = self.get(nurse_id, global_day)
        return a[0] if a else None

    def is_working(self, nurse_id: str, global_day: int) -> bool:
        return global_day in self._by_nurse.get(nurse_id, {})

    def coverage(self, global_day: int, shift_type: str, skill: str) -> int:
        return self._coverage.get((global_day, shift_type, skill), 0)

    @classmethod
    def from_solutions(
        cls, scenario: dict[str, Any], solutions: list[dict[str, Any]]
    ) -> Schedule:
        nurse_ids = [n["id"] for n in scenario["nurses"]]
        schedule = cls(len(solutions), nurse_ids)
        for week_idx, sol in enumerate(solutions):
            for a in sol.get("assignments", []):
                day_in_week = SHORT_DAY_MAP[a["day"]]
                global_day = week_idx * 7 + day_in_week
                schedule.add_assignment(
                    a["nurse"], global_day, a["shiftType"], a["skill"]
                )
        return schedule
