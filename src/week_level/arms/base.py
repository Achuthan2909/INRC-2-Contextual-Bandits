"""Base class for week-level bandit arms.

A week-level arm is a full schedule-generating heuristic. Given the scenario,
the history *as of the start of the week*, and the week's demand/requests, it
emits a list of assignment dicts for that one week. The runner is responsible
for folding those assignments into a shared ``Schedule`` and advancing history
between weeks — arms do not see the ``Schedule`` object.
"""
from __future__ import annotations

from typing import Any


class WeekArm:
    name: str = "base"

    def generate(
        self,
        scenario: dict[str, Any],
        history: dict[str, Any],
        week_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Return a list of assignment dicts for one week.

        Each assignment must contain the keys:
            "nurseId", "day", "shiftType", "skill"
        where ``day`` is a full day name ("Monday", ..., "Sunday").
        """
        raise NotImplementedError
