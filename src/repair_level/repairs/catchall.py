"""Catch-all repair (Strategy 18): propose small batches of random
feasible same-day swaps between two nurses.

Each violation describes a single candidate swap; ``apply`` re-verifies all
hard-constraint preconditions (H1/H2/H3/H4) and either swaps or rejects.
"""
from __future__ import annotations

import random as _random
from typing import Any

from schedule.representation import Schedule
from repair_level.repairs.base import RepairStrategy
from repair_level.repairs._helpers import (
    build_forbidden_map,
    build_nurse_skills,
    build_minimum_lookup,
    h3_ok_for_assignment,
)


class RandomFeasibleSameDaySwap(RepairStrategy):
    """Sample a handful of candidate swaps each call; ``apply`` swaps the
    two nurses' day-d states iff all hard constraints survive."""

    name = "random_feasible_same_day_swap"

    def __init__(
        self,
        scenario: dict[str, Any],
        initial_history: dict[str, Any],
        week_data_list: list[dict[str, Any]],
        sample_size: int = 8,
        seed: int | None = None,
    ):
        self._initial_history = initial_history
        self._nurse_skills = build_nurse_skills(scenario)
        self._nurse_ids = [n["id"] for n in scenario["nurses"]]
        self._forbidden = build_forbidden_map(scenario)
        self._minimum_lookup = build_minimum_lookup(week_data_list)
        self._sample_size = sample_size
        self._rng = _random.Random(seed)

    def find_violations(
        self,
        schedule: Schedule,
        scenario: dict[str, Any],
        week_data_list: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if schedule.num_days <= 0 or len(self._nurse_ids) < 2:
            return []
        candidates: list[dict[str, Any]] = []
        for _ in range(self._sample_size):
            gd = self._rng.randrange(schedule.num_days)
            a, b = self._rng.sample(self._nurse_ids, 2)
            # Reject trivially identical states up-front (both off, or both
            # with the same (shift, skill)) — apply would no-op anyway.
            if (schedule.get(a, gd) == schedule.get(b, gd)):
                continue
            candidates.append({
                "type": "random_swap",
                "global_day": gd,
                "nurse_a": a,
                "nurse_b": b,
            })
        return candidates

    def _swap_feasible(
        self,
        schedule: Schedule,
        gd: int,
        nurse_a: str,
        nurse_b: str,
        state_a: tuple[str, str] | None,
        state_b: tuple[str, str] | None,
    ) -> bool:
        """Check all hard-constraint preconditions for the proposed swap."""
        # After the swap: nurse_a takes state_b, nurse_b takes state_a.
        # H4 (skill) — each nurse must possess the skill they'll take.
        if state_b is not None:
            shift_b, skill_b = state_b
            if skill_b not in self._nurse_skills[nurse_a]:
                return False
        if state_a is not None:
            shift_a, skill_a = state_a
            if skill_a not in self._nurse_skills[nurse_b]:
                return False

        # H3 — check each nurse's neighbours against their new shift.
        if state_b is not None:
            if not h3_ok_for_assignment(
                schedule, self._initial_history, self._forbidden,
                nurse_a, gd, state_b[0],
            ):
                return False
        if state_a is not None:
            if not h3_ok_for_assignment(
                schedule, self._initial_history, self._forbidden,
                nurse_b, gd, state_a[0],
            ):
                return False

        # H2 — coverage on each affected slot must stay at/above minimum.
        # Only asymmetric swaps (one working / one off) change coverage counts.
        if state_a is not None and state_b is None:
            # nurse_a's slot loses a body, nurse_b's slot gains one.
            shift_a, skill_a = state_a
            min_a = self._minimum_lookup.get((gd, shift_a, skill_a), 0)
            if schedule.coverage(gd, shift_a, skill_a) - 1 < min_a:
                return False
        if state_b is not None and state_a is None:
            shift_b, skill_b = state_b
            min_b = self._minimum_lookup.get((gd, shift_b, skill_b), 0)
            if schedule.coverage(gd, shift_b, skill_b) - 1 < min_b:
                return False
        # Symmetric working/working: counts swap slot-by-slot; each slot
        # loses one nurse and gains another → net unchanged → H2 safe.
        return True

    def apply(
        self,
        schedule: Schedule,
        violation: dict[str, Any],
        scenario: dict[str, Any],
    ) -> bool:
        gd = violation["global_day"]
        nurse_a = violation["nurse_a"]
        nurse_b = violation["nurse_b"]
        state_a = schedule.get(nurse_a, gd)
        state_b = schedule.get(nurse_b, gd)
        if state_a == state_b:
            return False

        if not self._swap_feasible(
            schedule, gd, nurse_a, nurse_b, state_a, state_b,
        ):
            return False

        if state_a is not None:
            schedule.remove_assignment(nurse_a, gd)
        if state_b is not None:
            schedule.remove_assignment(nurse_b, gd)
        if state_b is not None:
            schedule.add_assignment(nurse_a, gd, state_b[0], state_b[1])
        if state_a is not None:
            schedule.add_assignment(nurse_b, gd, state_a[0], state_a[1])
        return True
