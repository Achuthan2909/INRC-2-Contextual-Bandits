"""Base class for repair strategies in the repair-level bandit framework."""
from __future__ import annotations

from typing import Any

from schedule.representation import Schedule


class RepairStrategy:
    """Interface that every repair strategy must implement.

    A repair strategy knows how to:
      1. Identify violations in a schedule that it can address.
      2. Apply a single fix to one violation, preserving all hard constraints.
    """

    name: str = "base"

    def find_violations(
        self,
        schedule: Schedule,
        scenario: dict[str, Any],
        week_data_list: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Return violation objects this strategy can address.

        Each violation is a dict with enough info for ``apply()`` to act on it.
        """
        raise NotImplementedError

    def apply(
        self,
        schedule: Schedule,
        violation: dict[str, Any],
        scenario: dict[str, Any],
    ) -> bool:
        """Mutate *schedule* in place to attempt fixing *violation*.

        Returns True if a modification was made, False if no feasible fix
        exists.  Must preserve all hard constraints — if fixing this would
        create a hard-constraint violation, do nothing and return False.
        """
        raise NotImplementedError
