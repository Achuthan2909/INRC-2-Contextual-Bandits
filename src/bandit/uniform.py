"""Uniform-random selector — pure exploration baseline."""
from __future__ import annotations

from bandit.base import BanditSelector


class UniformSelector(BanditSelector):
    """Pick one of the active strategies uniformly at random."""

    def _pick(self, active_names: list[str]) -> str:
        return self._rng.choice(active_names)
