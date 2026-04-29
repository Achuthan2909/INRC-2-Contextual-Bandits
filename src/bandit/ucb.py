"""Upper Confidence Bound (UCB1) selector.

For each strategy *i* with play-count ``n_i`` and mean reward ``x_i``::

    score_i = x_i + c * sqrt(ln(t) / n_i)

Strategies that have never been played get ``score = +inf`` so every arm is
tried at least once before exploitation kicks in. ``c = sqrt(2)`` matches the
original Auer et al. 2002 formulation.
"""
from __future__ import annotations

import math

from bandit.base import BanditSelector


class UCB1Selector(BanditSelector):
    def __init__(
        self,
        strategy_names: list[str],
        c: float = math.sqrt(2.0),
        seed: int | None = None,
    ):
        super().__init__(strategy_names, seed=seed)
        self.c = c

    def _score(self, name: str) -> float:
        n = self._counts.get(name, 0)
        if n == 0:
            return math.inf
        bonus = self.c * math.sqrt(math.log(max(self._total_plays, 1)) / n)
        return self._mean(name) + bonus

    def _pick(self, active_names: list[str]) -> str:
        scored = [(self._score(n), self._rng.random(), n) for n in active_names]
        scored.sort(reverse=True)
        return scored[0][2]
