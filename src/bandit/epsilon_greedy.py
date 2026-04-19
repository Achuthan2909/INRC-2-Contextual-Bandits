"""ε-greedy selector.

With probability ``epsilon`` pick a uniformly-random active strategy; otherwise
pick the strategy with the highest observed mean reward. Unplayed strategies
are always prioritised over played ones when exploiting, so every arm is
guaranteed at least one pull.
"""
from __future__ import annotations

from bandit.base import BanditSelector


class EpsilonGreedySelector(BanditSelector):
    def __init__(
        self,
        strategy_names: list[str],
        epsilon: float = 0.1,
        seed: int | None = None,
    ):
        super().__init__(strategy_names, seed=seed)
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        self.epsilon = epsilon

    def _pick(self, active_names: list[str]) -> str:
        if self._rng.random() < self.epsilon:
            return self._rng.choice(active_names)

        unplayed = [n for n in active_names if self._counts.get(n, 0) == 0]
        if unplayed:
            return self._rng.choice(unplayed)

        scored = [(self._mean(n), self._rng.random(), n) for n in active_names]
        scored.sort(reverse=True)
        return scored[0][2]
