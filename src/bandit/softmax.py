"""Boltzmann / Softmax selector.

Sampling probability for strategy *i* is proportional to ``exp(x_i / tau)``
where ``x_i`` is its mean reward. ``tau → 0`` is greedy; ``tau → ∞`` is
uniform random. Unplayed arms are forced first (guarantees each arm is tried
at least once).
"""
from __future__ import annotations

import math

from bandit.base import BanditSelector


class SoftmaxSelector(BanditSelector):
    def __init__(
        self,
        strategy_names: list[str],
        tau: float = 1.0,
        seed: int | None = None,
    ):
        super().__init__(strategy_names, seed=seed)
        if tau <= 0:
            raise ValueError("tau must be positive")
        self.tau = tau

    def _pick(self, active_names: list[str]) -> str:
        unplayed = [n for n in active_names if self._counts.get(n, 0) == 0]
        if unplayed:
            return self._rng.choice(unplayed)

        means = [self._mean(n) for n in active_names]
        # Numerically-stable softmax
        m_max = max(means)
        exps = [math.exp((m - m_max) / self.tau) for m in means]
        total = sum(exps)
        r = self._rng.random() * total
        acc = 0.0
        for n, e in zip(active_names, exps):
            acc += e
            if r <= acc:
                return n
        return active_names[-1]
