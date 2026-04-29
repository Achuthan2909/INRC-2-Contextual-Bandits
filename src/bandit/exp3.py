"""EXP3 adversarial bandit (Auer et al. 2002).

Maintains a weight per arm; sampling probability mixes the weight-distribution
with a uniform floor of ``gamma / K`` to guarantee exploration::

    p_i = (1 - gamma) * w_i / sum(w) + gamma / K

After selecting arm *i* with probability *p_i* and observing reward *r* we
update only that arm's weight with an unbiased importance-weighted estimator::

    x̂_i = r_norm / p_i
    w_i *= exp(gamma * x̂_i / K)

``r_norm`` is the reward clamped to ``[0, 1]`` via ``reward_scale``. For the
repair loop rewards are penalty deltas; set ``reward_scale`` to a rough upper
bound on per-round improvement (default 200).

Note: unlike UCB/Thompson, EXP3 does *not* assume stationary rewards — useful
for this loop because the schedule changes every round and a strategy's
effective reward distribution drifts over time.
"""
from __future__ import annotations

import math
from typing import Any

from bandit.base import BanditSelector
from repair_level.repairs.base import RepairStrategy


class EXP3Selector(BanditSelector):
    def __init__(
        self,
        strategy_names: list[str],
        gamma: float = 0.1,
        reward_scale: float = 200.0,
        seed: int | None = None,
    ):
        super().__init__(strategy_names, seed=seed)
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if reward_scale <= 0:
            raise ValueError("reward_scale must be positive")
        self.gamma = gamma
        self.reward_scale = reward_scale
        self._weights: dict[str, float] = {name: 1.0 for name in strategy_names}

        # Per-round bookkeeping for the next update() call.
        self._last_chosen: str | None = None
        self._last_prob: float = 0.0
        self._last_K: int = 0

    def _probs(self, active_names: list[str]) -> list[float]:
        K = len(active_names)
        w = [self._weights.get(n, 1.0) for n in active_names]
        total_w = sum(w) or 1.0
        return [(1.0 - self.gamma) * wi / total_w + self.gamma / K for wi in w]

    def __call__(
        self,
        strategies: list[RepairStrategy],
        candidates: list[tuple[RepairStrategy, dict[str, Any]]],
    ) -> tuple[RepairStrategy, dict[str, Any]]:
        by_name, strat_by_name = self._group(candidates)
        active = list(by_name.keys())
        probs = self._probs(active)

        r = self._rng.random()
        acc = 0.0
        chosen = active[-1]
        chosen_p = probs[-1]
        for n, p in zip(active, probs):
            acc += p
            if r <= acc:
                chosen = n
                chosen_p = p
                break

        self._last_chosen = chosen
        self._last_prob = max(chosen_p, 1e-12)
        self._last_K = len(active)

        return strat_by_name[chosen], self._rng.choice(by_name[chosen])

    def _pick(self, active_names: list[str]) -> str:
        # EXP3 overrides ``__call__`` directly so it can record selection
        # probabilities; this method is still provided for API consistency.
        return active_names[0]

    def update(self, name: str, reward: float) -> None:
        super().update(name, reward)

        if name != self._last_chosen:
            return  # out-of-order update; skip EXP3-specific weight update

        r_norm = reward / self.reward_scale
        if r_norm < 0.0:
            r_norm = 0.0
        elif r_norm > 1.0:
            r_norm = 1.0

        if name not in self._weights:
            self._weights[name] = 1.0

        x_hat = r_norm / self._last_prob
        self._weights[name] *= math.exp(self.gamma * x_hat / max(self._last_K, 1))
