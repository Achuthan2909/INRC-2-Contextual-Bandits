"""Gaussian Thompson Sampling selector.

Conjugate Normal-Normal posterior for each arm:

  posterior_precision = 1/prior_var + n_i / noise_var
  posterior_var       = 1 / posterior_precision
  posterior_mean      = posterior_var * (prior_mean/prior_var + sum_rewards_i/noise_var)

At selection time we draw one sample from each active arm's posterior and
pick the arm with the highest draw. Unplayed arms draw from the prior, so the
magnitude of ``prior_var`` controls initial exploration.

Defaults (``prior_mean=0``, ``prior_var=1e4``, ``noise_var=1e4``) are
deliberately weakly-informative; tune them to the reward scale of your task
(for the repair loop, reward = penalty_before − penalty_after so magnitudes
of 10²–10³ are typical).
"""
from __future__ import annotations

import math

from bandit.base import BanditSelector


class ThompsonSamplingSelector(BanditSelector):
    def __init__(
        self,
        strategy_names: list[str],
        prior_mean: float = 0.0,
        prior_var: float = 1e4,
        noise_var: float = 1e4,
        seed: int | None = None,
    ):
        super().__init__(strategy_names, seed=seed)
        if prior_var <= 0 or noise_var <= 0:
            raise ValueError("prior_var and noise_var must be positive")
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.noise_var = noise_var

    def _posterior(self, name: str) -> tuple[float, float]:
        n = self._counts.get(name, 0)
        s = self._sum_rewards.get(name, 0.0)
        post_prec = 1.0 / self.prior_var + n / self.noise_var
        post_var = 1.0 / post_prec
        post_mean = post_var * (
            self.prior_mean / self.prior_var + s / self.noise_var
        )
        return post_mean, post_var

    def _pick(self, active_names: list[str]) -> str:
        best_name = active_names[0]
        best_draw = -math.inf
        for name in active_names:
            mean, var = self._posterior(name)
            draw = self._rng.gauss(mean, math.sqrt(var))
            # Random tie-breaking is built in via Gaussian draws.
            if draw > best_draw:
                best_draw = draw
                best_name = name
        return best_name
