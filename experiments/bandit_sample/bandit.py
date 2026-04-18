"""
LinUCB contextual bandit — one arm per scheduling strategy.

Each arm corresponds to a scheduling strategy:
  0 = coverage_first
  1 = preference_first
  2 = fairness_first
  3 = max_coverage
  4 = random_baseline

At each episode (one week), the bandit:
  1. Receives a context vector (8 features from context_builder.py)
  2. Picks the arm with the highest upper-confidence-bound estimate
  3. Runs that arm's strategy to produce a schedule
  4. Receives a reward (negative penalty)
  5. Updates its internal model

The bandit learns over episodes which strategy works best for a given
context — e.g. high staffing demand → max_coverage, many requests →
preference_first.
"""
from __future__ import annotations

import numpy as np

ARM_NAMES = [
    "coverage_first",
    "preference_first",
    "fairness_first",
    "max_coverage",
    "random_baseline",
]


class LinUCB:
    """
    Linear Upper Confidence Bound bandit (Chu et al., 2011).

    Parameters
    ----------
    n_arms     : number of scheduling strategies (arms)
    n_features : dimension of the context vector
    alpha      : exploration parameter — higher = more exploration
    """

    def __init__(self, n_arms: int, n_features: int, alpha: float = 1.0) -> None:
        self.n_arms     = n_arms
        self.n_features = n_features
        self.alpha      = alpha

        # One (A, b) pair per arm.
        # A: feature covariance matrix, shape (d, d), initialised to identity
        # b: reward-weighted context accumulator, shape (d,), initialised to 0
        self.A = [np.eye(n_features) for _ in range(n_arms)]
        self.b = [np.zeros(n_features) for _ in range(n_arms)]

    def select_arm(self, context: np.ndarray) -> int:
        """
        Choose the arm with the highest UCB score.

        For each arm a:
          theta_a = A_a^{-1} b_a          (estimated reward weights)
          p_a = theta_a . x               (expected reward)
                + alpha * sqrt(x^T A_a^{-1} x)   (exploration bonus)
        Pick argmax over a.
        """
        scores = []
        for a in range(self.n_arms):
            theta   = np.linalg.solve(self.A[a], self.b[a])
            exploit = theta @ context
            explore = self.alpha * np.sqrt(context @ np.linalg.solve(self.A[a], context))
            scores.append(exploit + explore)
        return int(np.argmax(scores))

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        """Update model for the chosen arm after observing reward."""
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context

    def arm_name(self, arm: int) -> str:
        return ARM_NAMES[arm]
