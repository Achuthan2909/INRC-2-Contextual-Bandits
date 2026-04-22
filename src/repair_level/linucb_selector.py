"""LinUCB-backed selector for repair-level strategies.

Implements the ``Selector`` callable contract used by
:func:`repair_level.runner.run_repairs`, plus two optional hooks that the
runner will pick up via ``getattr`` when present:

* ``observe_state(penalty, round_idx, total_rounds)`` — called by the runner
  before selection so we reuse the pre-computed :class:`PenaltyResult`
  instead of recomputing it from the live schedule.
* ``update(strategy_name, reward)`` — per-round reward feedback (same shape
  as for UCB1/eps-greedy selectors in this project).
"""
from __future__ import annotations

import random as _random
from typing import Any

import numpy as np

from bandit.linucb import LinUCB
from repair_level.context import FEATURE_LABELS, build_repair_context
from repair_level.repairs.base import RepairStrategy
from schedule.penalty import PenaltyResult


class LinUCBRepairSelector:
    """Disjoint LinUCB over repair strategies, keyed by ``strategy.name``."""

    def __init__(
        self,
        strategy_names: list[str],
        *,
        alpha: float = 1.0,
        reward_scale: float = 100.0,
        seed: int | None = None,
        linucb: LinUCB | None = None,
    ) -> None:
        if not strategy_names:
            raise ValueError("strategy_names must be non-empty")
        self.strategy_names = list(strategy_names)
        self._name_to_idx = {n: i for i, n in enumerate(self.strategy_names)}
        self.reward_scale = float(reward_scale) if reward_scale else 1.0
        self._rng = _random.Random(seed)

        if linucb is None:
            linucb = LinUCB(
                num_arms=len(strategy_names),
                context_dim=len(FEATURE_LABELS),
                alpha=alpha,
                seed=seed,
            )
        else:
            if linucb.num_arms != len(strategy_names):
                raise ValueError("linucb.num_arms must match len(strategy_names)")
            if linucb.context_dim != len(FEATURE_LABELS):
                raise ValueError(
                    f"linucb.context_dim must be {len(FEATURE_LABELS)}"
                )
        self.linucb = linucb

        # Populated by observe_state() each round and consumed by __call__/update.
        self._ctx: np.ndarray | None = None
        # Last arm chosen — consumed by update().
        self._last_arm_idx: int | None = None
        self._last_ctx: np.ndarray | None = None

        self.pick_counts: dict[str, int] = {n: 0 for n in self.strategy_names}

    # --- hooks the runner looks for ---------------------------------------

    def observe_state(
        self, penalty: PenaltyResult, round_idx: int, total_rounds: int
    ) -> None:
        self._ctx = build_repair_context(penalty, round_idx, total_rounds)

    def update(self, strategy_name: str, reward: float) -> None:
        if self._last_arm_idx is None or self._last_ctx is None:
            return
        scaled = float(reward) / self.reward_scale
        self.linucb.update(self._last_arm_idx, self._last_ctx, scaled)

    def stats(self) -> dict[str, Any]:
        return {
            "kind": "linucb_repair",
            "alpha": self.linucb.alpha,
            "reward_scale": self.reward_scale,
            "pick_counts": dict(self.pick_counts),
            "feature_labels": list(FEATURE_LABELS),
        }

    # --- selector call ----------------------------------------------------

    def __call__(
        self,
        strategies: list[RepairStrategy],
        candidates: list[tuple[RepairStrategy, dict[str, Any]]],
    ) -> tuple[RepairStrategy, dict[str, Any]]:
        if self._ctx is None:
            # Runner didn't call observe_state; fall back to zero context.
            ctx = np.zeros(len(FEATURE_LABELS), dtype=np.float64)
        else:
            ctx = self._ctx

        # Restrict choice to strategies that actually have candidates this round.
        by_strategy: dict[str, list[dict[str, Any]]] = {}
        strat_by_name: dict[str, RepairStrategy] = {}
        for strat, viol in candidates:
            by_strategy.setdefault(strat.name, []).append(viol)
            strat_by_name.setdefault(strat.name, strat)

        active_names = [n for n in self.strategy_names if n in by_strategy]
        if not active_names:
            # Shouldn't happen — runner only calls selector when candidates exist.
            strat, viol = candidates[0]
            return strat, viol

        # Score only active arms; pick the best, tie-break randomly.
        scores: list[tuple[float, int, str]] = []
        for name in active_names:
            a = self._name_to_idx[name]
            A = self.linucb._A[a]
            b = self.linucb._b[a]
            A_inv_x = np.linalg.solve(A, ctx)
            exploit = float(b @ A_inv_x)
            explore = float(np.sqrt(max(0.0, ctx @ A_inv_x)))
            scores.append((exploit + self.linucb.alpha * explore, a, name))
        best_score = max(s[0] for s in scores)
        tol = 1e-9
        tied = [s for s in scores if abs(s[0] - best_score) <= tol]
        _, arm_idx, chosen_name = self._rng.choice(tied)

        self._last_arm_idx = arm_idx
        self._last_ctx = ctx.copy()
        self.pick_counts[chosen_name] = self.pick_counts.get(chosen_name, 0) + 1

        chosen_strategy = strat_by_name[chosen_name]
        chosen_violation = self._rng.choice(by_strategy[chosen_name])
        return chosen_strategy, chosen_violation


__all__ = ["LinUCBRepairSelector"]
