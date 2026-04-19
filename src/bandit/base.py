"""Shared base class for bandit selectors.

All bandit selectors share the same plumbing:

* group the runner's flat ``list[(strategy, violation)]`` into a per-strategy
  dict of active violations,
* pick one strategy using the algorithm's own rule (``_pick``),
* sample one violation from that strategy uniformly at random.

Concrete subclasses only need to implement ``_pick(active_names)``.

Every selector tracks ``counts``, ``sum_rewards``, and ``total_plays`` so the
runner can feed ``update(name, reward)`` back after each round and so ``stats()``
produces a consistent log across algorithms.
"""
from __future__ import annotations

import random as _random
from typing import Any

from repair_level.repairs.base import RepairStrategy


class BanditSelector:
    def __init__(
        self,
        strategy_names: list[str],
        seed: int | None = None,
    ):
        self._rng = _random.Random(seed)
        self._counts: dict[str, int] = {name: 0 for name in strategy_names}
        self._sum_rewards: dict[str, float] = {name: 0.0 for name in strategy_names}
        self._total_plays: int = 0

    def __call__(
        self,
        strategies: list[RepairStrategy],
        candidates: list[tuple[RepairStrategy, dict[str, Any]]],
    ) -> tuple[RepairStrategy, dict[str, Any]]:
        by_name, strat_by_name = self._group(candidates)
        active = list(by_name.keys())
        name = self._pick(active)
        return strat_by_name[name], self._rng.choice(by_name[name])

    def _pick(self, active_names: list[str]) -> str:
        raise NotImplementedError

    def update(self, name: str, reward: float) -> None:
        """Record the observed reward for a strategy we just selected."""
        if name not in self._counts:
            self._counts[name] = 0
            self._sum_rewards[name] = 0.0
        self._counts[name] += 1
        self._sum_rewards[name] += float(reward)
        self._total_plays += 1

    def _group(
        self,
        candidates: list[tuple[RepairStrategy, dict[str, Any]]],
    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, RepairStrategy]]:
        by_name: dict[str, list[dict[str, Any]]] = {}
        strat_by_name: dict[str, RepairStrategy] = {}
        for strat, violation in candidates:
            by_name.setdefault(strat.name, []).append(violation)
            strat_by_name[strat.name] = strat
        return by_name, strat_by_name

    def _mean(self, name: str) -> float:
        n = self._counts.get(name, 0)
        return self._sum_rewards[name] / n if n > 0 else 0.0

    def stats(self) -> dict[str, dict[str, float]]:
        return {
            name: {
                "count": self._counts[name],
                "mean_reward": self._mean(name),
            }
            for name in self._counts
        }
