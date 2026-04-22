"""Bandit selectors for the repair-level loop.

Use :func:`get_bandit` to construct a selector by name from the registry,
or import the concrete classes directly.
"""
from __future__ import annotations

from typing import Any

from bandit.base import BanditSelector
from bandit.uniform import UniformSelector
from bandit.epsilon_greedy import EpsilonGreedySelector
from bandit.ucb import UCB1Selector
from bandit.thompson import ThompsonSamplingSelector
from bandit.softmax import SoftmaxSelector
from bandit.exp3 import EXP3Selector
from bandit.linucb import LinUCB

_REGISTRY: dict[str, type[BanditSelector]] = {
    "random": UniformSelector,
    "uniform": UniformSelector,
    "epsilon_greedy": EpsilonGreedySelector,
    "eps_greedy": EpsilonGreedySelector,
    "ucb1": UCB1Selector,
    "ucb": UCB1Selector,
    "thompson": ThompsonSamplingSelector,
    "ts": ThompsonSamplingSelector,
    "softmax": SoftmaxSelector,
    "boltzmann": SoftmaxSelector,
    "exp3": EXP3Selector,
}


def available() -> list[str]:
    """Return the list of canonical bandit names."""
    return sorted({
        "random", "epsilon_greedy", "ucb1", "thompson", "softmax", "exp3",
    })


def get_bandit(
    name: str,
    strategy_names: list[str],
    seed: int | None = None,
    **kwargs: Any,
) -> BanditSelector:
    """Build a bandit selector by name.

    Any extra ``**kwargs`` are forwarded to the selector's constructor (e.g.
    ``epsilon=0.2`` for ε-greedy, ``c=1.4`` for UCB1, ``tau=0.5`` for softmax).
    """
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown bandit '{name}'. Available: {available()}"
        )
    cls = _REGISTRY[key]
    return cls(strategy_names=strategy_names, seed=seed, **kwargs)


__all__ = [
    "BanditSelector",
    "UniformSelector",
    "EpsilonGreedySelector",
    "UCB1Selector",
    "ThompsonSamplingSelector",
    "SoftmaxSelector",
    "EXP3Selector",
    "LinUCB",
    "available",
    "get_bandit",
]
