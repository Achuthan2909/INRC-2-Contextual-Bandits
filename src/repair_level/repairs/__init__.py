"""Repair strategies for the repair-level bandit loop.

``build_all_strategies`` walks every concrete ``RepairStrategy`` subclass in
this package and instantiates it by name-matching its ``__init__`` parameters
against the scenario / history / week data passed in. Any subclass with
required arguments we don't recognise is silently skipped.
"""
from __future__ import annotations

import inspect
from typing import Any

from repair_level.repairs.base import RepairStrategy


_STRATEGY_MODULES: list[str] = [
    "repair_level.repairs.coverage",
    "repair_level.repairs.consecutive_work",
    "repair_level.repairs.consecutive_shift",
    "repair_level.repairs.days_off",
    "repair_level.repairs.preference",
    "repair_level.repairs.total_assignments",
    "repair_level.repairs.weekend",
    "repair_level.repairs.catchall",
]


def _discover_strategy_classes() -> list[type[RepairStrategy]]:
    classes: list[type[RepairStrategy]] = []
    seen: set[type] = set()
    for mod_path in _STRATEGY_MODULES:
        mod = __import__(mod_path, fromlist=["*"])
        for name, obj in mod.__dict__.items():
            if (
                isinstance(obj, type)
                and issubclass(obj, RepairStrategy)
                and obj is not RepairStrategy
                and not name.startswith("_")
                and obj not in seen
            ):
                seen.add(obj)
                classes.append(obj)
    return classes


def build_all_strategies(
    scenario: dict[str, Any],
    history: dict[str, Any],
    week_data_list: list[dict[str, Any]],
    seed: int | None = None,
) -> list[RepairStrategy]:
    """Instantiate every concrete repair strategy that can be auto-wired."""
    strategies: list[RepairStrategy] = []
    for cls in _discover_strategy_classes():
        try:
            sig = inspect.signature(cls.__init__)
        except (TypeError, ValueError):
            continue

        kwargs: dict[str, Any] = {}
        skip = False
        for p in sig.parameters.values():
            if p.name == "self":
                continue
            if p.name == "scenario":
                kwargs[p.name] = scenario
            elif p.name in ("initial_history", "history"):
                kwargs[p.name] = history
            elif p.name in ("week_data_list", "weeks"):
                kwargs[p.name] = week_data_list
            elif p.name == "seed":
                kwargs[p.name] = seed
            elif p.default is inspect.Parameter.empty:
                skip = True
                break
        if not skip:
            strategies.append(cls(**kwargs))
    return strategies


__all__ = ["build_all_strategies", "RepairStrategy"]
