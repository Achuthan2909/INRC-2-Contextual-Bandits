"""Shared evaluation utilities for experiment drivers.

Functions here run the full week+repair pipeline (or just one stage) on a
fixed list of held-out instances and return per-instance metrics. They are
designed to be called from per-experiment drivers and to write JSON
artifacts that plotting scripts consume.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bandit import get_bandit  # noqa: E402
from bandit.linucb import LinUCB  # noqa: E402
from data.splits import split_instances  # noqa: E402
from repair_level.init import generate_initial_schedule  # noqa: E402
from repair_level.linucb_selector import LinUCBRepairSelector  # noqa: E402
from repair_level.repairs import build_all_strategies  # noqa: E402
from repair_level.runner import run_repairs  # noqa: E402
from schedule.penalty import compute_penalty  # noqa: E402
from week_level.arms import (  # noqa: E402
    CoverageFirstArm,
    FatigueAwareArm,
    PreferenceRespectingArm,
    WeekendBalancingArm,
)
from week_level.runner import run_week_level  # noqa: E402
from week_level.runner_baseline import run_week_level_baseline  # noqa: E402


def week_arms() -> list:
    return [
        CoverageFirstArm(),
        FatigueAwareArm(),
        WeekendBalancingArm(),
        PreferenceRespectingArm(),
    ]


def get_eval_instances(split: str, n: int, seed: int = 0) -> list:
    """Return up to ``n`` deterministic eval instances from ``split``."""
    out = []
    for inst in split_instances(split, seed=seed, shuffle=True, week_combos_per_scenario=1):
        out.append(inst)
        if len(out) >= n:
            break
    return out


def run_pipeline_one(
    inst,
    *,
    week_bandit: str,
    week_checkpoint: str | None,
    repair_bandit: str,
    repair_checkpoint: str | None,
    rounds: int,
    seed: int,
    week_reward_scale: float = 1000.0,
    repair_reward_scale: float = 50.0,
) -> dict[str, Any]:
    """Run the full pipeline on one instance and return metrics."""
    arms = week_arms()
    t0 = time.perf_counter()

    # ---- week stage ----
    if week_bandit == "linucb":
        if not week_checkpoint:
            raise ValueError("week_bandit=linucb needs week_checkpoint")
        lin_w = LinUCB.load(week_checkpoint)
        week_out = run_week_level(
            scenario=inst.scenario,
            initial_history=inst.initial_history,
            week_data_list=inst.weeks,
            arms=arms,
            bandit=lin_w,
            reward_scale=week_reward_scale,
        )
    else:
        sel = get_bandit(week_bandit, strategy_names=[a.name for a in arms], seed=seed)
        week_out = run_week_level_baseline(
            scenario=inst.scenario,
            initial_history=inst.initial_history,
            week_data_list=inst.weeks,
            arms=arms,
            selector=sel,
            reward_scale=week_reward_scale,
        )
    schedule = week_out["schedule"]
    post_week_penalty = week_out["total_penalty"]
    t_week = time.perf_counter() - t0

    # ---- repair stage ----
    t1 = time.perf_counter()
    strategies = build_all_strategies(
        inst.scenario, inst.initial_history, inst.weeks, seed=seed,
    )
    if repair_bandit == "linucb":
        if not repair_checkpoint:
            raise ValueError("repair_bandit=linucb needs repair_checkpoint")
        lin_r = LinUCB.load(repair_checkpoint)
        selector = LinUCBRepairSelector(
            strategy_names=[s.name for s in strategies],
            alpha=lin_r.alpha,
            reward_scale=repair_reward_scale,
            seed=seed,
            linucb=lin_r,
        )
    else:
        selector = get_bandit(
            repair_bandit, strategy_names=[s.name for s in strategies], seed=seed,
        )
    rep = run_repairs(
        scenario=inst.scenario,
        history=inst.initial_history,
        week_data_list=inst.weeks,
        strategies=strategies,
        schedule=schedule,
        selector=selector,
        num_rounds=rounds,
        seed=seed,
    )
    t_rep = time.perf_counter() - t1

    final_pen = compute_penalty(
        rep.schedule, inst.scenario, inst.weeks, inst.initial_history,
    )
    return {
        "dataset": inst.dataset_name,
        "post_week_penalty": int(post_week_penalty),
        "final_penalty": int(final_pen.total),
        "delta_repair": int(post_week_penalty - final_pen.total),
        "rounds_run": int(rep.rounds_run),
        "strategy_counts": dict(rep.strategy_counts),
        "arms_picked": list(week_out["arms_picked"]),
        "soft_breakdown": {
            "s1": int(final_pen.s1_optimal_coverage),
            "s2": int(final_pen.s2_consecutive),
            "s3": int(final_pen.s3_days_off),
            "s4": int(final_pen.s4_preferences),
            "s5": int(final_pen.s5_complete_weekends),
            "s6": int(final_pen.s6_total_assignments),
            "s7": int(final_pen.s7_working_weekends),
        },
        "hard_violations": dict(final_pen.hard),
        "t_week_s": t_week,
        "t_repair_s": t_rep,
        "penalty_trajectory": list(rep.penalty_trajectory),
    }


def run_repair_only(
    inst,
    schedule,
    *,
    repair_bandit: str,
    repair_checkpoint: str | None,
    rounds: int,
    seed: int,
    repair_reward_scale: float = 50.0,
) -> dict[str, Any]:
    """Run only repair, given a pre-built schedule (for Exp 11 / OOD)."""
    strategies = build_all_strategies(
        inst.scenario, inst.initial_history, inst.weeks, seed=seed,
    )
    if repair_bandit == "linucb":
        lin_r = LinUCB.load(repair_checkpoint)
        selector = LinUCBRepairSelector(
            strategy_names=[s.name for s in strategies],
            alpha=lin_r.alpha,
            reward_scale=repair_reward_scale,
            seed=seed,
            linucb=lin_r,
        )
    else:
        selector = get_bandit(
            repair_bandit, strategy_names=[s.name for s in strategies], seed=seed,
        )

    init_pen = compute_penalty(
        schedule, inst.scenario, inst.weeks, inst.initial_history,
    )
    t0 = time.perf_counter()
    rep = run_repairs(
        scenario=inst.scenario,
        history=inst.initial_history,
        week_data_list=inst.weeks,
        strategies=strategies,
        schedule=schedule,
        selector=selector,
        num_rounds=rounds,
        seed=seed,
    )
    t = time.perf_counter() - t0
    final_pen = compute_penalty(
        rep.schedule, inst.scenario, inst.weeks, inst.initial_history,
    )
    return {
        "dataset": inst.dataset_name,
        "initial_penalty": int(init_pen.total),
        "final_penalty": int(final_pen.total),
        "delta_repair": int(init_pen.total - final_pen.total),
        "rounds_run": int(rep.rounds_run),
        "strategy_counts": dict(rep.strategy_counts),
        "penalty_trajectory": list(rep.penalty_trajectory),
        "hard_violations": dict(final_pen.hard),
        "t_repair_s": t,
    }


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))
