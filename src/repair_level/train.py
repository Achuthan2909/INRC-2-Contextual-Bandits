"""Cross-instance repair-level LinUCB training.

Streams instances from :func:`data.instances.enumerate_instances`, runs the
repair loop on each with a single shared :class:`LinUCBRepairSelector`
(so ``A_a`` and ``b_a`` accumulate across instances), and checkpoints the
trained bandit plus a trajectory sidecar for diagnostics.

This is the repair-level analog of ``src/week_level/train.py``. The key
difference is that each instance contributes *many* bandit rounds (one per
repair iteration, not one per week), so per-instance wall-clock is much
higher than at the week level.

CLI:
    PYTHONPATH=src python -m repair_level.train --help
"""
from __future__ import annotations

import argparse
import json
import logging
import random as _random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from bandit.linucb import LinUCB
from data.instances import enumerate_instances
from data.splits import SPLITS, split_instances
from repair_level.context import FEATURE_LABELS, build_repair_context
from repair_level.init import generate_initial_schedule
from repair_level.linucb_selector import LinUCBRepairSelector
from repair_level.repairs import build_all_strategies
from repair_level.repairs.base import RepairStrategy
from repair_level.runner import run_repairs
from schedule.penalty import PenaltyResult


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Warm-start wrapper: round-robin picks for the first N rounds, then LinUCB.
# We wrap the real ``LinUCBRepairSelector`` so the inner bandit still sees
# every (context, reward) pair during warm-up — we just override *which arm*
# gets picked to guarantee coverage of all 18 arms before LinUCB takes over.
# ---------------------------------------------------------------------------
class _WarmStartSelector:
    """Forces round-robin picks for the first ``warm_start_rounds`` calls.

    After warm-up, delegates entirely to the wrapped ``LinUCBRepairSelector``.
    The wrapped selector's ``_last_arm_idx`` / ``_last_ctx`` are set during
    warm-up so its ``update()`` continues to feed the LinUCB correctly.
    """

    def __init__(self, inner: LinUCBRepairSelector, warm_start_rounds: int) -> None:
        self.inner = inner
        self.warm_start_rounds = int(warm_start_rounds)
        self._calls = 0
        # Exposed for the trainer's bookkeeping.
        self.pick_counts: dict[str, int] = dict(inner.pick_counts)

    # --- runner hooks pass-through ----------------------------------------

    def observe_state(
        self, penalty: PenaltyResult, round_idx: int, total_rounds: int
    ) -> None:
        self.inner.observe_state(penalty, round_idx, total_rounds)

    def update(self, strategy_name: str, reward: float) -> None:
        self.inner.update(strategy_name, reward)

    def __call__(
        self,
        strategies: list[RepairStrategy],
        candidates: list[tuple[RepairStrategy, dict[str, Any]]],
    ) -> tuple[RepairStrategy, dict[str, Any]]:
        if self._calls >= self.warm_start_rounds:
            # Past warm-up: delegate fully to the LinUCB selector.
            out = self.inner(strategies, candidates)
            self.pick_counts = dict(self.inner.pick_counts)
            self._calls += 1
            return out

        # Warm-up: pick the next arm in round-robin order *among arms that
        # currently have candidates*. If the scheduled arm has none, walk
        # forward until we find one that does.
        by_strategy: dict[str, list[dict[str, Any]]] = {}
        strat_by_name: dict[str, RepairStrategy] = {}
        for strat, viol in candidates:
            by_strategy.setdefault(strat.name, []).append(viol)
            strat_by_name.setdefault(strat.name, strat)

        order = self.inner.strategy_names
        if not order:
            raise RuntimeError("inner selector has no strategy_names")

        n = len(order)
        chosen_name: str | None = None
        for offset in range(n):
            name = order[(self._calls + offset) % n]
            if name in by_strategy:
                chosen_name = name
                break
        if chosen_name is None:
            # Shouldn't happen — runner only calls selector when candidates exist.
            strat, viol = candidates[0]
            chosen_name = strat.name
            by_strategy[chosen_name] = [viol]
            strat_by_name[chosen_name] = strat

        arm_idx = self.inner._name_to_idx[chosen_name]
        ctx = self.inner._ctx
        if ctx is None:
            ctx = np.zeros(len(FEATURE_LABELS), dtype=np.float64)
        self.inner._last_arm_idx = arm_idx
        self.inner._last_ctx = ctx.copy()
        self.inner.pick_counts[chosen_name] = (
            self.inner.pick_counts.get(chosen_name, 0) + 1
        )
        self.pick_counts = dict(self.inner.pick_counts)

        chosen_strategy = strat_by_name[chosen_name]
        chosen_violation = self.inner._rng.choice(by_strategy[chosen_name])
        self._calls += 1
        return chosen_strategy, chosen_violation


# ---------------------------------------------------------------------------
# Main training entry point.
# ---------------------------------------------------------------------------
def train_linucb_repair(
    *,
    split: str = "train",
    dataset_root: str | None = None,
    alpha: float = 1.0,
    max_instances: int | None = None,
    week_combos_per_scenario: int = 20,
    seed: int = 0,
    reward_scale: float = 50.0,
    num_rounds: int = 500,
    checkpoint_path: str | Path = "runs/linucb_repair_level.npz",
    log_every: int = 10,
    warm_start_rounds: int = 0,
    with_replacement: bool = False,
    lazy_find_violations: bool = False,
) -> LinUCB:
    """Stream instances, update one shared repair-level LinUCB, save checkpoint.

    Parameters
    ----------
    num_rounds
        Number of repair rounds per instance (the inner repair loop's budget).
    reward_scale
        Divides raw ``r_i = P_i - P_{i+1}`` before the ridge update. 50 is a
        good starting value for ``n030w4``-scale instances.
    warm_start_rounds
        Counted in *rounds*, not instances. Forces round-robin picks until
        every arm has had a chance to be updated with real contexts.
    """
    t_train0 = time.perf_counter()

    # Either a named split (train/dev/val/test) or an explicit dataset_root.
    if dataset_root is not None:
        stream = enumerate_instances(
            dataset_root,
            seed=seed,
            shuffle=True,
            week_combos_per_scenario=week_combos_per_scenario,
            with_replacement=with_replacement,
        )
    else:
        stream = split_instances(
            split,
            seed=seed,
            shuffle=True,
            week_combos_per_scenario=week_combos_per_scenario,
            with_replacement=with_replacement,
        )

    # One shared LinUCB over all training instances; strategy names come from
    # the first instance (they're structural — same set for every instance).
    linucb: LinUCB | None = None
    inner_selector: LinUCBRepairSelector | None = None
    warm_selector: _WarmStartSelector | None = None
    strategy_names: list[str] = []

    pick_counts: Counter[str] = Counter()
    pick_trajectory: list[str] = []
    scaled_reward_values: list[float] = []
    theta_snapshots: list[list[list[float]]] = []
    snapshot_rounds: list[int] = []
    initial_penalties: list[int] = []
    final_penalties: list[int] = []
    instance_count = 0
    total_rounds_run = 0

    for inst in stream:
        if max_instances is not None and instance_count >= max_instances:
            break

        try:
            strategies = build_all_strategies(
                inst.scenario, inst.initial_history, inst.weeks, seed=seed,
            )
            if not strategies:
                log.warning(
                    "No repair strategies built for %s; skipping.",
                    getattr(inst, "dataset_name", "?"),
                )
                continue

            names_here = [s.name for s in strategies]

            # Lazy one-time init: first instance fixes K', strategy order, etc.
            if linucb is None:
                strategy_names = names_here
                linucb = LinUCB(
                    num_arms=len(strategy_names),
                    context_dim=len(FEATURE_LABELS),
                    alpha=alpha,
                    seed=seed,
                )
                inner_selector = LinUCBRepairSelector(
                    strategy_names=strategy_names,
                    alpha=alpha,
                    reward_scale=reward_scale,
                    seed=seed,
                    linucb=linucb,
                )
                warm_selector = _WarmStartSelector(
                    inner_selector, warm_start_rounds=warm_start_rounds,
                )

            assert linucb is not None
            assert inner_selector is not None
            assert warm_selector is not None

            # Strategy set must be stable across instances — otherwise arm
            # index i points to a different heuristic on different instances
            # and the learned θ_a is meaningless.
            if names_here != strategy_names:
                log.warning(
                    "Strategy set differs from first instance on %s; skipping.",
                    getattr(inst, "dataset_name", "?"),
                )
                continue

            t0 = time.perf_counter()
            schedule = generate_initial_schedule(
                inst.scenario, inst.initial_history, inst.weeks,
            )
            result = run_repairs(
                scenario=inst.scenario,
                history=inst.initial_history,
                week_data_list=inst.weeks,
                strategies=strategies,
                schedule=schedule,
                selector=warm_selector,
                num_rounds=num_rounds,
                seed=seed + instance_count,
                lazy_find_violations=lazy_find_violations,
            )
            dt = time.perf_counter() - t0

            # Collect scaled rewards from the penalty trajectory:
            #   r_i = P_i - P_{i+1},  scaled = r_i / reward_scale
            traj = result.penalty_trajectory
            rs = float(reward_scale) if reward_scale else 1.0
            for i in range(len(traj) - 1):
                scaled_reward_values.append(float(traj[i] - traj[i + 1]) / rs)

            # Pick trajectory from this instance's strategy_counts is already
            # accumulated; we want per-round ordering, which the runner does
            # not expose. Fall back to per-instance aggregate counts — good
            # enough for arm-usage plots.
            for name, n in result.strategy_counts.items():
                pick_counts[name] += n
                pick_trajectory.extend([name] * n)

            initial_penalties.append(result.initial_penalty)
            final_penalties.append(result.final_penalty)
            total_rounds_run += result.rounds_run
            instance_count += 1

            if log_every > 0 and instance_count % log_every == 0:
                theta_snapshots.append(
                    [linucb.theta(i).tolist() for i in range(linucb.num_arms)]
                )
                snapshot_rounds.append(total_rounds_run)

            log.info(
                "[%d] %s | %.2fs | rounds=%d | P: %d -> %d (Δ=%+d) | hard=%s",
                instance_count,
                getattr(inst, "dataset_name", "?"),
                dt,
                result.rounds_run,
                result.initial_penalty,
                result.final_penalty,
                result.initial_penalty - result.final_penalty,
                dict(result.hard_violations),
            )

            if log_every > 0 and instance_count % log_every == 0:
                tail = scaled_reward_values[-2000:]
                mr = float(np.mean(tail)) if tail else 0.0
                top = ", ".join(
                    f"{n}={c}"
                    for n, c in sorted(pick_counts.items(), key=lambda kv: -kv[1])[:5]
                )
                log.info(
                    "[%d] mean_scaled_reward(last≤2000)=%.4f  top5_arms={%s}",
                    instance_count, mr, top,
                )
        except Exception as e:
            log.warning(
                "Skipping instance %s: %s",
                getattr(inst, "dataset_name", "?"), e,
            )
            continue

    if linucb is None:
        raise RuntimeError("No instance completed training (empty stream or all skipped)")

    training_config: dict[str, Any] = {
        "split": split,
        "dataset_root": dataset_root,
        "alpha": alpha,
        "max_instances": max_instances,
        "week_combos_per_scenario": week_combos_per_scenario,
        "seed": seed,
        "reward_scale": float(reward_scale),
        "num_rounds": num_rounds,
        "warm_start_rounds": warm_start_rounds,
        "with_replacement": with_replacement,
        "lazy_find_violations": lazy_find_violations,
        "log_every": log_every,
        "instances_run": instance_count,
        "total_bandit_rounds": total_rounds_run,
        "wall_clock_s": time.perf_counter() - t_train0,
        "arm_names": list(strategy_names),
    }

    metadata = {
        "feature_labels": list(FEATURE_LABELS),
        "reward_scale": float(reward_scale),
        "training_config": training_config,
        "level": "repair",
    }

    linucb.save(checkpoint_path, metadata=metadata)

    sidecar_path = Path(checkpoint_path).with_suffix(".trajectory.json")
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(json.dumps({
        "arm_names": list(strategy_names),
        "feature_labels": list(FEATURE_LABELS),
        "pick_trajectory": pick_trajectory,
        "reward_trajectory_scaled": scaled_reward_values,
        "theta_snapshots": theta_snapshots,
        "snapshot_rounds": snapshot_rounds,
        "initial_penalties": initial_penalties,
        "final_penalties": final_penalties,
        "warm_start_rounds": warm_start_rounds,
        "training_config": training_config,
    }))
    log.info(
        "Saved checkpoint to %s (%.1fs, %d instances, %d rounds)",
        checkpoint_path,
        training_config["wall_clock_s"],
        instance_count,
        total_rounds_run,
    )
    return linucb


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(
        description="Train repair-level LinUCB (cross-instance).",
    )
    p.add_argument(
        "--split", choices=SPLITS, default="train",
        help="Named split (train/dev/val/test). Ignored if --dataset-root is given.",
    )
    p.add_argument(
        "--dataset-root", default=None,
        help="Explicit dataset root; overrides --split.",
    )
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--max-instances", type=int, default=None)
    p.add_argument("--week-combos-per-scenario", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--reward-scale", type=float, default=50.0,
        help="Divides raw r_i = P_i - P_{i+1} before the ridge update.",
    )
    p.add_argument(
        "--num-rounds", type=int, default=500,
        help="Repair-loop budget per instance.",
    )
    p.add_argument(
        "--checkpoint", default="runs/linucb_repair_level.npz",
    )
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument(
        "--warm-start-rounds", type=int, default=0,
        help="Force round-robin picks for the first N bandit rounds (across instances) before letting LinUCB choose.",
    )
    p.add_argument(
        "--with-replacement", action="store_true",
        help="Allow repeated WD files in a week-combo.",
    )
    p.add_argument(
        "--lazy-find-violations", action="store_true",
        help="Pick arm first, only scan that arm's violations (faster for many arms).",
    )
    args = p.parse_args()

    train_linucb_repair(
        split=args.split,
        dataset_root=args.dataset_root,
        alpha=args.alpha,
        max_instances=args.max_instances,
        week_combos_per_scenario=args.week_combos_per_scenario,
        seed=args.seed,
        reward_scale=args.reward_scale,
        num_rounds=args.num_rounds,
        checkpoint_path=args.checkpoint,
        log_every=args.log_every,
        warm_start_rounds=args.warm_start_rounds,
        with_replacement=args.with_replacement,
        lazy_find_violations=args.lazy_find_violations,
    )


if __name__ == "__main__":
    _cli()
