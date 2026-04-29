"""Experiment 8 — Rounds x Bandit Interaction.

Sweeps {linucb, ucb1, uniform} x {100, 500, 2000} rounds, using the same
bandit at both week and repair levels. 3 seeds x 5 dev instances per cell.

The "money plot": penalty vs. rounds, one curve per bandit. Tests whether
LinUCB's advantage over non-contextual baselines shrinks as rounds → ∞.

Example:
    PYTHONPATH=src python scripts/run_exp8.py \
        --week-checkpoint runs/final_week.npz \
        --repair-checkpoint runs/final_repair.npz \
        --workers 4 --out-dir runs/exp8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

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

BANDITS = ["linucb", "ucb1", "uniform"]
ROUND_BUDGETS = [100, 500, 2000]

WEEK_ARMS_FACTORY = lambda: [  # noqa: E731
    CoverageFirstArm(),
    FatigueAwareArm(),
    WeekendBalancingArm(),
    PreferenceRespectingArm(),
]


def _run_cell(
    bandit: str,
    rounds: int,
    seed: int,
    max_instances: int,
    week_checkpoint: str | None,
    repair_checkpoint: str | None,
    week_reward_scale: float,
    repair_reward_scale: float,
    out_path: str,
) -> dict:
    """Run one (bandit, rounds, seed) cell. Called in a worker process."""
    week_lin = LinUCB.load(week_checkpoint) if bandit == "linucb" else None
    repair_lin = LinUCB.load(repair_checkpoint) if bandit == "linucb" else None

    stream = split_instances("dev", seed=seed, shuffle=True, week_combos_per_scenario=2)
    instance_results = []
    n = 0

    for inst in stream:
        if n >= max_instances:
            break
        try:
            arms = WEEK_ARMS_FACTORY()

            # Week stage
            if bandit == "linucb":
                week_out = run_week_level(
                    scenario=inst.scenario,
                    initial_history=inst.initial_history,
                    week_data_list=inst.weeks,
                    arms=arms,
                    bandit=week_lin,
                    reward_scale=week_reward_scale,
                )
            else:
                selector = get_bandit(
                    bandit,
                    strategy_names=[a.name for a in arms],
                    seed=seed,
                )
                week_out = run_week_level_baseline(
                    scenario=inst.scenario,
                    initial_history=inst.initial_history,
                    week_data_list=inst.weeks,
                    arms=arms,
                    selector=selector,
                    reward_scale=week_reward_scale,
                )
            schedule = week_out["schedule"]
            post_week_penalty = week_out["total_penalty"]

            # Repair stage
            strategies = build_all_strategies(
                inst.scenario, inst.initial_history, inst.weeks, seed=seed,
            )
            if bandit == "linucb":
                selector = LinUCBRepairSelector(
                    strategy_names=[s.name for s in strategies],
                    alpha=repair_lin.alpha,
                    reward_scale=repair_reward_scale,
                    seed=seed,
                    linucb=repair_lin,
                )
            else:
                selector = get_bandit(
                    bandit,
                    strategy_names=[s.name for s in strategies],
                    seed=seed,
                )

            result = run_repairs(
                scenario=inst.scenario,
                history=inst.initial_history,
                week_data_list=inst.weeks,
                strategies=strategies,
                schedule=schedule,
                selector=selector,
                num_rounds=rounds,
                seed=seed,
            )
            final = compute_penalty(
                result.schedule, inst.scenario, inst.weeks, inst.initial_history,
            )
            instance_results.append({
                "dataset": inst.dataset_name,
                "post_week_penalty": post_week_penalty,
                "final_penalty": final.total,
                "delta": post_week_penalty - final.total,
            })
            n += 1
        except Exception as e:
            print(f"  [SKIP {inst.dataset_name} b={bandit} r={rounds} s={seed}] {e}",
                  file=sys.stderr)

    deltas = [r["delta"] for r in instance_results]
    finals = [r["final_penalty"] for r in instance_results]
    summary = {
        "bandit": bandit,
        "rounds": rounds,
        "seed": seed,
        "n": len(instance_results),
        "mean_delta": round(float(np.mean(deltas)), 2) if deltas else float("nan"),
        "mean_final_penalty": round(float(np.mean(finals)), 2) if finals else float("nan"),
        "instances": instance_results,
    }
    Path(out_path).write_text(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description="Exp 8: rounds x bandit interaction.")
    p.add_argument("--week-checkpoint", default="runs/final_week.npz")
    p.add_argument("--repair-checkpoint", default="runs/final_repair.npz")
    p.add_argument("--rounds", type=int, nargs="+", default=ROUND_BUDGETS)
    p.add_argument("--bandits", nargs="+", default=BANDITS, choices=BANDITS)
    p.add_argument("--max-instances", type=int, default=5)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--week-reward-scale", type=float, default=1000.0)
    p.add_argument("--repair-reward-scale", type=float, default=50.0)
    p.add_argument("--out-dir", default="runs/exp8")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = list(product(args.bandits, args.rounds, args.seeds))
    print(f"[exp8] {len(jobs)} jobs  workers={args.workers}  "
          f"max_instances={args.max_instances}")

    futures = {}
    completed = 0
    all_summaries = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for bandit, rounds, seed in jobs:
            tag = f"b{bandit}_r{rounds}_s{seed}"
            fut = pool.submit(
                _run_cell,
                bandit=bandit,
                rounds=rounds,
                seed=seed,
                max_instances=args.max_instances,
                week_checkpoint=args.week_checkpoint,
                repair_checkpoint=args.repair_checkpoint,
                week_reward_scale=args.week_reward_scale,
                repair_reward_scale=args.repair_reward_scale,
                out_path=str(out_dir / f"{tag}.json"),
            )
            futures[fut] = tag

        for fut in as_completed(futures):
            tag = futures[fut]
            completed += 1
            try:
                s = fut.result()
                print(f"  [{completed}/{len(jobs)}] {tag}  "
                      f"meanΔ={s['mean_delta']}  meanFinal={s['mean_final_penalty']}  n={s['n']}")
                all_summaries.append(s)
            except Exception as e:
                print(f"  [{completed}/{len(jobs)}] {tag} FAILED: {e}", file=sys.stderr)

    # Aggregate across seeds: mean per (bandit, rounds)
    from collections import defaultdict
    cell_data: dict[tuple, list] = defaultdict(list)
    for s in all_summaries:
        cell_data[(s["bandit"], s["rounds"])].append(s)

    grid_rows = []
    for (bandit, rounds), seed_results in sorted(cell_data.items(), key=lambda x: (x[0][0], x[0][1])):
        all_finals = [r["mean_final_penalty"] for r in seed_results if not np.isnan(r["mean_final_penalty"])]
        all_deltas = [r["mean_delta"] for r in seed_results if not np.isnan(r["mean_delta"])]
        grid_rows.append({
            "bandit": bandit,
            "rounds": rounds,
            "seeds_completed": len(seed_results),
            "mean_delta": round(float(np.mean(all_deltas)), 2) if all_deltas else float("nan"),
            "mean_final_penalty": round(float(np.mean(all_finals)), 2) if all_finals else float("nan"),
        })

    (out_dir / "grid_summary.json").write_text(json.dumps(grid_rows, indent=2))

    # Print table: bandits as rows, rounds as columns
    round_budgets = sorted(set(args.rounds))
    print("\n=== Exp 8: Mean Final Penalty (lower is better) ===")
    print(f"{'bandit':>14}", end="")
    for r in round_budgets:
        print(f"  rounds={r:>5}", end="")
    print()
    for bandit in args.bandits:
        print(f"  {bandit:>12}", end="")
        for r in round_budgets:
            row = next((x for x in grid_rows if x["bandit"] == bandit and x["rounds"] == r), None)
            val = f"{row['mean_final_penalty']:.0f}" if row and not np.isnan(row["mean_final_penalty"]) else "N/A"
            print(f"  {val:>12}", end="")
        print()

    print(f"\nWrote per-cell JSONs and grid_summary.json to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
