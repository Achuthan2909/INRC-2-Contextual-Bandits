"""Experiment 2 — Bandit vs. Baseline Grid (5x5).

Evaluates all 25 (week_bandit x repair_bandit) combinations on the dev split
across 3 seeds. Parallelises at the cell level with a configurable worker cap
so RAM usage stays bounded.

Each job = one (week_bandit, repair_bandit, seed) run on --max-instances dev
instances. Results are saved per-cell and a summary table is printed.

Example:
    PYTHONPATH=src python scripts/run_exp2.py \
        --week-checkpoint runs/final_week.npz \
        --repair-checkpoint runs/final_repair.npz \
        --rounds 500 --max-instances 5 --workers 4 \
        --out-dir runs/exp2
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

from bandit import available as available_bandits, get_bandit  # noqa: E402
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

BANDITS = ["uniform", "ucb1", "epsilon_greedy", "thompson", "linucb"]

WEEK_ARMS_FACTORY = lambda: [  # noqa: E731
    CoverageFirstArm(),
    FatigueAwareArm(),
    WeekendBalancingArm(),
    PreferenceRespectingArm(),
]


def _run_cell(
    week_bandit: str,
    repair_bandit: str,
    seed: int,
    rounds: int,
    max_instances: int,
    week_checkpoint: str | None,
    repair_checkpoint: str | None,
    week_reward_scale: float,
    repair_reward_scale: float,
    out_path: str,
) -> dict:
    """Run one (week_bandit, repair_bandit, seed) cell. Called in a worker process."""
    # Load checkpoints inside worker so each process owns its own objects.
    week_lin = LinUCB.load(week_checkpoint) if week_bandit == "linucb" else None
    repair_lin = LinUCB.load(repair_checkpoint) if repair_bandit == "linucb" else None

    stream = split_instances("dev", seed=seed, shuffle=True, week_combos_per_scenario=2)
    instance_results = []
    n = 0

    for inst in stream:
        if n >= max_instances:
            break
        try:
            arms = WEEK_ARMS_FACTORY()

            # Week stage
            if week_bandit == "linucb":
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
                    week_bandit,
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
            if repair_bandit == "linucb":
                selector = LinUCBRepairSelector(
                    strategy_names=[s.name for s in strategies],
                    alpha=repair_lin.alpha,
                    reward_scale=repair_reward_scale,
                    seed=seed,
                    linucb=repair_lin,
                )
            else:
                selector = get_bandit(
                    repair_bandit,
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
            print(f"  [SKIP {inst.dataset_name} w={week_bandit} r={repair_bandit} s={seed}] {e}",
                  file=sys.stderr)

    deltas = [r["delta"] for r in instance_results]
    finals = [r["final_penalty"] for r in instance_results]
    summary = {
        "week_bandit": week_bandit,
        "repair_bandit": repair_bandit,
        "seed": seed,
        "n": len(instance_results),
        "mean_delta": round(float(np.mean(deltas)), 2) if deltas else float("nan"),
        "mean_final_penalty": round(float(np.mean(finals)), 2) if finals else float("nan"),
        "instances": instance_results,
    }
    Path(out_path).write_text(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description="Exp 2: 5x5 bandit grid evaluation.")
    p.add_argument("--week-checkpoint", default="runs/final_week.npz")
    p.add_argument("--repair-checkpoint", default="runs/final_repair.npz")
    p.add_argument("--rounds", type=int, default=500)
    p.add_argument("--max-instances", type=int, default=5,
                   help="Dev instances per (cell, seed). Default 5 → 375 total runs.")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel worker processes. Keep ≤4 on 8GB RAM.")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--week-reward-scale", type=float, default=1000.0)
    p.add_argument("--repair-reward-scale", type=float, default=50.0)
    p.add_argument("--out-dir", default="runs/exp2")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = list(product(BANDITS, BANDITS, args.seeds))
    print(f"[exp2] {len(jobs)} jobs  workers={args.workers}  "
          f"max_instances={args.max_instances}  rounds={args.rounds}")

    futures = {}
    completed = 0
    all_summaries = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for wb, rb, seed in jobs:
            tag = f"w{wb}_r{rb}_s{seed}"
            out_path = str(out_dir / f"{tag}.json")
            fut = pool.submit(
                _run_cell,
                week_bandit=wb,
                repair_bandit=rb,
                seed=seed,
                rounds=args.rounds,
                max_instances=args.max_instances,
                week_checkpoint=args.week_checkpoint,
                repair_checkpoint=args.repair_checkpoint,
                week_reward_scale=args.week_reward_scale,
                repair_reward_scale=args.repair_reward_scale,
                out_path=out_path,
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

    # Aggregate across seeds per cell
    from collections import defaultdict
    cell_data: dict[tuple, list] = defaultdict(list)
    for s in all_summaries:
        cell_data[(s["week_bandit"], s["repair_bandit"])].append(s)

    grid_rows = []
    for (wb, rb), seed_results in sorted(cell_data.items()):
        all_deltas = [r["mean_delta"] for r in seed_results if not np.isnan(r["mean_delta"])]
        all_finals = [r["mean_final_penalty"] for r in seed_results if not np.isnan(r["mean_final_penalty"])]
        grid_rows.append({
            "week_bandit": wb,
            "repair_bandit": rb,
            "seeds_completed": len(seed_results),
            "mean_delta": round(float(np.mean(all_deltas)), 2) if all_deltas else float("nan"),
            "mean_final_penalty": round(float(np.mean(all_finals)), 2) if all_finals else float("nan"),
        })

    (out_dir / "grid_summary.json").write_text(json.dumps(grid_rows, indent=2))

    # Print 5x5 grid table
    print("\n=== Exp 2: Mean Final Penalty (lower is better) ===")
    print(f"{'':>16}", end="")
    for rb in BANDITS:
        print(f"  {rb:>14}", end="")
    print()
    for wb in BANDITS:
        print(f"  week={wb:>12}", end="")
        for rb in BANDITS:
            row = next((r for r in grid_rows if r["week_bandit"] == wb and r["repair_bandit"] == rb), None)
            val = f"{row['mean_final_penalty']:.0f}" if row and not np.isnan(row["mean_final_penalty"]) else "N/A"
            print(f"  {val:>14}", end="")
        print()

    print(f"\nWrote per-cell JSONs and grid_summary.json to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
