"""Experiment 7 — Cross-split generalization.

Evaluates the best LinUCB checkpoint (week + repair) on dev, val, and test splits.
Uses split_instances() so it handles all three directory layouts automatically.

Example:
    PYTHONPATH=src python scripts/run_exp7.py \
        --week-checkpoint runs/final_week.npz \
        --repair-checkpoint runs/final_repair.npz \
        --rounds 500 \
        --max-instances 10 \
        --out-dir runs/exp7
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

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


WEEK_ARMS_FACTORY = lambda: [  # noqa: E731
    CoverageFirstArm(),
    FatigueAwareArm(),
    WeekendBalancingArm(),
    PreferenceRespectingArm(),
]


def run_instance(inst, week_lin, repair_lin, rounds, seed, week_reward_scale, repair_reward_scale):
    arms = WEEK_ARMS_FACTORY()

    # Stage 2: week-level
    week_out = run_week_level(
        scenario=inst.scenario,
        initial_history=inst.initial_history,
        week_data_list=inst.weeks,
        arms=arms,
        bandit=week_lin,
        reward_scale=week_reward_scale,
    )
    schedule = week_out["schedule"]
    post_week_penalty = week_out["total_penalty"]

    # Stage 3: repair-level
    strategies = build_all_strategies(inst.scenario, inst.initial_history, inst.weeks, seed=seed)
    selector = LinUCBRepairSelector(
        strategy_names=[s.name for s in strategies],
        alpha=repair_lin.alpha,
        reward_scale=repair_reward_scale,
        seed=seed,
        linucb=repair_lin,
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

    return {
        "dataset": inst.dataset_name,
        "post_week_penalty": post_week_penalty,
        "final_penalty": final.total,
        "delta": post_week_penalty - final.total,
        "arms_picked": week_out["arms_picked"],
        "repair_strategy_counts": dict(result.strategy_counts),
        "soft_breakdown": {
            "s1_optimal_coverage": final.s1_optimal_coverage,
            "s2_consecutive": final.s2_consecutive,
            "s3_days_off": final.s3_days_off,
            "s4_preferences": final.s4_preferences,
            "s5_complete_weekends": final.s5_complete_weekends,
            "s6_total_assignments": final.s6_total_assignments,
            "s7_working_weekends": final.s7_working_weekends,
        },
        "hard_violations": dict(final.hard),
    }


def eval_split(split, week_lin, repair_lin, rounds, max_instances, seed,
               week_reward_scale, repair_reward_scale):
    stream = split_instances(split, seed=seed, shuffle=True, week_combos_per_scenario=1)
    results = []
    n = 0
    for inst in stream:
        if n >= max_instances:
            break
        try:
            t0 = time.perf_counter()
            r = run_instance(inst, week_lin, repair_lin, rounds, seed,
                             week_reward_scale, repair_reward_scale)
            elapsed = time.perf_counter() - t0
            print(f"  [{split}] {inst.dataset_name} → post-week={r['post_week_penalty']} "
                  f"final={r['final_penalty']} Δ={r['delta']:+d} ({elapsed:.1f}s)")
            results.append(r)
            n += 1
        except Exception as e:
            print(f"  [{split}] SKIP {inst.dataset_name}: {e}", file=sys.stderr)
    return results


def main() -> int:
    p = argparse.ArgumentParser(description="Exp 7: cross-split generalization.")
    p.add_argument("--week-checkpoint", default="runs/final_week.npz")
    p.add_argument("--repair-checkpoint", default="runs/final_repair.npz")
    p.add_argument("--splits", nargs="+", default=["dev", "val", "test"],
                   choices=["train", "dev", "val", "test"])
    p.add_argument("--rounds", type=int, default=500)
    p.add_argument("--max-instances", type=int, default=10)
    p.add_argument("--week-reward-scale", type=float, default=1000.0)
    p.add_argument("--repair-reward-scale", type=float, default=50.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="runs/exp7")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    week_lin = LinUCB.load(args.week_checkpoint)
    repair_lin = LinUCB.load(args.repair_checkpoint)
    print(f"Loaded week checkpoint: {args.week_checkpoint} (alpha={week_lin.alpha})")
    print(f"Loaded repair checkpoint: {args.repair_checkpoint} (alpha={repair_lin.alpha})")

    all_results = {}
    summary_rows = []

    for split in args.splits:
        print(f"\n=== Split: {split} ===")
        results = eval_split(
            split, week_lin, repair_lin,
            rounds=args.rounds,
            max_instances=args.max_instances,
            seed=args.seed,
            week_reward_scale=args.week_reward_scale,
            repair_reward_scale=args.repair_reward_scale,
        )
        all_results[split] = results

        # Save per-split JSON
        (out_dir / f"{split}.json").write_text(json.dumps(results, indent=2))

        if results:
            deltas = [r["delta"] for r in results]
            finals = [r["final_penalty"] for r in results]
            mean_d = float(np.mean(deltas))
            mean_f = float(np.mean(finals))
            summary_rows.append({
                "split": split,
                "n": len(results),
                "mean_delta": round(mean_d, 2),
                "mean_final_penalty": round(mean_f, 2),
            })
            print(f"  → n={len(results)}  meanΔ={mean_d:.2f}  meanFinal={mean_f:.2f}")
        else:
            print(f"  → no results")

    # Save summary
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2))

    print("\n=== Exp 7 Summary ===")
    for row in summary_rows:
        print(f"  {row['split']:6s}  n={row['n']}  meanΔ={row['mean_delta']}  "
              f"meanFinal={row['mean_final_penalty']}")
    print(f"\nWrote results to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
