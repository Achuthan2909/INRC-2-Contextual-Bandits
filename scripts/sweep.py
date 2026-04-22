"""Hyperparameter sweep for the week-level or repair-level LinUCB.

Trains one LinUCB per (alpha, reward_scale, warm_start) combo on the chosen
``--train-split``, then scores each checkpoint on the ``--eval-split`` by
running a short evaluation pass that replays the arms greedily (no exploration,
no bandit update) and reports the mean per-instance penalty reduction.

Writes a CSV of ``(alpha, reward_scale, warm_start, train_s, mean_delta,
mean_final_penalty)`` and a JSON copy with full per-config stats.

Example (week-level):
    PYTHONPATH=src python scripts/sweep.py \\
        --level week \\
        --train-split train --eval-split dev \\
        --alphas 0.5 1.0 2.0 --reward-scales 500 1000 2000 \\
        --warm-starts 0 8 \\
        --max-train-instances 30 --max-eval-instances 10 \\
        --out-dir runs/sweep_week

Example (repair-level):
    PYTHONPATH=src python scripts/sweep.py \\
        --level repair \\
        --train-split train --eval-split dev \\
        --alphas 0.5 1.0 --reward-scales 25 50 100 \\
        --warm-starts 0 36 \\
        --num-rounds 300 \\
        --max-train-instances 20 --max-eval-instances 5 \\
        --out-dir runs/sweep_repair
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

from bandit.linucb import LinUCB  # noqa: E402
from data.splits import SPLITS, split_instances  # noqa: E402
from repair_level.init import generate_initial_schedule  # noqa: E402
from repair_level.linucb_selector import LinUCBRepairSelector  # noqa: E402
from repair_level.repairs import build_all_strategies  # noqa: E402
from repair_level.runner import run_repairs  # noqa: E402
from repair_level.train import train_linucb_repair  # noqa: E402
from schedule.penalty import compute_penalty  # noqa: E402
from week_level.arms import (  # noqa: E402
    CoverageFirstArm,
    FatigueAwareArm,
    PreferenceRespectingArm,
    WeekendBalancingArm,
)
from week_level.runner import run_week_level  # noqa: E402
from week_level.train import train_linucb  # noqa: E402


WEEK_ARMS_FACTORY = lambda: [  # noqa: E731
    CoverageFirstArm(),
    FatigueAwareArm(),
    WeekendBalancingArm(),
    PreferenceRespectingArm(),
]


def _eval_week(
    checkpoint: Path,
    eval_split: str,
    max_instances: int,
    seed: int,
    reward_scale: float,
) -> dict:
    lin = LinUCB.load(str(checkpoint))
    arms = WEEK_ARMS_FACTORY()
    stream = split_instances(
        eval_split, seed=seed, shuffle=True, week_combos_per_scenario=1,
    )
    deltas, finals = [], []
    n = 0
    for inst in stream:
        if n >= max_instances:
            break
        try:
            out = run_week_level(
                scenario=inst.scenario,
                initial_history=inst.initial_history,
                week_data_list=inst.weeks,
                arms=arms,
                bandit=lin,
                reward_scale=reward_scale,
            )
            final_p = out["total_penalty"]
            # Baseline = empty schedule penalty before any arm fires.
            rewards = out.get("linucb_reward_trajectory", [])
            # Sum of per-week penalty reduction ≈ total Δ from empty schedule.
            delta = float(sum(rewards))
            deltas.append(delta)
            finals.append(float(final_p))
            n += 1
        except Exception as e:
            print(f"  [eval skip {inst.dataset_name}] {e}", file=sys.stderr)
            continue
    return {
        "n_eval": n,
        "mean_delta": float(np.mean(deltas)) if deltas else 0.0,
        "mean_final_penalty": float(np.mean(finals)) if finals else float("nan"),
        "per_instance_delta": deltas,
        "per_instance_final": finals,
    }


def _eval_repair(
    checkpoint: Path,
    eval_split: str,
    max_instances: int,
    seed: int,
    reward_scale: float,
    num_rounds: int,
) -> dict:
    lin = LinUCB.load(str(checkpoint))
    stream = split_instances(
        eval_split, seed=seed, shuffle=True, week_combos_per_scenario=1,
    )
    deltas, finals = [], []
    n = 0
    for inst in stream:
        if n >= max_instances:
            break
        try:
            schedule = generate_initial_schedule(
                inst.scenario, inst.initial_history, inst.weeks,
            )
            init_p = compute_penalty(
                schedule, inst.scenario, inst.weeks, inst.initial_history,
            ).total
            strategies = build_all_strategies(
                inst.scenario, inst.initial_history, inst.weeks, seed=seed,
            )
            selector = LinUCBRepairSelector(
                strategy_names=[s.name for s in strategies],
                alpha=lin.alpha,
                reward_scale=reward_scale,
                seed=seed,
                linucb=lin,
            )
            result = run_repairs(
                scenario=inst.scenario,
                history=inst.initial_history,
                week_data_list=inst.weeks,
                strategies=strategies,
                schedule=schedule,
                selector=selector,
                num_rounds=num_rounds,
                seed=seed,
            )
            delta = float(init_p - result.final_penalty)
            deltas.append(delta)
            finals.append(float(result.final_penalty))
            n += 1
        except Exception as e:
            print(f"  [eval skip {inst.dataset_name}] {e}", file=sys.stderr)
            continue
    return {
        "n_eval": n,
        "mean_delta": float(np.mean(deltas)) if deltas else 0.0,
        "mean_final_penalty": float(np.mean(finals)) if finals else float("nan"),
        "per_instance_delta": deltas,
        "per_instance_final": finals,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Hyperparameter sweep for LinUCB.")
    p.add_argument("--level", choices=["week", "repair"], required=True)
    p.add_argument("--train-split", choices=SPLITS, default="train")
    p.add_argument("--eval-split", choices=SPLITS, default="dev")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    p.add_argument("--reward-scales", type=float, nargs="+", default=[1000.0])
    p.add_argument("--warm-starts", type=int, nargs="+", default=[0])
    p.add_argument("--max-train-instances", type=int, default=30)
    p.add_argument("--max-eval-instances", type=int, default=10)
    p.add_argument("--week-combos-per-scenario", type=int, default=5)
    p.add_argument("--num-rounds", type=int, default=300,
                   help="Repair-level only: rounds per instance.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="runs/sweep")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    combos = list(product(args.alphas, args.reward_scales, args.warm_starts))
    print(f"[sweep] level={args.level} combos={len(combos)} "
          f"train={args.train_split} eval={args.eval_split}")

    rows: list[dict] = []
    for i, (alpha, rs, ws) in enumerate(combos, 1):
        tag = f"a{alpha}_rs{rs}_ws{ws}".replace(".", "p")
        ckpt = ckpt_dir / f"{args.level}_{tag}.npz"
        print(f"\n[sweep {i}/{len(combos)}] alpha={alpha} reward_scale={rs} warm={ws}")

        t0 = time.perf_counter()
        if args.level == "week":
            train_linucb(
                WEEK_ARMS_FACTORY(),
                split=args.train_split,
                alpha=alpha,
                max_instances=args.max_train_instances,
                week_combos_per_scenario=args.week_combos_per_scenario,
                seed=args.seed,
                reward_scale=rs,
                checkpoint_path=str(ckpt),
                log_every=0,
                warm_start_rounds=ws,
            )
        else:
            train_linucb_repair(
                split=args.train_split,
                alpha=alpha,
                max_instances=args.max_train_instances,
                week_combos_per_scenario=args.week_combos_per_scenario,
                seed=args.seed,
                reward_scale=rs,
                num_rounds=args.num_rounds,
                checkpoint_path=str(ckpt),
                log_every=0,
                warm_start_rounds=ws,
            )
        train_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        if args.level == "week":
            ev = _eval_week(
                ckpt, args.eval_split, args.max_eval_instances, args.seed, rs,
            )
        else:
            ev = _eval_repair(
                ckpt, args.eval_split, args.max_eval_instances, args.seed, rs,
                args.num_rounds,
            )
        eval_s = time.perf_counter() - t1

        row = {
            "alpha": alpha,
            "reward_scale": rs,
            "warm_start": ws,
            "train_s": round(train_s, 2),
            "eval_s": round(eval_s, 2),
            "n_eval": ev["n_eval"],
            "mean_delta": round(ev["mean_delta"], 2),
            "mean_final_penalty": round(ev["mean_final_penalty"], 2),
            "checkpoint": str(ckpt),
        }
        rows.append({**row, "per_instance_delta": ev["per_instance_delta"],
                     "per_instance_final": ev["per_instance_final"]})
        print(f"  train={train_s:.1f}s eval={eval_s:.1f}s "
              f"meanΔ={row['mean_delta']} meanFinal={row['mean_final_penalty']}")

    # Write CSV (compact) + JSON (full)
    csv_path = out_dir / "results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "alpha", "reward_scale", "warm_start",
                "train_s", "eval_s", "n_eval",
                "mean_delta", "mean_final_penalty", "checkpoint",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})
    (out_dir / "results.json").write_text(json.dumps(rows, indent=2))

    # Rank by mean_delta (higher is better).
    ranked = sorted(rows, key=lambda r: -r["mean_delta"])
    print("\n=== Ranking (by mean Δpenalty, higher is better) ===")
    for r in ranked:
        print(
            f"  alpha={r['alpha']} rs={r['reward_scale']} ws={r['warm_start']} "
            f"→ meanΔ={r['mean_delta']}  meanFinal={r['mean_final_penalty']}  "
            f"(n={r['n_eval']})"
        )
    print(f"\nwrote {csv_path} and {out_dir/'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
