"""Exp 02: 5x5 bandit grid (week × repair) at fixed round budget.

Runs every (week_bandit, repair_bandit) ∈ {linucb, uniform, ucb1, eps, thompson}^2
on a held-out eval set, ``--seeds`` seeds, ``--rounds`` repair rounds.
LinUCB cells use the base checkpoints from experiments/_shared/checkpoints/.

Writes:
  experiments/exp02_bandit_grid/results/grid.json
  experiments/exp02_bandit_grid/results/per_run.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from _shared.eval_pipeline import (  # noqa: E402
    get_eval_instances,
    run_pipeline_one,
    write_json,
)


BANDITS = ["linucb", "uniform", "ucb1", "epsilon_greedy", "thompson"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=500)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--eval-split", default="dev")
    p.add_argument("--n-eval", type=int, default=8)
    p.add_argument("--week-checkpoint",
                   default=str(ROOT / "experiments/_shared/checkpoints/base_week.npz"))
    p.add_argument("--repair-checkpoint",
                   default=str(ROOT / "experiments/_shared/checkpoints/base_repair.npz"))
    p.add_argument("--out-dir",
                   default=str(ROOT / "experiments/exp02_bandit_grid/results"))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    instances = get_eval_instances(args.eval_split, args.n_eval, seed=0)
    print(f"[exp02] eval instances: {[i.dataset_name for i in instances]}", flush=True)

    grid: list[dict] = []
    per_run_path = out_dir / "per_run.jsonl"
    per_run_f = per_run_path.open("w")
    t_total = time.perf_counter()

    total_cells = len(BANDITS) * len(BANDITS)
    cell_i = 0
    for wb in BANDITS:
        for rb in BANDITS:
            cell_i += 1
            cell_t0 = time.perf_counter()
            cell = {
                "week_bandit": wb,
                "repair_bandit": rb,
                "rounds": args.rounds,
                "seeds": args.seeds,
                "n_eval": len(instances),
                "runs": [],
            }
            for seed in args.seeds:
                for inst in instances:
                    try:
                        out = run_pipeline_one(
                            inst,
                            week_bandit=wb,
                            week_checkpoint=args.week_checkpoint if wb == "linucb" else None,
                            repair_bandit=rb,
                            repair_checkpoint=args.repair_checkpoint if rb == "linucb" else None,
                            rounds=args.rounds,
                            seed=seed,
                        )
                        out["week_bandit"] = wb
                        out["repair_bandit"] = rb
                        out["seed"] = seed
                        cell["runs"].append({
                            "seed": seed,
                            "dataset": inst.dataset_name,
                            "post_week_penalty": out["post_week_penalty"],
                            "final_penalty": out["final_penalty"],
                            "delta_repair": out["delta_repair"],
                        })
                        per_run_f.write(json.dumps(out) + "\n")
                        per_run_f.flush()
                    except Exception as e:
                        print(f"  [skip {wb}/{rb} seed={seed} {inst.dataset_name}]: {e}",
                              flush=True)
            grid.append(cell)
            dt = time.perf_counter() - cell_t0
            print(f"[exp02] cell {cell_i}/{total_cells} ({wb}/{rb}) "
                  f"{len(cell['runs'])} runs in {dt:.1f}s", flush=True)

    per_run_f.close()
    write_json(out_dir / "grid.json", {
        "config": vars(args),
        "grid": grid,
        "wall_clock_s": time.perf_counter() - t_total,
    })
    print(f"[exp02] done in {time.perf_counter() - t_total:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
