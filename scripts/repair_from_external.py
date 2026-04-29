"""Run the repair-level bandit on an externally-produced (hard-feasible) schedule.

Skips the week-level construction stage entirely: loads a bundled NurseScheduler
solution (Sol-*.json files under a ``Solution_H_*-WD_*`` directory), builds a
``Schedule`` from it, and hands it to the repair runner. Useful for isolating
whether the repair bandit adds value on top of a good start, rather than just
papering over a bad greedy init.

Example:
    PYTHONPATH=src python scripts/repair_from_external.py \\
        --dataset-root Dataset/testdatasets_json --dataset n021w4 \\
        --solution-dir Solution_H_0-WD_5-4-1-2 \\
        --repair-bandit linucb --repair-checkpoint runs/final_repair.npz \\
        --rounds 10000 --seed 0 \\
        --output runs/repair_from_ns_n021w4_R10000_s0.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bandit import get_bandit  # noqa: E402
from bandit.linucb import LinUCB  # noqa: E402
from instance_loader import load_instance_from_bundle  # noqa: E402
from repair_level.linucb_selector import LinUCBRepairSelector  # noqa: E402
from repair_level.repairs import build_all_strategies  # noqa: E402
from repair_level.runner import run_repairs  # noqa: E402
from schedule.penalty import compute_penalty  # noqa: E402
from schedule.representation import Schedule  # noqa: E402


SOL_DIR_RE = re.compile(r"Solution_H_(\d+)-WD_([\d\-]+)$")
SOL_FILE_RE = re.compile(r"Sol-.+-(\d+)-(\d+)\.json$")


def parse_solution_dir(name: str) -> tuple[int, list[int]]:
    """Return (history_idx, week_indices) parsed from a Solution_H_*-WD_* name."""
    m = SOL_DIR_RE.match(name)
    if not m:
        raise SystemExit(f"Unrecognized solution dir name: {name!r}")
    history_idx = int(m.group(1))
    weeks = [int(x) for x in m.group(2).split("-")]
    return history_idx, weeks


def load_external_schedule(
    scenario: dict,
    solution_dir: Path,
) -> tuple[Schedule, list[str]]:
    """Load Sol-*.json files (sorted by position index) and build a Schedule."""
    files = []
    for p in solution_dir.iterdir():
        m = SOL_FILE_RE.match(p.name)
        if m:
            files.append((int(m.group(2)), p))
    if not files:
        raise SystemExit(f"No Sol-*.json files found in {solution_dir}")
    files.sort(key=lambda x: x[0])
    solutions = [json.loads(p.read_text()) for _, p in files]
    schedule = Schedule.from_solutions(scenario, solutions)
    return schedule, [p.name for _, p in files]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default="Dataset/testdatasets_json")
    p.add_argument("--dataset", required=True)
    p.add_argument(
        "--solution-dir",
        required=True,
        help="Directory name under the dataset, e.g. 'Solution_H_0-WD_5-4-1-2'.",
    )
    p.add_argument("--repair-bandit", default="linucb")
    p.add_argument("--repair-checkpoint", default=None)
    p.add_argument("--repair-reward-scale", type=float, default=50.0)
    p.add_argument("--rounds", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    ds_dir = Path(args.dataset_root) / args.dataset
    sol_dir = ds_dir / args.solution_dir
    if not sol_dir.is_dir():
        raise SystemExit(f"Solution dir not found: {sol_dir}")

    history_idx, weeks = parse_solution_dir(args.solution_dir)

    scenario_file = ds_dir / f"Sc-{args.dataset}.json"
    history_files = sorted(ds_dir.glob(f"H0-{args.dataset}-*.json"))
    week_files = sorted(ds_dir.glob(f"WD-{args.dataset}-*.json"))

    t0 = time.time()
    inst = load_instance_from_bundle(
        dataset_name=args.dataset,
        scenario_file=scenario_file,
        history_files=history_files,
        week_files=week_files,
        history_idx=history_idx,
        week_indices=weeks,
    )
    print(
        f"[stage 1] loaded {args.dataset} H={history_idx} W={weeks} "
        f"in {time.time()-t0:.2f}s"
    )

    t0 = time.time()
    schedule, sol_files = load_external_schedule(inst.scenario, sol_dir)
    init_pen = compute_penalty(schedule, inst.scenario, inst.weeks, inst.initial_history)
    print(
        f"[stage 2] loaded external schedule from {len(sol_files)} files "
        f"(initial P={init_pen.total} hard={init_pen.hard}) "
        f"({time.time()-t0:.2f}s)"
    )

    t0 = time.time()
    strategies = build_all_strategies(
        inst.scenario, inst.initial_history, inst.weeks, seed=args.seed
    )
    if args.repair_bandit == "linucb":
        if not args.repair_checkpoint:
            raise SystemExit("--repair-bandit linucb requires --repair-checkpoint.")
        lin = LinUCB.load(args.repair_checkpoint)
        if lin.num_arms != len(strategies):
            raise SystemExit(
                f"Checkpoint has {lin.num_arms} arms; expected {len(strategies)}."
            )
        selector = LinUCBRepairSelector(
            strategy_names=[s.name for s in strategies],
            alpha=lin.alpha,
            reward_scale=args.repair_reward_scale,
            seed=args.seed,
            linucb=lin,
        )
    else:
        selector = get_bandit(
            args.repair_bandit,
            strategy_names=[s.name for s in strategies],
            seed=args.seed,
        )

    result = run_repairs(
        scenario=inst.scenario,
        history=inst.initial_history,
        week_data_list=inst.weeks,
        strategies=strategies,
        schedule=schedule,
        selector=selector,
        num_rounds=args.rounds,
        seed=args.seed,
    )
    print(
        f"[stage 3] repair-bandit={args.repair_bandit} rounds={args.rounds} "
        f"P: {init_pen.total} -> {result.final_penalty} "
        f"(Δ={init_pen.total - result.final_penalty}) "
        f"hard={result.hard_violations} ({time.time()-t0:.2f}s)"
    )

    final_pen = compute_penalty(
        result.schedule, inst.scenario, inst.weeks, inst.initial_history
    )
    print(
        f"[final] P={final_pen.total}  "
        f"S1={final_pen.s1_optimal_coverage}  "
        f"S2={final_pen.s2_consecutive}  "
        f"S3={final_pen.s3_days_off}  "
        f"S4={final_pen.s4_preferences}  "
        f"S5={final_pen.s5_complete_weekends}  "
        f"S6={final_pen.s6_total_assignments}  "
        f"S7={final_pen.s7_working_weekends}  "
        f"hard={final_pen.hard}"
    )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps({
            "dataset": args.dataset,
            "solution_dir": args.solution_dir,
            "history_idx": history_idx,
            "weeks": weeks,
            "source_files": sol_files,
            "repair_bandit": args.repair_bandit,
            "repair_checkpoint": args.repair_checkpoint,
            "rounds": args.rounds,
            "seed": args.seed,
            "initial_penalty": init_pen.total,
            "initial_hard": init_pen.hard,
            "initial_soft": {
                "S1": init_pen.s1_optimal_coverage,
                "S2": init_pen.s2_consecutive,
                "S3": init_pen.s3_days_off,
                "S4": init_pen.s4_preferences,
                "S5": init_pen.s5_complete_weekends,
                "S6": init_pen.s6_total_assignments,
                "S7": init_pen.s7_working_weekends,
            },
            "final_penalty": final_pen.total,
            "final_hard": final_pen.hard,
            "final_soft": {
                "S1": final_pen.s1_optimal_coverage,
                "S2": final_pen.s2_consecutive,
                "S3": final_pen.s3_days_off,
                "S4": final_pen.s4_preferences,
                "S5": final_pen.s5_complete_weekends,
                "S6": final_pen.s6_total_assignments,
                "S7": final_pen.s7_working_weekends,
            },
            "penalty_trajectory": list(result.penalty_trajectory),
            "strategy_counts": dict(result.strategy_counts),
            "total_attempted": result.total_attempted,
            "total_succeeded": result.total_succeeded,
        }, indent=2))
        print(f"wrote artifact: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
