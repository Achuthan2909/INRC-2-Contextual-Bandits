"""End-to-end pipeline driver: week-level bandit + repair-level bandit.

Stage 1 — load one INRCInstance.
Stage 2 — week-level bandit (LinUCB with trained checkpoint, or baseline) builds
          a multi-week schedule by picking a construction arm per week.
Stage 3 — repair-level bandit (LinUCB with trained checkpoint, or baseline) runs
          ``--rounds`` repair iterations on the shared schedule.
Reports initial / post-week / final penalty + hard violations + runtime.

Example (trained LinUCB at both levels):
    PYTHONPATH=src python scripts/run_pipeline.py \
        --dataset n030w4 --weeks 0 1 2 3 \
        --week-bandit linucb --week-checkpoint runs/linucb_week_level.npz \
        --repair-bandit linucb --repair-checkpoint runs/linucb_repair_level.npz \
        --rounds 500

Baseline (uniform week, UCB1 repair):
    PYTHONPATH=src python scripts/run_pipeline.py \
        --dataset n030w4 --weeks 0 1 2 3 \
        --week-bandit uniform --repair-bandit ucb1 --rounds 500
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bandit import available as available_bandits, get_bandit  # noqa: E402
from bandit.linucb import LinUCB  # noqa: E402
from instance_loader import load_instance  # noqa: E402
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


WEEK_ARMS = [
    CoverageFirstArm(),
    FatigueAwareArm(),
    WeekendBalancingArm(),
    PreferenceRespectingArm(),
]


def _run_week_stage(
    args: argparse.Namespace,
    scenario: dict,
    history: dict,
    weeks: list[dict],
) -> tuple[dict, list[str]]:
    """Stage 2 — pick a construction arm per week."""
    if args.week_bandit == "linucb":
        if not args.week_checkpoint:
            raise SystemExit(
                "--week-bandit linucb requires --week-checkpoint "
                "(or pass --week-bandit uniform for an untrained run)."
            )
        lin = LinUCB.load(args.week_checkpoint)
        if lin.num_arms != len(WEEK_ARMS):
            raise SystemExit(
                f"Week checkpoint has {lin.num_arms} arms; expected {len(WEEK_ARMS)}."
            )
        out = run_week_level(
            scenario=scenario,
            initial_history=history,
            week_data_list=weeks,
            arms=WEEK_ARMS,
            bandit=lin,
            reward_scale=args.week_reward_scale,
        )
    else:
        selector = get_bandit(
            args.week_bandit,
            strategy_names=[a.name for a in WEEK_ARMS],
            seed=args.seed,
        )
        out = run_week_level_baseline(
            scenario=scenario,
            initial_history=history,
            week_data_list=weeks,
            arms=WEEK_ARMS,
            selector=selector,
            reward_scale=args.week_reward_scale,
        )
    return out, out["arms_picked"]


def _run_repair_stage(
    args: argparse.Namespace,
    scenario: dict,
    history: dict,
    weeks: list[dict],
    schedule,
) -> dict:
    """Stage 3 — repair-level bandit for ``--rounds`` iterations."""
    strategies = build_all_strategies(scenario, history, weeks, seed=args.seed)

    if args.repair_bandit == "linucb":
        if not args.repair_checkpoint:
            raise SystemExit("--repair-bandit linucb requires --repair-checkpoint.")
        lin = LinUCB.load(args.repair_checkpoint)
        if lin.num_arms != len(strategies):
            raise SystemExit(
                f"Repair checkpoint has {lin.num_arms} arms; expected {len(strategies)}."
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
        scenario=scenario,
        history=history,
        week_data_list=weeks,
        strategies=strategies,
        schedule=schedule,
        selector=selector,
        num_rounds=args.rounds,
        seed=args.seed,
    )
    return {
        "final_penalty": result.final_penalty,
        "rounds_run": result.rounds_run,
        "total_attempted": result.total_attempted,
        "total_succeeded": result.total_succeeded,
        "strategy_counts": dict(result.strategy_counts),
        "hard_violations": dict(result.hard_violations),
        "penalty_trajectory": list(result.penalty_trajectory),
        "schedule": result.schedule,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="End-to-end week + repair pipeline.")
    p.add_argument("--dataset-root", default="Dataset/datasets_json")
    p.add_argument("--dataset", default="n030w4")
    p.add_argument("--history-idx", type=int, default=0)
    p.add_argument("--weeks", type=int, nargs="+", default=[0, 1, 2, 3])

    p.add_argument("--week-bandit", default="linucb",
                   choices=available_bandits() + ["linucb"])
    p.add_argument("--week-checkpoint", default=None,
                   help="Trained LinUCB .npz (required if --week-bandit linucb).")
    p.add_argument("--week-reward-scale", type=float, default=1000.0)

    p.add_argument("--repair-bandit", default="linucb",
                   choices=available_bandits() + ["linucb"])
    p.add_argument("--repair-checkpoint", default=None,
                   help="Trained LinUCB .npz (required if --repair-bandit linucb).")
    p.add_argument("--repair-reward-scale", type=float, default=50.0)

    p.add_argument("--rounds", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=None,
                   help="Where to write the JSON artifact. Default: runs/pipeline_<ts>.json")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    def log(msg: str) -> None:
        if not args.quiet:
            print(msg)

    # --- Stage 1: load instance
    t_load0 = time.perf_counter()
    instance = load_instance(
        dataset_root=args.dataset_root,
        dataset_name=args.dataset,
        history_idx=args.history_idx,
        week_indices=args.weeks,
    )
    t_load = time.perf_counter() - t_load0
    log(f"[stage 1] loaded {args.dataset} in {t_load:.2f}s")

    # --- Stage 2: week-level bandit builds the schedule
    t_week0 = time.perf_counter()
    week_out, arms_picked = _run_week_stage(
        args, instance.scenario, instance.initial_history, instance.weeks,
    )
    t_week = time.perf_counter() - t_week0
    schedule = week_out["schedule"]
    post_week_penalty = week_out["total_penalty"]
    log(
        f"[stage 2] week-bandit={args.week_bandit} "
        f"arms_picked={arms_picked} post-week P={post_week_penalty} "
        f"hard={week_out['hard_violations']} ({t_week:.2f}s)"
    )

    # --- Stage 3: repair-level bandit
    t_rep0 = time.perf_counter()
    rep = _run_repair_stage(
        args, instance.scenario, instance.initial_history, instance.weeks, schedule,
    )
    t_rep = time.perf_counter() - t_rep0
    log(
        f"[stage 3] repair-bandit={args.repair_bandit} "
        f"rounds={rep['rounds_run']} "
        f"P: {rep['penalty_trajectory'][0]} -> {rep['final_penalty']} "
        f"(Δ={rep['penalty_trajectory'][0] - rep['final_penalty']:+d}) "
        f"hard={rep['hard_violations']} ({t_rep:.2f}s)"
    )

    # --- Final report
    final = compute_penalty(
        rep["schedule"], instance.scenario, instance.weeks, instance.initial_history,
    )
    log(
        f"[final] P={final.total}  S1={final.s1_optimal_coverage}  "
        f"S2={final.s2_consecutive}  S3={final.s3_days_off}  "
        f"S4={final.s4_preferences}  S5={final.s5_complete_weekends}  "
        f"S6={final.s6_total_assignments}  S7={final.s7_working_weekends}  "
        f"hard={final.hard}"
    )

    artifact = {
        "meta": {
            "dataset": args.dataset,
            "dataset_root": args.dataset_root,
            "history_idx": args.history_idx,
            "weeks": args.weeks,
            "week_bandit": args.week_bandit,
            "week_checkpoint": args.week_checkpoint,
            "repair_bandit": args.repair_bandit,
            "repair_checkpoint": args.repair_checkpoint,
            "rounds_requested": args.rounds,
            "seed": args.seed,
            "t_load_s": t_load,
            "t_week_s": t_week,
            "t_repair_s": t_rep,
        },
        "result": {
            "arms_picked": arms_picked,
            "post_week_penalty": post_week_penalty,
            "final_penalty": final.total,
            "delta_total": post_week_penalty - final.total,
            "repair_rounds_run": rep["rounds_run"],
            "repair_strategy_counts": rep["strategy_counts"],
            "hard_violations": dict(final.hard),
            "soft_breakdown": {
                "s1_optimal_coverage": final.s1_optimal_coverage,
                "s2_consecutive": final.s2_consecutive,
                "s3_days_off": final.s3_days_off,
                "s4_preferences": final.s4_preferences,
                "s5_complete_weekends": final.s5_complete_weekends,
                "s6_total_assignments": final.s6_total_assignments,
                "s7_working_weekends": final.s7_working_weekends,
            },
            "penalty_trajectory": rep["penalty_trajectory"],
        },
    }
    out_path = args.output
    if out_path is None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        out_path = f"runs/pipeline_{args.dataset}_{args.week_bandit}_{args.repair_bandit}_{ts}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(artifact, indent=2, sort_keys=True))
    log(f"wrote artifact: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
