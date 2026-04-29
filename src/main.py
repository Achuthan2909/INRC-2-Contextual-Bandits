"""Argparse-based driver for repair-level experiments.

Loads an instance, builds an initial schedule (greedy), then optionally runs the
repair bandit loop (``--no-repair`` skips iterations; initial penalty only).

Example:
    python src/main.py --dataset n030w4 --weeks 0 1 2 3 --bandit ucb1 --rounds 500
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bandit import available as available_bandits, get_bandit  # noqa: E402
from evaluate import format_validator_report  # noqa: E402
from instance_loader import load_instance  # noqa: E402
from repair_level.init import generate_initial_schedule  # noqa: E402
from repair_level.repairs import build_all_strategies  # noqa: E402
from repair_level.runner import RunResult, run_repairs  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="inrc2",
        description="Run a repair-level bandit experiment on an INRC-II instance.",
    )
    p.add_argument("--dataset-root", default="Dataset/datasets_json")
    p.add_argument("--dataset", default="n030w4", help="Instance family, e.g. n030w4.")
    p.add_argument("--history-idx", type=int, default=0)
    p.add_argument("--weeks", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument(
        "--bandit",
        default="ucb1",
        choices=available_bandits(),
        help="Repair-level arm selector.",
    )
    p.add_argument("--rounds", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--final-display",
        choices=["none", "summary", "validator"],
        default="summary",
    )
    p.add_argument("--output-dir", default="runs", help="Where to write the JSON artifact.")
    p.add_argument("--quiet", action="store_true", help="Suppress INFO logs.")
    p.add_argument(
        "--no-repair",
        action="store_true",
        help="Only build greedy initial schedule and score it (0 repair rounds).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    log = logging.getLogger("inrc2")

    instance = load_instance(
        dataset_root=args.dataset_root,
        dataset_name=args.dataset,
        history_idx=args.history_idx,
        week_indices=args.weeks,
    )
    scenario = instance.scenario
    history = instance.initial_history
    week_data_list = instance.weeks

    strategies = build_all_strategies(scenario, history, week_data_list, seed=args.seed)
    if not strategies:
        log.error("No repair strategies could be auto-constructed.")
        return 1

    selector = get_bandit(
        args.bandit,
        strategy_names=[s.name for s in strategies],
        seed=args.seed,
    )

    schedule = generate_initial_schedule(scenario, history, week_data_list)
    repair_rounds = 0 if args.no_repair else args.rounds

    t0 = time.perf_counter()
    out: RunResult = run_repairs(
        scenario=scenario,
        history=history,
        week_data_list=week_data_list,
        strategies=strategies,
        schedule=schedule,
        selector=selector,
        num_rounds=repair_rounds,
        seed=args.seed,
    )
    runtime_s = time.perf_counter() - t0

    log.info(
        "Final penalty: %d (Δ=%+d) in %.1fs",
        out.final_penalty,
        out.initial_penalty - out.final_penalty,
        runtime_s,
    )

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    artifact_path = out_path / f"{instance.dataset_name}_{args.bandit}_seed{args.seed}_{ts}.json"

    artifact = {
        "meta": {
            "dataset": instance.dataset_name,
            "dataset_root": args.dataset_root,
            "history_idx": args.history_idx,
            "weeks": args.weeks,
            "init": "greedy",
            "no_repair": args.no_repair,
            "bandit": args.bandit,
            "rounds_requested": repair_rounds,
            "seed": args.seed,
            "runtime_s": runtime_s,
        },
        "result": out.to_dict(),
    }
    stats_fn = getattr(selector, "stats", None)
    if callable(stats_fn):
        artifact["bandit_stats"] = stats_fn()
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True))
    log.info("Wrote run artifact: %s", artifact_path)

    if args.final_display == "validator":
        from schedule.penalty import compute_penalty

        final = compute_penalty(out.schedule, scenario, week_data_list, history)
        print(format_validator_report(final))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
