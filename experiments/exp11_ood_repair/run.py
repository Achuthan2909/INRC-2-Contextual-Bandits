"""Exp 11: OOD repair experiment.

Tests whether the repair-level LinUCB *helps* or *hurts* when started from
near-optimal schedules (NurseScheduler bundled solutions in testdatasets_json)
versus the in-distribution greedy initializer.

Conditions per eval instance × seed:
  start ∈ {greedy, mid (greedy+200 rounds), bandp (Solution_*)}
  repair_bandit ∈ {linucb, uniform}
  rounds eval ∈ {0, 50, 100, 250, 500, 1000}

Outputs:
  results/curves.jsonl — per (start, bandit, rounds, seed, instance) row
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from _shared.eval_pipeline import run_repair_only, write_json  # noqa: E402

from instance_loader import load_instance_from_bundle  # noqa: E402
from repair_level.init import generate_initial_schedule  # noqa: E402
from repair_level.linucb_selector import LinUCBRepairSelector  # noqa: E402
from repair_level.repairs import build_all_strategies  # noqa: E402
from repair_level.runner import run_repairs  # noqa: E402
from bandit.linucb import LinUCB  # noqa: E402
from schedule.penalty import compute_penalty  # noqa: E402
from schedule.representation import Schedule  # noqa: E402


SOL_DIR_RE = re.compile(r"Solution_H_(\d+)-WD_([\d\-]+)$")
SOL_FILE_RE = re.compile(r"Sol-.+-(\d+)-(\d+)\.json$")


def _enumerate_bundled(dataset_root: Path, scenarios: list[str] | None = None):
    """Yield (dataset_name, history_idx, week_indices, sol_dir)."""
    for ds_dir in sorted(dataset_root.iterdir()):
        if not ds_dir.is_dir():
            continue
        if scenarios and ds_dir.name not in scenarios:
            continue
        for sub in sorted(ds_dir.iterdir()):
            if not sub.is_dir():
                continue
            m = SOL_DIR_RE.match(sub.name)
            if not m:
                continue
            history_idx = int(m.group(1))
            weeks = [int(x) for x in m.group(2).split("-")]
            yield ds_dir.name, history_idx, weeks, sub


def _load_inst(dataset_root: Path, dataset: str, history_idx: int, weeks: list[int]):
    ds_dir = dataset_root / dataset
    scenario_file = ds_dir / f"Sc-{dataset}.json"
    history_files = sorted(ds_dir.glob(f"H0-{dataset}-*.json"))
    week_files = sorted(ds_dir.glob(f"WD-{dataset}-*.json"))
    return load_instance_from_bundle(
        dataset_name=dataset,
        scenario_file=scenario_file,
        history_files=history_files,
        week_files=week_files,
        history_idx=history_idx,
        week_indices=weeks,
    )


def _load_external_schedule(scenario, sol_dir: Path) -> Schedule:
    files = []
    for p in sol_dir.iterdir():
        m = SOL_FILE_RE.match(p.name)
        if m:
            files.append((int(m.group(2)), p))
    files.sort(key=lambda x: x[0])
    sols = [json.loads(p.read_text()) for _, p in files]
    return Schedule.from_solutions(scenario, sols)


def _fresh_repair_selector(repair_bandit: str, repair_ckpt: str, strategy_names, seed):
    if repair_bandit == "linucb":
        lin = LinUCB.load(repair_ckpt)
        return LinUCBRepairSelector(
            strategy_names=strategy_names, alpha=lin.alpha,
            reward_scale=50.0, seed=seed, linucb=lin,
        )
    from bandit import get_bandit
    return get_bandit(repair_bandit, strategy_names=strategy_names, seed=seed)


def _greedy_then_n_rounds(inst, repair_ckpt, n_rounds, seed):
    """Build a 'mid' starting schedule by running n_rounds of LinUCB repair."""
    schedule = generate_initial_schedule(inst.scenario, inst.initial_history, inst.weeks)
    if n_rounds <= 0:
        return schedule
    strategies = build_all_strategies(
        inst.scenario, inst.initial_history, inst.weeks, seed=seed,
    )
    sel = _fresh_repair_selector("linucb", repair_ckpt,
                                 [s.name for s in strategies], seed)
    res = run_repairs(
        scenario=inst.scenario, history=inst.initial_history,
        week_data_list=inst.weeks, strategies=strategies,
        schedule=schedule, selector=sel, num_rounds=n_rounds, seed=seed,
    )
    return res.schedule


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default=str(ROOT / "Dataset/testdatasets_json"))
    p.add_argument("--repair-checkpoint",
                   default=str(ROOT / "experiments/_shared/checkpoints/base_repair.npz"))
    p.add_argument("--rounds-list", type=int, nargs="+",
                   default=[0, 50, 100, 250, 500, 1000])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--bandits", nargs="+", default=["linucb", "uniform"])
    p.add_argument("--mid-rounds", type=int, default=200,
                   help="# repair rounds to build the 'mid' starting schedule.")
    p.add_argument("--n-instances", type=int, default=6)
    p.add_argument("--scenarios", nargs="*", default=None,
                   help="Only run these scenario names (e.g. n005w4 n012w8).")
    args = p.parse_args()

    out_dir = ROOT / "experiments/exp11_ood_repair/results"
    out_dir.mkdir(parents=True, exist_ok=True)

    bundled = list(_enumerate_bundled(Path(args.dataset_root), args.scenarios))[:args.n_instances]
    print(f"[exp11] {len(bundled)} bundled instances", flush=True)

    out_path = out_dir / "curves.jsonl"
    f = out_path.open("w")
    t_total = time.perf_counter()

    for ds_name, h_idx, weeks, sol_dir in bundled:
        try:
            inst = _load_inst(Path(args.dataset_root), ds_name, h_idx, weeks)
        except Exception as e:
            print(f"  [skip {ds_name} H{h_idx} W{weeks}]: {e}", flush=True)
            continue
        print(f"\n[exp11] {ds_name} H{h_idx} W{weeks}", flush=True)

        for seed in args.seeds:
            # Build all three start schedules (deterministic-ish; depends on seed)
            starts: dict[str, Schedule] = {}
            try:
                starts["greedy"] = generate_initial_schedule(
                    inst.scenario, inst.initial_history, inst.weeks,
                )
                starts["mid"] = _greedy_then_n_rounds(
                    inst, args.repair_checkpoint, args.mid_rounds, seed,
                )
                starts["bandp"] = _load_external_schedule(inst.scenario, sol_dir)
            except Exception as e:
                print(f"  [skip starts seed={seed}]: {e}", flush=True)
                continue

            for start_name, sched in starts.items():
                init_pen = compute_penalty(
                    sched, inst.scenario, inst.weeks, inst.initial_history,
                ).total
                for bandit in args.bandits:
                    for n_rounds in args.rounds_list:
                        # Each repair call mutates schedule in place — clone first.
                        sched_copy = Schedule.from_dict(sched.to_dict()) if hasattr(sched, "to_dict") and hasattr(Schedule, "from_dict") else sched
                        try:
                            # Safer: re-build the start each time to avoid mutation issues.
                            if start_name == "greedy":
                                start_sched = generate_initial_schedule(
                                    inst.scenario, inst.initial_history, inst.weeks,
                                )
                            elif start_name == "mid":
                                start_sched = _greedy_then_n_rounds(
                                    inst, args.repair_checkpoint, args.mid_rounds, seed,
                                )
                            else:
                                start_sched = _load_external_schedule(inst.scenario, sol_dir)

                            res = run_repair_only(
                                inst, start_sched,
                                repair_bandit=bandit,
                                repair_checkpoint=args.repair_checkpoint if bandit == "linucb" else None,
                                rounds=n_rounds,
                                seed=seed,
                            )
                            row = {
                                "dataset": ds_name,
                                "history_idx": h_idx,
                                "weeks": weeks,
                                "seed": seed,
                                "start": start_name,
                                "repair_bandit": bandit,
                                "rounds": n_rounds,
                                "initial_penalty": int(init_pen),
                                "final_penalty": int(res["final_penalty"]),
                                "delta": int(init_pen - res["final_penalty"]),
                                "t_repair_s": res["t_repair_s"],
                            }
                            f.write(json.dumps(row) + "\n"); f.flush()
                            print(f"  start={start_name} bandit={bandit} rounds={n_rounds} "
                                  f"P0={init_pen} -> {res['final_penalty']} "
                                  f"(Δ={init_pen - res['final_penalty']:+d})", flush=True)
                        except Exception as e:
                            print(f"  [skip start={start_name} bandit={bandit} R={n_rounds}]: {e}",
                                  flush=True)

    f.close()
    print(f"\n[exp11] done in {time.perf_counter() - t_total:.1f}s, wrote {out_path}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
