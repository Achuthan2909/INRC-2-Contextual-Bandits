"""Smoke test: LinUCB-contextual repair selector on one n030w4 instance."""
from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from instance_loader import load_instance  # noqa: E402
from repair_level.init import generate_initial_schedule  # noqa: E402
from repair_level.linucb_selector import LinUCBRepairSelector  # noqa: E402
from repair_level.repairs import build_all_strategies  # noqa: E402
from repair_level.runner import run_repairs  # noqa: E402


def main() -> int:
    inst = load_instance(
        dataset_root="Dataset/datasets_json",
        dataset_name="n030w4",
        history_idx=0,
        week_indices=[0, 1, 2, 3],
    )
    scenario, history, weeks = inst.scenario, inst.initial_history, inst.weeks

    strategies = build_all_strategies(scenario, history, weeks, seed=0)
    print(f"strategies: {[s.name for s in strategies]}")

    selector = LinUCBRepairSelector(
        strategy_names=[s.name for s in strategies],
        alpha=1.0,
        reward_scale=50.0,
        seed=0,
    )

    schedule = generate_initial_schedule(scenario, history, weeks)

    t0 = time.perf_counter()
    out = run_repairs(
        scenario=scenario,
        history=history,
        week_data_list=weeks,
        strategies=strategies,
        schedule=schedule,
        selector=selector,
        num_rounds=500,
        seed=0,
    )
    dt = time.perf_counter() - t0

    print(f"\nRan {out.rounds_run} rounds in {dt:.2f}s")
    print(f"initial_penalty = {out.initial_penalty}")
    print(f"final_penalty   = {out.final_penalty}  (Δ = {out.initial_penalty - out.final_penalty:+d})")
    print(f"hard violations = {out.hard_violations}")
    print(f"attempted/succeeded = {out.total_attempted}/{out.total_succeeded}")

    print("\nPick distribution:")
    c = Counter(out.strategy_counts)
    total = sum(c.values()) or 1
    for name, n in sorted(c.items(), key=lambda x: -x[1]):
        print(f"  {name:28s} {n:5d}  ({100 * n / total:5.1f}%)")

    print("\nθ per arm (7 features):")
    import numpy as np
    for i, name in enumerate(selector.strategy_names):
        theta = selector.linucb.theta(i)
        print(f"  {name:28s} ||θ||={np.linalg.norm(theta):6.3f}  "
              f"[{', '.join(f'{v:+.2f}' for v in theta)}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
