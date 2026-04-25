"""Exp 09: context ablation — mask one feature at a time, retrain, evaluate.

Strategy: monkey-patch the relevant context builder to zero out the masked
feature index *both* during training and evaluation. Re-train a fresh
LinUCB per ablation, then run the held-out evaluation pipeline.

Outputs:
  results/{level}_ablation.json — per-ablation final-penalty/Δ rows.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from _shared.eval_pipeline import (  # noqa: E402
    get_eval_instances,
    run_pipeline_one,
    write_json,
)


def _mask_week_feature(idx: int | None) -> None:
    """Wrap week_level.context_builder.build_context to zero feature ``idx``."""
    import week_level.context_builder as cb

    if not hasattr(cb, "_orig_build_context"):
        cb._orig_build_context = cb.build_context

    def wrapped(*args, **kwargs):
        ctx, labels = cb._orig_build_context(*args, **kwargs)
        if idx is not None:
            ctx = ctx.copy()
            ctx[idx] = 0.0
        return ctx, labels

    cb.build_context = wrapped
    # Also patch the import inside week_level.runner if it imported by name.
    import week_level.runner as wr
    if hasattr(wr, "build_context"):
        wr.build_context = wrapped
    import week_level.train as wt
    if hasattr(wt, "build_context"):
        wt.build_context = wrapped


def _mask_repair_feature(idx: int | None) -> None:
    import repair_level.context as rc

    if not hasattr(rc, "_orig_build_repair_context"):
        rc._orig_build_repair_context = rc.build_repair_context

    def wrapped(penalty, round_idx, total_rounds):
        ctx = rc._orig_build_repair_context(penalty, round_idx, total_rounds)
        if idx is not None:
            ctx = ctx.copy()
            ctx[idx] = 0.0
        return ctx

    rc.build_repair_context = wrapped
    import repair_level.linucb_selector as ls
    if hasattr(ls, "build_repair_context"):
        ls.build_repair_context = wrapped
    import repair_level.runner as rr
    if hasattr(rr, "build_repair_context"):
        rr.build_repair_context = wrapped


def _train_week(ckpt: Path, max_inst: int) -> None:
    from week_level.arms import (
        CoverageFirstArm, FatigueAwareArm, PreferenceRespectingArm, WeekendBalancingArm,
    )
    from week_level.train import train_linucb
    arms = [CoverageFirstArm(), FatigueAwareArm(), WeekendBalancingArm(),
            PreferenceRespectingArm()]
    train_linucb(
        arms,
        split="train",
        alpha=1.0,
        max_instances=max_inst,
        week_combos_per_scenario=10,
        seed=0,
        reward_scale=1000.0,
        checkpoint_path=str(ckpt),
        log_every=0,
        warm_start_rounds=8,
    )


def _train_repair(ckpt: Path, max_inst: int) -> None:
    from repair_level.train import train_linucb_repair
    train_linucb_repair(
        split="train",
        alpha=1.0,
        max_instances=max_inst,
        week_combos_per_scenario=5,
        seed=0,
        reward_scale=50.0,
        num_rounds=500,
        checkpoint_path=str(ckpt),
        log_every=0,
        warm_start_rounds=36,
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--level", choices=["week", "repair"], required=True)
    p.add_argument("--n-eval", type=int, default=8)
    p.add_argument("--rounds", type=int, default=500)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--max-train-instances", type=int, default=100)
    args = p.parse_args()

    out_dir = ROOT / "experiments/exp09_context_ablation/results"
    ckpt_dir = ROOT / "experiments/exp09_context_ablation/checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.level == "week":
        from week_level.context_builder import FEATURE_LABELS
        labels = list(FEATURE_LABELS)
        mask_fn = _mask_week_feature
        train_fn = _train_week
    else:
        from repair_level.context import FEATURE_LABELS
        labels = list(FEATURE_LABELS)
        mask_fn = _mask_repair_feature
        train_fn = _train_repair

    base_repair_ckpt = ROOT / "experiments/_shared/checkpoints/base_repair.npz"
    base_week_ckpt = ROOT / "experiments/_shared/checkpoints/base_week.npz"

    instances = get_eval_instances("dev", args.n_eval, seed=0)
    print(f"[exp09:{args.level}] instances={[i.dataset_name for i in instances]}",
          flush=True)

    # Ablations: full + one masked feature at a time.
    rows = []
    ablations: list[tuple[str, int | None]] = [("full", None)]
    for i, lab in enumerate(labels):
        ablations.append((f"mask:{lab}", i))

    for tag, idx in ablations:
        print(f"\n[exp09:{args.level}] ablation={tag}", flush=True)
        # Patch context builder.
        mask_fn(idx)
        ckpt = ckpt_dir / f"{args.level}_{tag.replace(':','_')}.npz"
        t0 = time.perf_counter()
        train_fn(ckpt, args.max_train_instances)
        train_s = time.perf_counter() - t0

        # Evaluate
        finals: list[int] = []
        deltas: list[int] = []
        per: list[dict] = []
        for seed in args.seeds:
            for inst in instances:
                try:
                    if args.level == "week":
                        # Full pipeline with this week ckpt + base repair
                        out = run_pipeline_one(
                            inst,
                            week_bandit="linucb",
                            week_checkpoint=str(ckpt),
                            repair_bandit="linucb",
                            repair_checkpoint=str(base_repair_ckpt),
                            rounds=args.rounds,
                            seed=seed,
                        )
                    else:
                        out = run_pipeline_one(
                            inst,
                            week_bandit="linucb",
                            week_checkpoint=str(base_week_ckpt),
                            repair_bandit="linucb",
                            repair_checkpoint=str(ckpt),
                            rounds=args.rounds,
                            seed=seed,
                        )
                    finals.append(out["final_penalty"])
                    deltas.append(out["delta_repair"])
                    per.append({"seed": seed, "dataset": inst.dataset_name,
                                "final_penalty": out["final_penalty"],
                                "delta_repair": out["delta_repair"]})
                except Exception as e:
                    print(f"  [skip {tag} seed={seed} {inst.dataset_name}]: {e}",
                          flush=True)

        rows.append({
            "ablation": tag,
            "feature_idx": idx,
            "train_s": train_s,
            "n_runs": len(finals),
            "mean_final_penalty": float(np.mean(finals)) if finals else float("nan"),
            "std_final_penalty": float(np.std(finals)) if finals else float("nan"),
            "mean_delta": float(np.mean(deltas)) if deltas else float("nan"),
            "per_run": per,
        })
        print(f"[exp09:{args.level}] {tag}: mean_final={rows[-1]['mean_final_penalty']:.1f} "
              f"mean_Δ={rows[-1]['mean_delta']:.1f} (train {train_s:.1f}s)",
              flush=True)

        # Restore unmasked builder for the next iteration.
        mask_fn(None)

    write_json(out_dir / f"{args.level}_ablation.json", {
        "config": vars(args),
        "feature_labels": labels,
        "rows": rows,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
