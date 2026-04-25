"""Plot Exp 11: penalty vs rounds, one curve per (start, bandit)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).parent
SRC = HERE / "results" / "curves.jsonl"
FIG = HERE / "figures"


def main() -> None:
    if not SRC.exists():
        print(f"[exp11 plot] missing {SRC}")
        return
    rows = [json.loads(l) for l in SRC.read_text().splitlines() if l.strip()]
    if not rows:
        print("[exp11 plot] no rows")
        return
    FIG.mkdir(parents=True, exist_ok=True)

    # Average final_penalty per (start, bandit, rounds) across instances+seeds.
    bucket = defaultdict(list)
    init_bucket = defaultdict(list)
    for r in rows:
        bucket[(r["start"], r["repair_bandit"], r["rounds"])].append(r["final_penalty"])
        init_bucket[(r["start"], r["repair_bandit"], r["rounds"])].append(r["initial_penalty"])

    starts = sorted({r["start"] for r in rows})
    bandits = sorted({r["repair_bandit"] for r in rows})
    rounds_set = sorted({r["rounds"] for r in rows})

    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.get_cmap("tab10")
    style = {"linucb": "-", "uniform": "--"}
    color_map = {"greedy": "tab:blue", "mid": "tab:orange", "bandp": "tab:red"}

    for start in starts:
        for i, bandit in enumerate(bandits):
            ys = []
            for n in rounds_set:
                vs = bucket.get((start, bandit, n), [])
                ys.append(np.mean(vs) if vs else np.nan)
            ax.plot(rounds_set, ys, marker="o",
                    color=color_map.get(start, cmap(i)),
                    linestyle=style.get(bandit, "-"),
                    label=f"{start}/{bandit}")

    ax.set_xlabel("repair rounds")
    ax.set_ylabel("mean final penalty")
    ax.set_title("Repair penalty vs rounds, by starting schedule × bandit")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIG / "ood_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[exp11 plot] wrote {out}")

    # Per-start: Δ from initial vs rounds
    fig, axes = plt.subplots(1, len(starts), figsize=(5 * len(starts), 4), squeeze=False)
    for k, start in enumerate(starts):
        ax = axes[0, k]
        for bandit in bandits:
            ys = []
            for n in rounds_set:
                vs = bucket.get((start, bandit, n), [])
                inits = init_bucket.get((start, bandit, n), [])
                if vs and inits:
                    deltas = [i - f for i, f in zip(inits, vs)]
                    ys.append(np.mean(deltas))
                else:
                    ys.append(np.nan)
            ax.plot(rounds_set, ys, marker="o", linestyle=style.get(bandit, "-"),
                    label=bandit)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("repair rounds")
        ax.set_ylabel("mean Δ (initial − final)")
        ax.set_title(f"start={start}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Repair Δ vs rounds (positive = repair helped)")
    fig.tight_layout()
    out = FIG / "ood_delta_per_start.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[exp11 plot] wrote {out}")


if __name__ == "__main__":
    main()
