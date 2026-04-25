"""Plot Exp 10: per-size breakdown of LinUCB advantage over baselines."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).parent
SRC = HERE / "results" / "size_breakdown.json"
FIG = HERE / "figures"


def main() -> None:
    if not SRC.exists():
        print(f"[exp10 plot] missing {SRC}")
        return
    rows = json.loads(SRC.read_text())
    if not rows:
        print("[exp10 plot] no rows")
        return
    FIG.mkdir(parents=True, exist_ok=True)

    # bucket by (size, weeks); compare bandit pairs.
    by_size = defaultdict(list)
    for r in rows:
        by_size[(r["size"], r["weeks"])].append(r)

    sizes = sorted(by_size.keys())
    bandits_pairs = sorted({(r["week_bandit"], r["repair_bandit"]) for r in rows})

    # Bar chart: x = (size, weeks), groups = bandit pairs, y = mean final penalty.
    x_labels = [f"n{s}w{w}" for (s, w) in sizes]
    x = np.arange(len(sizes))
    width = 0.8 / max(len(bandits_pairs), 1)

    fig, ax = plt.subplots(figsize=(max(8, len(sizes) * 1.2), 5))
    for i, (wb, rb) in enumerate(bandits_pairs):
        ys = []
        for sz in sizes:
            ms = [r for r in by_size[sz] if r["week_bandit"] == wb and r["repair_bandit"] == rb]
            ys.append(ms[0]["mean_final_penalty"] if ms else np.nan)
        ax.bar(x + i * width - 0.4, ys, width, label=f"{wb[:3]}/{rb[:3]}")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
    ax.set_ylabel("mean final penalty")
    ax.set_title("Final penalty by scenario size × bandit pair")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out = FIG / "size_bars.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[exp10 plot] wrote {out}")

    # Heatmap: LinUCB/LinUCB final penalty vs uniform/uniform per size
    lu = defaultdict(float)
    un = defaultdict(float)
    for r in rows:
        if r["week_bandit"] == "linucb" and r["repair_bandit"] == "linucb":
            lu[(r["size"], r["weeks"])] = r["mean_final_penalty"]
        if r["week_bandit"] == "uniform" and r["repair_bandit"] == "uniform":
            un[(r["size"], r["weeks"])] = r["mean_final_penalty"]
    advantage = []
    for sz in sizes:
        if lu.get(sz) and un.get(sz):
            advantage.append((sz, un[sz] - lu[sz]))
    if advantage:
        fig, ax = plt.subplots(figsize=(8, 4))
        labels = [f"n{s}w{w}" for (s, w), _ in advantage]
        vals = [v for _, v in advantage]
        ax.bar(labels, vals, color=["tab:green" if v > 0 else "tab:red" for v in vals])
        ax.axhline(0, color="black", lw=0.7)
        ax.set_ylabel("Δ final penalty (uniform − LinUCB)\npositive = LinUCB better")
        ax.set_title("LinUCB advantage over uniform×uniform per scenario size")
        for tick in ax.get_xticklabels():
            tick.set_rotation(30); tick.set_ha("right")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        out2 = FIG / "linucb_advantage_by_size.png"
        fig.savefig(out2, dpi=150)
        plt.close(fig)
        print(f"[exp10 plot] wrote {out2}")


if __name__ == "__main__":
    main()
