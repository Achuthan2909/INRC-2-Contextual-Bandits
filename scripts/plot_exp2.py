"""Plot Experiment 2 results as a 5x5 heatmap (mean final penalty).

Example:
    python scripts/plot_exp2.py --results runs/exp2/grid_summary.json --out plots/exp2_heatmap.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

BANDIT_ORDER = ["uniform", "ucb1", "epsilon_greedy", "thompson", "linucb"]
BANDIT_LABELS = ["Uniform", "UCB1", "Eps-Greedy", "Thompson", "LinUCB"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="runs/exp2/grid_summary.json")
    p.add_argument("--out", default="plots/exp2_heatmap.png")
    args = p.parse_args()

    data = json.load(open(args.results))

    # Build 5x5 matrix: rows = week bandit, cols = repair bandit
    n = len(BANDIT_ORDER)
    matrix = np.full((n, n), np.nan)
    for row in data:
        r = BANDIT_ORDER.index(row["repair_bandit"])
        w = BANDIT_ORDER.index(row["week_bandit"])
        matrix[w, r] = row["mean_final_penalty"]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=False,
        cmap="RdYlGn_r",
        xticklabels=BANDIT_LABELS,
        yticklabels=BANDIT_LABELS,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Mean Final Penalty (lower is better)"},
        vmin=np.nanmin(matrix),
        vmax=np.nanmax(matrix),
    )

    # Manually annotate every cell so seaborn's internal path can't skip any
    norm = plt.Normalize(vmin=np.nanmin(matrix), vmax=np.nanmax(matrix))
    cmap = plt.get_cmap("RdYlGn_r")
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            r, g, b, _ = cmap(norm(val))
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            color = "white" if luminance < 0.5 else "black"
            ax.text(j + 0.5, i + 0.5, f"{val:.0f}",
                    ha="center", va="center", fontsize=10, color=color)

    # Highlight the best cell
    best_val = np.nanmin(matrix)
    best_w, best_r = np.unravel_index(np.nanargmin(matrix), matrix.shape)
    ax.add_patch(plt.Rectangle(
        (best_r, best_w), 1, 1,
        fill=False, edgecolor="black", lw=2.5, clip_on=False,
    ))

    ax.set_xlabel("Repair-level Bandit", fontsize=12, labelpad=8)
    ax.set_ylabel("Week-level Bandit", fontsize=12, labelpad=8)
    ax.set_title("Experiment 2: Bandit vs. Baseline Grid\nMean Final Penalty (3 seeds, dev split)",
                 fontsize=12, pad=12)
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved: {args.out}")
    print(f"Best cell: week={BANDIT_ORDER[best_w]} + repair={BANDIT_ORDER[best_r]} → {best_val:.0f}")


if __name__ == "__main__":
    main()
