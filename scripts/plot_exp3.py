"""Plot Experiment 3 alpha sweep results — week and repair levels side by side.

Example:
    python scripts/plot_exp3.py --out plots/exp3_alpha_sweep.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(path: str) -> list[dict]:
    return sorted(json.load(open(path)), key=lambda r: r["alpha"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--week-results", default="runs/exp3_alpha_week/results.json")
    p.add_argument("--repair-results", default="runs/exp3_alpha_repair/results.json")
    p.add_argument("--out", default="plots/exp3_alpha_sweep.png")
    args = p.parse_args()

    week = load_results(args.week_results)
    repair = load_results(args.repair_results)

    COLOR_FINAL = "#2ecc71"
    COLOR_DELTA = "#2980b9"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Experiment 3: Alpha Sweep — LinUCB Exploration Parameter", fontsize=13, y=1.02)

    for ax, rows, title in [
        (axes[0], week,   "Week-level LinUCB"),
        (axes[1], repair, "Repair-level LinUCB"),
    ]:
        alphas = [r["alpha"] for r in rows]
        mean_finals = [r["mean_final_penalty"] for r in rows]
        mean_deltas = [r["mean_delta"] for r in rows]

        x = np.arange(len(alphas))
        width = 0.38

        bars1 = ax.bar(x - width / 2, mean_finals, width, label="Mean Final Penalty",
                       color=COLOR_FINAL, edgecolor="white", zorder=3)
        bars2 = ax.bar(x + width / 2, mean_deltas, width, label="Mean Δ (penalty reduction)",
                       color=COLOR_DELTA, edgecolor="white", zorder=3)

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                    f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=8)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                    f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([f"α={a}" for a in alphas])
        ax.set_xlabel("Alpha (exploration parameter)", fontsize=11)
        ax.set_ylabel("Penalty", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.set_axisbelow(True)

    # Single shared legend below both panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, -0.08), framealpha=0.9)

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
