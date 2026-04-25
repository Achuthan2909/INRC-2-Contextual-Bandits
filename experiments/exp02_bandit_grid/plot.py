"""Plot Exp 02: 5x5 heatmap of mean final penalty per (week,repair) cell."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).parent
RESULTS = HERE / "results" / "grid.json"
FIG_DIR = HERE / "figures"


def main() -> None:
    data = json.loads(RESULTS.read_text())
    cells = data["grid"]
    bandits = sorted({c["week_bandit"] for c in cells})
    n = len(bandits)
    idx = {b: i for i, b in enumerate(bandits)}

    fp_mean = np.full((n, n), np.nan)
    fp_std = np.full((n, n), np.nan)
    delta_mean = np.full((n, n), np.nan)

    for c in cells:
        i = idx[c["week_bandit"]]
        j = idx[c["repair_bandit"]]
        finals = [r["final_penalty"] for r in c["runs"]]
        deltas = [r["delta_repair"] for r in c["runs"]]
        if finals:
            fp_mean[i, j] = float(np.mean(finals))
            fp_std[i, j] = float(np.std(finals))
            delta_mean[i, j] = float(np.mean(deltas))

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Heatmap of final penalty
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(fp_mean, cmap="viridis_r")
    ax.set_xticks(range(n)); ax.set_xticklabels(bandits, rotation=30, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(bandits)
    ax.set_xlabel("repair bandit")
    ax.set_ylabel("week bandit")
    ax.set_title("Mean final penalty (lower = better)")
    for i in range(n):
        for j in range(n):
            if not np.isnan(fp_mean[i, j]):
                ax.text(j, i, f"{fp_mean[i,j]:.0f}", ha="center", va="center",
                        color="white", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "grid_final_penalty.png", dpi=150)
    plt.close(fig)

    # Heatmap of repair Δ
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(delta_mean, cmap="RdYlGn")
    ax.set_xticks(range(n)); ax.set_xticklabels(bandits, rotation=30, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(bandits)
    ax.set_xlabel("repair bandit")
    ax.set_ylabel("week bandit")
    ax.set_title("Mean repair Δ (higher = better)")
    for i in range(n):
        for j in range(n):
            if not np.isnan(delta_mean[i, j]):
                ax.text(j, i, f"{delta_mean[i,j]:+.0f}", ha="center", va="center",
                        color="black", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "grid_delta.png", dpi=150)
    plt.close(fig)

    print(f"[exp02 plot] wrote {FIG_DIR}/grid_*.png")


if __name__ == "__main__":
    main()
