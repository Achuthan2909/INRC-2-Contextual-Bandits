"""Plot Exp 05: penalty / Δ vs warm-start length per level."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


HERE = Path(__file__).parent
RES = HERE / "results"
FIG = HERE / "figures"


def _plot(level: str) -> None:
    csv_path = RES / f"{level}_sweep" / "results.csv"
    if not csv_path.exists():
        print(f"[exp05 plot] missing {csv_path}")
        return
    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        return
    rows.sort(key=lambda r: int(r["warm_start"]))
    ws = [int(r["warm_start"]) for r in rows]
    fp = [float(r["mean_final_penalty"]) for r in rows]
    delta = [float(r["mean_delta"]) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(ws, fp, marker="o")
    axes[0].set_xlabel(f"warm-start rounds ({level})")
    axes[0].set_ylabel("mean final penalty")
    axes[0].set_title(f"{level}: held-out final penalty vs warm-start")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(ws, delta, marker="o", color="tab:green")
    axes[1].set_xlabel(f"warm-start rounds ({level})")
    axes[1].set_ylabel("mean Δ (held-out)")
    axes[1].set_title(f"{level}: held-out Δ vs warm-start")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG / f"warm_start_{level}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[exp05 plot] wrote {out}")


def main() -> None:
    for level in ("week", "repair"):
        _plot(level)


if __name__ == "__main__":
    main()
