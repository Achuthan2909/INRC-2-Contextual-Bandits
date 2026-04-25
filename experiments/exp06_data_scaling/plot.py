"""Plot Exp 06: held-out penalty vs # training instances."""
from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


HERE = Path(__file__).parent
RES = HERE / "results"
FIG = HERE / "figures"


def _collect(level: str) -> list[tuple[int, float, float]]:
    out: list[tuple[int, float, float]] = []
    for sub in sorted(RES.glob(f"{level}_n*")):
        m = re.match(rf"{level}_n(\d+)$", sub.name)
        if not m:
            continue
        n = int(m.group(1))
        csv_path = sub / "results.csv"
        if not csv_path.exists():
            continue
        rows = list(csv.DictReader(csv_path.open()))
        if not rows:
            continue
        r = rows[0]
        out.append((n, float(r["mean_final_penalty"]), float(r["mean_delta"])))
    out.sort()
    return out


def _plot(level: str) -> None:
    data = _collect(level)
    if not data:
        print(f"[exp06 plot] no data for {level}")
        return
    xs = [d[0] for d in data]
    fp = [d[1] for d in data]
    delta = [d[2] for d in data]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(xs, fp, marker="o")
    axes[0].set_xlabel(f"# training instances ({level})")
    axes[0].set_ylabel("mean final penalty")
    axes[0].set_title(f"{level}: learning curve (final penalty)")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(xs, delta, marker="o", color="tab:green")
    axes[1].set_xlabel(f"# training instances ({level})")
    axes[1].set_ylabel("mean Δ (held-out)")
    axes[1].set_title(f"{level}: learning curve (Δ)")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG / f"data_scaling_{level}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[exp06 plot] wrote {out}")


def main() -> None:
    for level in ("week", "repair"):
        _plot(level)


if __name__ == "__main__":
    main()
