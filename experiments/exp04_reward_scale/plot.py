"""Plot Exp 04: penalty / Δ vs reward scale (per level)."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


HERE = Path(__file__).parent
RES = HERE / "results"
FIG = HERE / "figures"


def _load_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _plot(rows: list[dict], level: str) -> None:
    if not rows:
        print(f"[exp04 plot] no rows for {level}")
        return
    rows.sort(key=lambda r: float(r["reward_scale"]))
    rs = [float(r["reward_scale"]) for r in rows]
    fp = [float(r["mean_final_penalty"]) for r in rows]
    delta = [float(r["mean_delta"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(rs, fp, marker="o")
    axes[0].set_xscale("log")
    axes[0].set_xlabel(f"ρ ({level})")
    axes[0].set_ylabel("mean final penalty")
    axes[0].set_title(f"{level}: held-out final penalty vs ρ")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(rs, delta, marker="o", color="tab:green")
    axes[1].set_xscale("log")
    axes[1].set_xlabel(f"ρ ({level})")
    axes[1].set_ylabel("mean Δ (held-out)")
    axes[1].set_title(f"{level}: held-out Δ vs ρ")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG / f"reward_scale_{level}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[exp04 plot] wrote {out}")


def main() -> None:
    for level in ("week", "repair"):
        csv_path = RES / f"{level}_sweep" / "results.csv"
        if not csv_path.exists():
            print(f"[exp04 plot] missing {csv_path}")
            continue
        _plot(_load_csv(csv_path), level)


if __name__ == "__main__":
    main()
