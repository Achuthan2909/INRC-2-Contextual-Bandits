"""Plot Exp 09: per-feature ablation bar charts."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).parent
RES = HERE / "results"
FIG = HERE / "figures"


def _plot(level: str) -> None:
    path = RES / f"{level}_ablation.json"
    if not path.exists():
        print(f"[exp09 plot] missing {path}")
        return
    data = json.loads(path.read_text())
    rows = data["rows"]
    if not rows:
        return
    full = next((r for r in rows if r["ablation"] == "full"), None)
    if full is None:
        return
    masked = [r for r in rows if r["ablation"] != "full"]

    labels = [r["ablation"].replace("mask:", "") for r in masked]
    delta_fp = [r["mean_final_penalty"] - full["mean_final_penalty"] for r in masked]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["tab:red" if d > 0 else "tab:green" for d in delta_fp]
    ax.barh(labels, delta_fp, color=colors)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_xlabel("Δ mean final penalty vs full context (positive = feature helps)")
    ax.set_title(f"{level}-level feature importance (ablation)")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    out = FIG / f"ablation_{level}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[exp09 plot] wrote {out}")


def main() -> None:
    for level in ("week", "repair"):
        _plot(level)


if __name__ == "__main__":
    main()
