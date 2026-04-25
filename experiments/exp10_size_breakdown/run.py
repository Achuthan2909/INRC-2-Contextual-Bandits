"""Exp 10: pivot Exp 02 per_run.jsonl by scenario size + width.

Reads experiments/exp02_bandit_grid/results/per_run.jsonl and aggregates
mean final-penalty per (week_bandit, repair_bandit, scenario_size) cell.
Writes a pivoted CSV + JSON and a heatmap per (week,repair) bandit pair.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

SRC = ROOT / "experiments/exp02_bandit_grid/results/per_run.jsonl"
OUT = ROOT / "experiments/exp10_size_breakdown/results"
SIZE_RE = re.compile(r"^n(\d+)w(\d+)$")


def main() -> int:
    if not SRC.exists():
        raise SystemExit(f"missing source: {SRC} (run exp02 first)")
    OUT.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in SRC.read_text().splitlines() if l.strip()]
    print(f"[exp10] loaded {len(rows)} per-run rows")

    # group by (week_bandit, repair_bandit, scenario_size, scenario_weeks)
    bucket: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        ds = r.get("dataset", "")
        m = SIZE_RE.match(ds)
        if not m:
            continue
        size = int(m.group(1))
        weeks = int(m.group(2))
        key = (r["week_bandit"], r["repair_bandit"], size, weeks)
        bucket[key].append(float(r["final_penalty"]))

    pivoted: list[dict] = []
    for (wb, rb, size, weeks), vals in sorted(bucket.items()):
        pivoted.append({
            "week_bandit": wb,
            "repair_bandit": rb,
            "size": size,
            "weeks": weeks,
            "n": len(vals),
            "mean_final_penalty": float(np.mean(vals)),
            "std_final_penalty": float(np.std(vals)),
        })

    csv_path = OUT / "size_breakdown.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pivoted[0].keys()) if pivoted else
                           ["week_bandit","repair_bandit","size","weeks","n","mean_final_penalty","std_final_penalty"])
        w.writeheader()
        for row in pivoted:
            w.writerow(row)
    (OUT / "size_breakdown.json").write_text(json.dumps(pivoted, indent=2))
    print(f"[exp10] wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
