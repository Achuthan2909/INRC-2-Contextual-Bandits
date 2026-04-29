#!/usr/bin/env bash
# Run Experiment 3 (alpha sweep) then Experiment 1 (round budget sweep).
# Must be run from the repo root:
#   cd /Users/riasonalker/Desktop/Sequential/project/INRC-2-Contextual-Bandits
#   bash run_exp1_and_3.sh

set -euo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"

mkdir -p runs/exp1_rounds

# ── Experiment 3a: Week-level alpha sweep ─────────────────────────────────────
echo ""
echo "=== EXP 3a: Week-level alpha sweep ==="
PYTHONPATH=src python scripts/sweep.py \
  --level week \
  --train-split train --eval-split dev \
  --alphas 0.25 0.5 1.0 2.0 \
  --reward-scales 1000 \
  --warm-starts 8 \
  --max-train-instances 50 --max-eval-instances 10 \
  --out-dir runs/exp3_alpha_week

# Pick best week checkpoint (highest mean_delta in results.csv)
BEST_WEEK=$(python3 - <<'PY'
import csv, sys
rows = list(csv.DictReader(open("runs/exp3_alpha_week/results.csv")))
best = max(rows, key=lambda r: float(r["mean_delta"]))
print(best["checkpoint"])
PY
)
echo "Best week checkpoint: $BEST_WEEK"
cp "$BEST_WEEK" runs/final_week.npz

# Plot diagnostics for best week checkpoint
BEST_WEEK_SIDECAR="${BEST_WEEK%.npz}.trajectory.json"
if [ -f "$BEST_WEEK_SIDECAR" ]; then
  echo "Generating week-level training plots..."
  python scripts/plot_training.py \
    --sidecar "$BEST_WEEK_SIDECAR" \
    --out-dir plots/exp3_week
fi

# ── Experiment 3b: Repair-level alpha sweep ───────────────────────────────────
echo ""
echo "=== EXP 3b: Repair-level alpha sweep ==="
PYTHONPATH=src python scripts/sweep.py \
  --level repair \
  --train-split train --eval-split dev \
  --alphas 0.25 0.5 1.0 2.0 \
  --reward-scales 50 \
  --warm-starts 36 \
  --num-rounds 500 \
  --max-train-instances 30 --max-eval-instances 10 \
  --out-dir runs/exp3_alpha_repair

# Pick best repair checkpoint (highest mean_delta in results.csv)
BEST_REPAIR=$(python3 - <<'PY'
import csv, sys
rows = list(csv.DictReader(open("runs/exp3_alpha_repair/results.csv")))
best = max(rows, key=lambda r: float(r["mean_delta"]))
print(best["checkpoint"])
PY
)
echo "Best repair checkpoint: $BEST_REPAIR"
cp "$BEST_REPAIR" runs/final_repair.npz

# Plot diagnostics for best repair checkpoint
BEST_REPAIR_SIDECAR="${BEST_REPAIR%.npz}.trajectory.json"
if [ -f "$BEST_REPAIR_SIDECAR" ]; then
  echo "Generating repair-level training plots..."
  python scripts/plot_training.py \
    --sidecar "$BEST_REPAIR_SIDECAR" \
    --out-dir plots/exp3_repair
fi

# ── Experiment 1: Round budget sweep ─────────────────────────────────────────
echo ""
echo "=== EXP 1: Round budget sweep ==="
for R in 50 100 250 500 1000 2000 4000; do
  for SEED in 0 1 2; do
    echo "  rounds=$R seed=$SEED"
    PYTHONPATH=src python scripts/run_pipeline.py \
      --dataset-root Dataset/testdatasets_json --dataset n021w4 --weeks 0 1 2 3 \
      --week-bandit linucb --week-checkpoint runs/final_week.npz \
      --repair-bandit linucb --repair-checkpoint runs/final_repair.npz \
      --rounds "$R" --seed "$SEED" \
      --output "runs/exp1_rounds/R${R}_s${SEED}.json"
  done
done

echo ""
echo "Done. Results in:"
echo "  runs/exp3_alpha_week/results.csv"
echo "  runs/exp3_alpha_repair/results.csv"
echo "  runs/exp1_rounds/"
echo "  plots/exp3_week/   (reward_rolling, arm_usage, theta_norms, heatmap)"
echo "  plots/exp3_repair/ (+ penalty_reduction)"
