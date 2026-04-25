#!/usr/bin/env bash
# Exp 06: training data scaling (learning curve of held-out penalty vs train size).
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate

EXP=experiments/exp06_data_scaling
LOG="$EXP/logs"
RES="$EXP/results"
CKPT="$EXP/checkpoints"
mkdir -p "$LOG" "$RES" "$CKPT"

# Use sweep.py with a single (alpha, rs, ws) combo per level — repeat across
# --max-train-instances by re-invoking. Sweep only iterates over (alpha,rs,ws),
# so we externally loop max-train-instances.

sizes_week="10 25 50 100 200"
sizes_repair="10 25 50 100"

echo "[exp06] week-level scaling"
for N in $sizes_week; do
  echo "  N_train=$N"
  PYTHONPATH=src python scripts/sweep.py \
    --level week \
    --train-split train --eval-split dev \
    --alphas 1.0 --reward-scales 1000 --warm-starts 8 \
    --max-train-instances "$N" --max-eval-instances 8 \
    --week-combos-per-scenario 10 \
    --seed 0 \
    --out-dir "$RES/week_n${N}" \
    >> "$LOG/week.stdout.log" 2>> "$LOG/week.stderr.log"
done

echo "[exp06] repair-level scaling"
for N in $sizes_repair; do
  echo "  N_train=$N"
  PYTHONPATH=src python scripts/sweep.py \
    --level repair \
    --train-split train --eval-split dev \
    --alphas 1.0 --reward-scales 50 --warm-starts 36 \
    --num-rounds 500 \
    --max-train-instances "$N" --max-eval-instances 8 \
    --week-combos-per-scenario 5 \
    --seed 0 \
    --out-dir "$RES/repair_n${N}" \
    >> "$LOG/repair.stdout.log" 2>> "$LOG/repair.stderr.log"
done

echo "[exp06] done"
