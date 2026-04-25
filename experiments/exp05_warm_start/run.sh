#!/usr/bin/env bash
# Exp 05: warm-start length sweep at week and repair levels.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate

EXP=experiments/exp05_warm_start
LOG="$EXP/logs"
RES="$EXP/results"
mkdir -p "$LOG" "$RES"

echo "[exp05] week-level warm-start sweep"
PYTHONPATH=src python scripts/sweep.py \
  --level week \
  --train-split train --eval-split dev \
  --alphas 1.0 \
  --reward-scales 1000 \
  --warm-starts 0 4 8 16 \
  --max-train-instances 200 --max-eval-instances 8 \
  --week-combos-per-scenario 10 \
  --seed 0 \
  --out-dir "$RES/week_sweep" \
  > "$LOG/week_sweep.stdout.log" 2> "$LOG/week_sweep.stderr.log"

echo "[exp05] repair-level warm-start sweep"
PYTHONPATH=src python scripts/sweep.py \
  --level repair \
  --train-split train --eval-split dev \
  --alphas 1.0 \
  --reward-scales 50 \
  --warm-starts 0 18 36 72 144 \
  --num-rounds 500 \
  --max-train-instances 60 --max-eval-instances 8 \
  --week-combos-per-scenario 5 \
  --seed 0 \
  --out-dir "$RES/repair_sweep" \
  > "$LOG/repair_sweep.stdout.log" 2> "$LOG/repair_sweep.stderr.log"

echo "[exp05] done"
