#!/usr/bin/env bash
# Exp 04: reward-scale ρ sweep at week and repair levels.
# Uses scripts/sweep.py twice (one per level).
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate

EXP=experiments/exp04_reward_scale
LOG="$EXP/logs"
RES="$EXP/results"
mkdir -p "$LOG" "$RES"

echo "[exp04] week-level ρ sweep"
PYTHONPATH=src python scripts/sweep.py \
  --level week \
  --train-split train --eval-split dev \
  --alphas 1.0 \
  --reward-scales 250 500 1000 2000 4000 \
  --warm-starts 8 \
  --max-train-instances 200 --max-eval-instances 8 \
  --week-combos-per-scenario 10 \
  --seed 0 \
  --out-dir "$RES/week_sweep" \
  > "$LOG/week_sweep.stdout.log" 2> "$LOG/week_sweep.stderr.log"

echo "[exp04] repair-level ρ sweep"
PYTHONPATH=src python scripts/sweep.py \
  --level repair \
  --train-split train --eval-split dev \
  --alphas 1.0 \
  --reward-scales 10 25 50 100 250 \
  --warm-starts 36 \
  --num-rounds 500 \
  --max-train-instances 60 --max-eval-instances 8 \
  --week-combos-per-scenario 5 \
  --seed 0 \
  --out-dir "$RES/repair_sweep" \
  > "$LOG/repair_sweep.stdout.log" 2> "$LOG/repair_sweep.stderr.log"

echo "[exp04] done"
