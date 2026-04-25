#!/usr/bin/env bash
# Train base week + repair LinUCB checkpoints reused across exps 02, 06, 09, 10, 11.
# Settings chosen to be substantial but bounded: ~200 instances train, 500 repair rounds,
# alpha=1.0, ρ_w=1000, ρ_r=50, warm starts at recommended values.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate

LOGDIR=experiments/_shared/logs
CKPTDIR=experiments/_shared/checkpoints
mkdir -p "$LOGDIR" "$CKPTDIR"

echo "[train_base] week-level LinUCB"
PYTHONPATH=src python -m week_level.train \
  --split train \
  --max-instances 200 \
  --week-combos-per-scenario 20 \
  --alpha 1.0 \
  --reward-scale 1000 \
  --warm-start-rounds 8 \
  --seed 0 \
  --checkpoint "$CKPTDIR/base_week.npz" \
  --log-every 20 \
  > "$LOGDIR/train_base_week.stdout.log" 2> "$LOGDIR/train_base_week.stderr.log"

echo "[train_base] repair-level LinUCB"
PYTHONPATH=src python -m repair_level.train \
  --split train \
  --max-instances 100 \
  --week-combos-per-scenario 10 \
  --alpha 1.0 \
  --reward-scale 50 \
  --warm-start-rounds 36 \
  --num-rounds 500 \
  --seed 0 \
  --checkpoint "$CKPTDIR/base_repair.npz" \
  --log-every 10 \
  > "$LOGDIR/train_base_repair.stdout.log" 2> "$LOGDIR/train_base_repair.stderr.log"

echo "[train_base] done. Checkpoints in $CKPTDIR/"
ls -la "$CKPTDIR/"
