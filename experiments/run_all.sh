#!/usr/bin/env bash
# Sequentially run every experiment and its plot stage.
# Each experiment writes:
#   experiments/expXX_*/logs/{stdout,stderr}.log
#   experiments/expXX_*/results/*
#   experiments/expXX_*/figures/*.png
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=src

mkdir -p experiments/_shared/checkpoints
if [[ ! -f experiments/_shared/checkpoints/base_week.npz ]] || \
   [[ ! -f experiments/_shared/checkpoints/base_repair.npz ]]; then
  echo "[run_all] training base checkpoints"
  bash experiments/_shared/train_base.sh
fi

run_exp() {
  local exp="$1"; shift
  local cmd="$*"
  local logdir="experiments/${exp}/logs"
  mkdir -p "$logdir"
  echo "[run_all] >>> $exp"
  bash -c "$cmd" \
    > "$logdir/run.stdout.log" 2> "$logdir/run.stderr.log"
  echo "[run_all] <<< $exp done"
}

# Exp 02: bandit grid (also produces input for Exp 10)
run_exp exp02_bandit_grid "python experiments/exp02_bandit_grid/run.py --rounds 500 --seeds 0 1 2 --n-eval 8"
python experiments/exp02_bandit_grid/plot.py 2>&1 | tee -a experiments/exp02_bandit_grid/logs/plot.log

# Exp 04: reward scale
run_exp exp04_reward_scale "bash experiments/exp04_reward_scale/run.sh"
python experiments/exp04_reward_scale/plot.py 2>&1 | tee -a experiments/exp04_reward_scale/logs/plot.log

# Exp 05: warm start
run_exp exp05_warm_start "bash experiments/exp05_warm_start/run.sh"
python experiments/exp05_warm_start/plot.py 2>&1 | tee -a experiments/exp05_warm_start/logs/plot.log

# Exp 06: data scaling
run_exp exp06_data_scaling "bash experiments/exp06_data_scaling/run.sh"
python experiments/exp06_data_scaling/plot.py 2>&1 | tee -a experiments/exp06_data_scaling/logs/plot.log

# Exp 09: context ablation (week + repair)
run_exp exp09_context_ablation "python experiments/exp09_context_ablation/run.py --level week --max-train-instances 100 --n-eval 6 --seeds 0 1 2; python experiments/exp09_context_ablation/run.py --level repair --max-train-instances 60 --n-eval 6 --seeds 0 1 2"
python experiments/exp09_context_ablation/plot.py 2>&1 | tee -a experiments/exp09_context_ablation/logs/plot.log

# Exp 10: pivot Exp 02
run_exp exp10_size_breakdown "python experiments/exp10_size_breakdown/run.py"
python experiments/exp10_size_breakdown/plot.py 2>&1 | tee -a experiments/exp10_size_breakdown/logs/plot.log

# Exp 11: OOD repair
run_exp exp11_ood_repair "python experiments/exp11_ood_repair/run.py --rounds-list 0 50 100 250 500 1000 --seeds 0 1 2 --n-instances 6"
python experiments/exp11_ood_repair/plot.py 2>&1 | tee -a experiments/exp11_ood_repair/logs/plot.log

echo "[run_all] ALL DONE"
