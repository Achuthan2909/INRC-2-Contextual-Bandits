# Experiments

Each `expXX_*/` directory contains:
- `run.{py,sh}` — driver
- `plot.py` — figure generation
- `logs/` — `{stdout,stderr}.log`
- `results/` — JSON / JSONL / CSV artifacts
- `figures/` — PNGs

`_shared/` holds the base week + repair LinUCB checkpoints reused by Exp 02,
06, 09, 10, 11. Train them with `bash _shared/train_base.sh` (~12 min).

`run_all.sh` runs everything sequentially. Per-experiment logs are isolated
so partial failures don't poison neighbours.

## Index

- **exp02_bandit_grid** — 5×5 (week × repair) bandit grid at fixed budget
- **exp04_reward_scale** — sweep ρ at week and repair levels
- **exp05_warm_start** — sweep warm-start length
- **exp06_data_scaling** — learning curve vs # train instances
- **exp09_context_ablation** — mask one feature at a time, retrain
- **exp10_size_breakdown** — pivot Exp 02 results by scenario size
- **exp11_ood_repair** — repair LinUCB on greedy / mid / branch-and-price starts
