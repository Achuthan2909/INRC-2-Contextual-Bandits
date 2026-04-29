"""
Sample run: LinUCB bandit across 30 episodes on the n030w4 dataset.

Each episode = one week scheduling problem.
The bandit picks one of 5 strategies (arms), runs it, gets a reward,
and updates its model.

Episode pool: 3 history files × 10 week-demand files = 30 combinations.

Run from repo root:
  python experiments/bandit_sample/run_sample.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make src/ importable from any working directory
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from loader import load_instance
from context_builder import build_context
from coverage_first    import generate_schedule_coverage_first
from preference_first  import generate_schedule_preference_first
from fairness_first    import generate_schedule_fairness_first
from max_coverage      import generate_schedule_max_coverage
from random_baseline   import generate_schedule_random

from bandit  import LinUCB, ARM_NAMES
from reward  import compute_proxy_reward
from simulator import update_history_from_assignments

# ── Configuration ─────────────────────────────────────────────────────────────

DATASET_ROOT = REPO_ROOT / "Dataset" / "datasets_json"
DATASET_NAME = "n030w4"
N_HISTORIES  = 3    # H0-n030w4-0, -1, -2
N_WEEKS      = 10   # WD-n030w4-0 … WD-n030w4-9
TOTAL_WEEKS  = 4    # n030w4 is a 4-week scenario
ALPHA        = 1.0  # LinUCB exploration parameter

# Map arm index → strategy function
STRATEGIES = [
    generate_schedule_coverage_first,
    generate_schedule_preference_first,
    generate_schedule_fairness_first,
    generate_schedule_max_coverage,
    lambda sc, hi, wd: generate_schedule_random(sc, hi, wd, seed=42),
]

# ── Build episode list ─────────────────────────────────────────────────────────

episodes = []
for hist_idx in range(N_HISTORIES):
    for week_idx in range(N_WEEKS):
        episodes.append((hist_idx, week_idx))

# ── Load scenario once (shared across all episodes) ───────────────────────────

instance = load_instance(
    dataset_root=str(DATASET_ROOT),
    dataset_name=DATASET_NAME,
    history_idx=0,
    week_indices=list(range(N_WEEKS)),
)
scenario = instance.scenario

# Pre-load all histories and week data
import json
histories = [
    json.loads((DATASET_ROOT / DATASET_NAME / f"H0-{DATASET_NAME}-{i}.json").read_text())
    for i in range(N_HISTORIES)
]
week_datas = [
    json.loads((DATASET_ROOT / DATASET_NAME / f"WD-{DATASET_NAME}-{i}.json").read_text())
    for i in range(N_WEEKS)
]

# ── Initialise bandit ─────────────────────────────────────────────────────────

n_features = 8   # context_builder returns 8 features
bandit = LinUCB(n_arms=5, n_features=n_features, alpha=ALPHA)

# ── Track results ─────────────────────────────────────────────────────────────

arm_counts  = [0] * 5
arm_rewards = [0.0] * 5
history_log = []

print("=" * 65)
print(f"LinUCB Bandit Sample Run  —  {DATASET_NAME}  —  {len(episodes)} episodes")
print(f"Arms: {ARM_NAMES}")
print(f"Alpha (exploration): {ALPHA}")
print("=" * 65)
print(f"{'Ep':>3}  {'History':>7}  {'Week':>4}  {'Arm chosen':<20}  {'Reward':>8}  {'Uncov':>5}")
print("-" * 65)

for ep, (hist_idx, week_idx) in enumerate(episodes):
    history   = histories[hist_idx]
    week_data = week_datas[week_idx]

    # 1. Build context
    context, _ = build_context(
        scenario    = scenario,
        history     = history,
        week_data   = week_data,
        week_idx    = week_idx % TOTAL_WEEKS,
        total_weeks = TOTAL_WEEKS,
    )

    # Normalise context so LinUCB features are on similar scales
    context = context / (np.linalg.norm(context) + 1e-8)

    # 2. Bandit picks an arm
    arm = bandit.select_arm(context)

    # 3. Run the chosen strategy
    assignments, uncovered = STRATEGIES[arm](scenario, history, week_data)

    # 4. Compute proxy reward
    reward = compute_proxy_reward(assignments, uncovered, week_data)

    # 5. Update bandit model
    bandit.update(arm, context, reward)

    # 6. Log
    arm_counts[arm]  += 1
    arm_rewards[arm] += reward
    history_log.append({
        "episode":    ep + 1,
        "hist_idx":   hist_idx,
        "week_idx":   week_idx,
        "arm":        arm,
        "arm_name":   ARM_NAMES[arm],
        "reward":     reward,
        "n_assigned": len(assignments),
        "n_uncovered": len(uncovered),
    })

    print(f"{ep+1:>3}  H{hist_idx:<6}  WD{week_idx:<3}  "
          f"{ARM_NAMES[arm]:<20}  {reward:>8.1f}  {len(uncovered):>5}")

# ── Summary ───────────────────────────────────────────────────────────────────

print("=" * 65)
print("\nArm selection summary:")
print(f"  {'Strategy':<22} {'Times chosen':>13} {'Avg reward':>11}")
print(f"  {'-'*22} {'-'*13} {'-'*11}")
for i, name in enumerate(ARM_NAMES):
    times  = arm_counts[i]
    avg_r  = arm_rewards[i] / times if times > 0 else float("nan")
    print(f"  {name:<22} {times:>13} {avg_r:>11.1f}")

print(f"\nOverall average reward: "
      f"{sum(e['reward'] for e in history_log) / len(history_log):.1f}")

best_arm = int(np.argmax([
    arm_rewards[i] / arm_counts[i] if arm_counts[i] > 0 else float("-inf")
    for i in range(5)
]))
print(f"Best performing arm:    {ARM_NAMES[best_arm]}  "
      f"(avg reward {arm_rewards[best_arm]/arm_counts[best_arm]:.1f})")

print("\nNote: reward is a proxy (coverage + preference penalty).")
print("For official P_total, pipe solutions through tools/validator.jar.")
