# INRC-II Contextual Bandit Project — Instructions

**DS 592: Introduction to Sequential Decision Making | Spring 2026**
**Team:** Ria Sonalker · Jihyeon Yun · Achuthan Rathinam
**Proposal Due:** April 6, 2026

---

## Overview

Frame the [INRC-II](https://mobiz.vives.be/inrc2/) nurse rostering competition as a **contextual bandit problem**. Each week `t`, observe a context vector `x_t` (nurse history + demand), select one of `K` scheduling heuristics (arms), and receive a reward based on the quality of the resulting roster. Goal: learn a policy that minimises cumulative soft-constraint penalties over the 4- or 8-week planning horizon.

---

## Step 1 — Parse Input Files

Three files per instance:

| File | Format | Contents |
|------|--------|---------|
| Scenario | XML | Nurse IDs, contracts, skills, shift types, forbidden successions |
| Week data | XML | Daily coverage requirements (min + optimal), day-off requests |
| History | JSON | Per-nurse state: last shift, consecutive counters, total assignments, weekends worked |

**Key history fields:** `lastAssignedShiftType`, `numberOfConsecutiveWorkingDays`, `numberOfConsecutiveDaysOff`, `numberOfAssignments`, `numberOfWorkingWeekends`.

Parse into Python dataclasses. Validate nurse ID consistency across all three files before proceeding.

---

## Step 2 — Build Context Vector

At week `t`, construct `x_t = [x_nurse || x_week]`.

**Nurse-level** (aggregate mean/max/min across all nurses):
- `consecutiveWorkingDays / maxConsecutiveWorkingDays`
- `consecutiveDaysOff / minConsecutiveDaysOff`
- `numberOfWorkingWeekends / maxWorkingWeekends`
- `numberOfAssignments / maxAssignments`

**Week-level:**
- Total coverage demand per shift type
- Total day-off requests
- Week position: `t / W`
- Coverage slack (available nurses − min demand)

Normalise all features to `[0, 1]` using bounds fitted on **training instances only**.

---

## Step 3 — Define Heuristic Arms (K = 5)

Each arm is a function `(scenario, week_data, history) → feasible roster`.

| Arm | Strategy |
|-----|---------|
| **A1** Coverage-First Greedy | Fill largest coverage gaps first |
| **A2** Fatigue-Aware | Prioritise rest for nurses near consecutive-day limits |
| **A3** Weekend-Balancing | Minimise variance in weekends worked across nurses |
| **A4** Preference-Respecting | Maximise satisfied day-off requests |
| **A5** Lookahead-Conservative | Prefer nurses with the most remaining contract slack |

After generation, apply a **feasibility repair**: check forbidden successions and minimum coverage; greedily swap assignments to fix violations.

---

## Step 4 — Design Reward Signal

Per-week proxy reward (computed immediately after each roster):

```
r_t = -(λ1·p_cov + λ2·p_pref + λ3·p_consec + λ4·p_succ)
```

- `p_cov` — understaffing vs. optimal coverage
- `p_pref` — violated day-off requests
- `p_consec` — violated consecutive-work/rest soft limits
- `p_succ` — undesirable (non-forbidden) shift successions

At final week `W`, run the official Java validator and apply a correction:

```
r_W_adj = r_W − α · (P_total − Σ|r_t|)
```

---

## Step 5 — Implement LinUCB

For each arm `a`, maintain `A_a = I_d` and `b_a = 0`.

Each week:
1. Compute `θ_a = A_a⁻¹ · b_a`
2. Compute `UCB(a) = θ_a·x_t + α · sqrt(x_t·A_a⁻¹·x_t)`
3. Select `a_t = argmax UCB(a)`
4. Execute heuristic, observe `r_t`
5. Update `A_{a_t} += x_t x_t^T`, `b_{a_t} += r_t x_t`

Tune `α ∈ {0.1, 0.5, 1.0, 2.0, 5.0}` via cross-validation.

Extensions: Thompson Sampling (better in low-data regimes), Neural Contextual Bandit (non-linear context-reward).

---

## Step 6 — Train Across Instances

A single instance gives only W = 4 or 8 updates — not enough to learn. Train a **shared model** across M instances (one episode per instance). Split instances by scenario type into train / val / test. Fit normalisation on train only.

---

## Step 7 — Evaluate

**Primary metric:** `P_total` from the official validator (lower = better).

**Baselines:**
- Best single arm (uniform across all weeks)
- Random arm selection
- Round-robin
- Oracle (hindsight best arm — upper bound)

**Ablations:** context features, reward structure (proxy vs. terminal-only), arm count/composition.

---

## Step 8 — Implementation Stack

| Component | Tool |
|-----------|------|
| Data parsing | Python (`json`, `xmltodict`) |
| Features | `numpy`, `pandas` |
| Heuristics | Python (custom, one module per arm) |
| Bandit | `numpy` (LinUCB from scratch) |
| Validator | INRC-II official (Java) |
| Experiment tracking | `mlflow` or `wandb` |
| Tests | `pytest` |

---

## References

- Ceschia et al. (2015). *INRC-II Problem Description and Rules.* arXiv:1501.04177.
- Li et al. (2010). *A contextual-bandit approach to personalized news article recommendation.* WWW.
- Lattimore & Szepesvári (2020). *Bandit Algorithms.* Cambridge University Press.
