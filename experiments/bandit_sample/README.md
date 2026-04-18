# Bandit Sample Run

A demonstration of a LinUCB contextual bandit choosing between five
scheduling strategies (arms) to minimise roster penalty on the `n030w4`
dataset.

---

## Files

| File | What it does |
|---|---|
| `bandit.py` | LinUCB algorithm — selects an arm, updates model after each episode |
| `reward.py` | Proxy reward function — computes negative penalty from assignments |
| `simulator.py` | Explains the Java simulator and provides a lightweight Python stand-in |
| `run_sample.py` | Main script — runs 30 episodes and prints results |

---

## How to run

From the repo root:

```bash
pip install numpy
python experiments/bandit_sample/run_sample.py
```

No Java required — uses the Python proxy reward.

---

## What the output looks like

```
=================================================================
LinUCB Bandit Sample Run  —  n030w4  —  30 episodes
Arms: ['coverage_first', 'preference_first', ...]
Alpha (exploration): 1.0
=================================================================
 Ep  History  Week  Arm chosen             Reward  Uncov
-----------------------------------------------------------------
  1  H0       WD0   coverage_first         -120.0      2
  2  H0       WD1   max_coverage             -90.0      1
  3  H0       WD2   preference_first         -45.0      0
...
=================================================================
Arm selection summary:
  Strategy                Times chosen  Avg reward
  coverage_first                     6       -98.0
  preference_first                   8       -52.0
  ...
Best performing arm:    preference_first  (avg reward -52.0)
```

---

## Episode structure

- **Dataset**: `n030w4` (30 nurses, 4-week scenario)
- **Episodes**: 30 total — 3 history files × 10 week-demand files
- **Each episode**:
  1. Build context vector (8 features from `context_builder.py`)
  2. Normalise context
  3. LinUCB picks an arm (strategy)
  4. Strategy runs → produces `assignments` + `uncovered` list
  5. Proxy reward computed
  6. LinUCB model updated

---

## The five arms

| Arm | Strategy | What it prioritises |
|---|---|---|
| 0 | `coverage_first` | Fill minimum staffing, lowest-penalty nurse first |
| 1 | `preference_first` | Protect nurse shift-off requests where possible |
| 2 | `fairness_first` | Assign least-worked nurses first |
| 3 | `max_coverage` | Fill to maximum staffing level |
| 4 | `random_baseline` | Random selection — lower-bound benchmark |

---

## The bandit (LinUCB)

**LinUCB** (Linear Upper Confidence Bound) maintains one linear model per arm.
For each arm `a` and context vector `x`:

```
theta_a = A_a^{-1} b_a          ← estimated reward weights
p_a = theta_a · x               ← expected reward
    + alpha * sqrt(x^T A_a^{-1} x)  ← exploration bonus
```

The arm with the highest `p_a` is selected. After observing reward `r`:

```
A_a += x x^T
b_a += r * x
```

**Alpha** controls exploration: high alpha = tries all arms more; low alpha = sticks with what works.

---

## The reward (proxy)

```
reward = -(lambda_cov * p_cov + lambda_pref * p_pref)
```

| Component | Meaning | Default weight |
|---|---|---|
| `p_cov` | Total nurse-shifts short across uncovered slots | 30 |
| `p_pref` | Shift-off request violations in assignments | 15 |

Higher reward = lower penalty = better schedule.

---

## The simulator

`tools/Simulator_withTimeout.jar` is the official INRC-2 Java simulator.
In production it:
1. Calls your scheduling program for each week sequentially
2. Updates nurse history between weeks (consecutive counters, last shift, etc.)
3. Scores the full 4-week solution via `tools/validator.jar` → official P_total

This sample run uses a **Python stand-in** (`simulator.py`) that updates
history approximately (last shift + assignment count only). For official
results, replace `run_sample.py`'s week loop with a subprocess call to the
Java simulator.

---

## Limitations of this sample

- 30 episodes is too few for LinUCB to fully converge — it's a demonstration
- Proxy reward omits consecutive-work and forbidden-succession penalties
- History update between weeks is approximate (see `simulator.py`)
- For real experiments, run across all dataset instances and use the Java validator
