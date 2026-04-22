# Codebase guide

This document explains how the INRC-II contextual bandits project is structured, how modules connect, and how the three execution paths (week-level only, repair-level only, and chained week → repair) relate to each other.

## What the project is

The project frames the [Second International Nurse Rostering Competition (INRC-II)](https://www.kuleuven-kortrijk.be/nrpcompetition) as a contextual bandit problem. Each instance supplies:

- **`Sc-*.json`** — scenario (nurses, contracts, skills, shift types, forbidden successions)
- **`H0-*.json`** — starting history (per-nurse counters before week 0)
- **`WD-*.json`** — per-week coverage requirements and shift-off requests

The goal is to learn or select policies that produce rosters minimizing the INRC-II soft/hard penalty (implemented in code, not via the official `validator.jar` in this repo).

## Two bandit loops on shared scoring

The same `Schedule` representation and `compute_penalty` implementation underpin two different learning loops:

| Loop | Driver | Arms | Bandit | Reward |
|------|--------|------|--------|--------|
| **Week-level (constructive)** | `src/week_level/runner.py`, `src/week_level/train.py` | Four full-schedule heuristics: `coverage_first`, `fatigue_aware`, `weekend_balancing`, `preference_respecting` | Disjoint `LinUCB` with a 6-D context, or fixed selectors | `penalty_before − penalty_after` per week |
| **Repair-level (iterative)** | `src/repair_level/runner.py`, `src/main.py`, `src/tui.py` | ~14 auto-discovered `RepairStrategy` classes | Context-free: `uniform`, `epsilon_greedy`, `ucb1`, `thompson`, `softmax`, `exp3` | Penalty delta per applied repair |

## How the code is wired

```
              ┌──────────────────────────┐
              │  Dataset/...*.json files │
              └─────────────┬────────────┘
                            │ load_json
              ┌─────────────▼────────────┐
              │ instance_loader.py       │  → INRCInstance(scenario, history, weeks, ...)
              │ data/instances.py        │  → enumerate_instances(...) for train/dev/val/test splits
              └─────┬────────────┬───────┘
                    │            │
        ┌───────────▼─┐       ┌──▼──────────────┐
        │ week_level  │       │  repair_level   │
        │             │       │                 │
        │ context_    │       │  init.py (greedy│
        │  builder.py │       │   schedule)     │
        │  6-dim ctx  │       │                 │
        │             │       │  repairs/...    │
        │ arms/*.py   │       │   (~14 classes) │
        │  (4 arms)   │       │                 │
        │             │       │  runner.run_    │
        │ runner.run_ │       │  repairs (bandit│
        │  week_level │       │  loop, reward = │
        │  (bandit or │       │  Δpenalty)      │
        │  fixed sel.)│       └─────────┬───────┘
        │             │                 │
        │ train.py    │                 │
        │  cross-     │                 │
        │  instance   │                 │
        │  LinUCB     │                 │
        └──────┬──────┘                 │
               │                        │
               └──┬─────────────────────┘
                  │
         ┌────────▼───────────┐
         │ schedule/          │
         │  representation.py │   Schedule + coverage index
         │  penalty.py        │   PenaltyResult (H1–H4, S1–S7)
         └────────────────────┘
```

## Three ways to run the system

These coexist; they are not mutually exclusive “modes” of one binary.

### 1. Week-level pipeline only

- **Files:** `src/week_level/runner.py`, `src/week_level/train.py`
- **Behavior:** Each week, one arm builds that week’s assignments. Penalty is computed with `compute_penalty` over the horizon being filled. `train.py` streams many instances via `enumerate_instances`, updates a **shared** `LinUCB`, and saves a checkpoint (`.npz`).
- **Repair loop:** Never invoked.

### 2. Repair-level pipeline only

- **Files:** `src/main.py`, or `src/tui.py` with init source **greedy** (default)
- **Behavior:** `generate_initial_schedule` (`src/repair_level/init.py`) builds a starting schedule; callers pass that `Schedule` into `run_repairs`, which runs the bandit over repair strategies.
- **Week-level:** Not used.

### 3. Chained week-level → repair-level

- **Files:** `src/tui.py` (init source **week_level** + `week_mode`)
- **Behavior:** Run `run_week_level` first to produce a `Schedule`, then pass that schedule into `run_repairs` (same as greedy init: the runner always receives an explicit initial `Schedule`).

**TUI:** `week_mode` can use a single coverage-first arm, all four arms with fixed/random/LinUCB selection, etc. `src/main.py` is greedy init only; use the TUI for week-level init. Both support skipping repairs (`--no-repair` / **Run repairs** OFF).

**Hand-off:** Callers build the initial schedule (`generate_initial_schedule`, `run_week_level`, etc.) before `run_repairs` (`src/repair_level/runner.py`).

## Entry points

| Entry | Role |
|-------|------|
| `src/main.py` | Argparse: repair loop (greedy init), dataset/bandit/rounds/seed, artifact JSON under `runs/` |
| `src/tui.py` | Curses UI: repair loop plus optional week-level seeding (`week_mode`, LinUCB), timings, lazy scan |
| `src/week_level/train.py` | Cross-instance LinUCB training; writes checkpoint + metadata |
| `src/evaluate.py` | Pretty-printers: schedule grid, validator-style report |
| `tests/test_linucb.py` | Sanity test for disjoint LinUCB numerics |

Running `python src/week_level/runner.py` executes its `__main__` self-test and demo runs.

## Module reference (by area)

### Schedule and penalty

- **`src/schedule/representation.py`** — `Schedule`: nurse → `{global_day: (shift, skill)}` and `(day, shift, skill) → count`. `add_assignment` / `remove_assignment` keep structures consistent; H1 is structurally avoided. Global day = `week_idx * 7 + day_in_week`.
- **`src/schedule/penalty.py`** — H1–H4 and S1–S7 with competition weights; consecutive and border logic for INRC-II; `_compute_week_history` rolls history forward one week for multi-week soft penalties.

### Data loading

- **`src/instance_loader.py`** — Load one instance from dataset root + name + indices.
- **`src/data/instances.py`** — Discover scenarios across folder layouts; `enumerate_instances` yields `INRCInstance` with optional random week combinations (see “Known gaps” below).

### Repair level

- **`src/repair_level/init.py`** — Two-stage greedy init (H2 minimums, then toward S1 optimals); nurse choice respects H3/H4/H1 ordering.
- **`src/repair_level/repairs/base.py`** — `RepairStrategy`: `find_violations`, `apply`.
- **`src/repair_level/repairs/_helpers.py`** — Scenario lookups, H3 checks, runs, coverage helpers.
- **Strategies:** `coverage.py`, `consecutive_work.py`, `consecutive_shift.py`, `days_off.py`, `preference.py`, `total_assignments.py`, `weekend.py`, `catchall.py`. `__init__.py::build_all_strategies` instantiates subclasses by matching constructor parameter names (`scenario`, `history`/`initial_history`, `week_data_list`/`weeks`, `seed`).

### Repair runner and bandits

- **`src/repair_level/runner.py`** — `run_repairs(schedule=...)`: repair loop only (initial schedule supplied by caller). Eager mode (scan all strategies) vs **lazy** mode (`--lazy-find`: pick strategy name first, then scan one). Updates bandit with `penalty_before - penalty_after` when `update` exists. `RunResult` serializes to JSON.
- **`src/bandit/`** — `BanditSelector` groups candidates by name, calls `_pick`, samples a violation. `linucb.py` is for **week-level** disjoint LinUCB (`choose`, `save`/`load`). EXP3 overrides `__call__` for probability tracking (relevant to lazy mode — see issues).

### Week level

- **`src/week_level/context_builder.py`** — Six features in `[0, 1]`: coverage slack, mean fatigue ratio, weekend spread, request density, max assignment saturation, week position.
- **`src/week_level/arms/_common.py`** — Shared feasibility; `greedy_minimum_schedule` with pluggable nurse sort key.
- **`src/week_level/arms/*.py`** — Four arms differing only in sort keys (coverage-first, fatigue-aware, weekend balancing, preference-respecting).
- **`src/week_level/runner.py`** — Per week: build context from rolling history, pick arm (fixed or LinUCB), merge assignments, soft penalty before/after, LinUCB update scaled by `reward_scale`, advance history.

## Interop and naming

- Week-level arm output may use `"nurseId"`; `Schedule.from_solutions` may expect `"nurse"`. INRC-II `H0-*.json` uses `"nurse"`. Solving interop with official `Sol-*.json` may require key normalization.

## Known gaps, bugs, and spec mismatches

The following are documented for maintainers; they are not exhaustive of every possible edge case.

**High priority**

1. **`repair_level/init.py` — stale `prev_shift` between stage 1 and stage 2** — After stage 1, `prev_shift` can reflect the last day of the sweep, not the true predecessor for week 0 day 0 vs. history. H3 checks on the first days of stage 2 can be wrong. Fix: reset `prev_shift` from history each stage or derive from `schedule.shift(nid, gd - 1)` in `_pick_nurse`.
2. **Week-level reward vs. partial horizon** — `compute_penalty` scores the full passed horizon; empty future weeks can inflate S6/S7-style terms so per-week deltas conflate “filled this week” with “changed global shortage from empty weeks.”
3. **Lazy mode + EXP3** — Lazy path uses `_pick`, which is a stub for `EXP3Selector`; real sampling lives in `__call__`. Updates can be wrong or no-ops.
4. **`PullFromAdjacentDaySurplus.apply`** — May not re-validate that the slot is still under optimal at apply time; can worsen coverage if state changed.

**Design / spec (`ref_docs/INRC2_Project_Instructions.md`)**

- Fifth arm (“lookahead-conservative”) not implemented; context features and train-only normalization differ from the written proposal; rewards are raw penalty deltas without per-category λ weights or terminal `validator.jar` correction.

**Quality / ergonomics**

- `get_bandit(..., **kwargs)` can raise if kwargs are passed to selectors that do not accept them (`main.py` filters; `tui.py` uses defaults only).
- `enumerate_instances` may sample non-contiguous week index permutations — intentional augmentation should be documented; otherwise prefer contiguous horizons.
- `README.md` may not match current folders (`Dataset/` vs. `data/`, script locations).
- Test coverage beyond `test_linucb.py` is minimal.

## See also

- `ref_docs/INRC2_Project_Instructions.md` — course / proposal requirements
- `README.md` — project overview (verify paths against this guide)
